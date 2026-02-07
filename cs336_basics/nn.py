import torch
import torch.nn as nn
import einops
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i->... o",x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = float(1)
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1,keepdim=True)
        rms = torch.sqrt(ms+self.eps)
        result = (x/rms)*self.weight

        return result.to(in_dtype)

def silu_fu(in_features):

    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff= d_ff
        self.w1 = Linear(self.d_model,self.d_ff,device,dtype)
        self.w3 = Linear(self.d_model,self.d_ff,device,dtype)

        self.w2 = Linear(self.d_ff,self.d_model,device,dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        glu = silu_fu(self.w1(x)) * self.w3(x)
        return self.w2(glu)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k =d_k
        self.max_seq_len = max_seq_len
        # 1. 计算频率频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        exponent = torch.arange(0,d_k,2,device=device,dtype=torch.float32) /d_k
        freq = 1/(theta**exponent)  # 形状: (d_k/2,)

        # 2. 创建位置序列 [0, 1, ..., max_seq_len - 1]
        t = torch.arange(0,max_seq_len,device=device) # 形状: (max_seq_len,)

        # 3. 计算所有位置的所有角度 (外积)
        # freqs形状: (max_seq_len, d_k/2)
        freqs = torch.outer(t,freq)
        # 最终形状 [max_seq_len, d_k]
        freq_cos = einops.rearrange(torch.stack((torch.cos(freqs),torch.cos(freqs)),dim=-1)," ... d j ->... (d j)")
        freq_sin = einops.rearrange(torch.stack((torch.sin(freqs),torch.sin(freqs)),dim=-1)," ... d j ->... (d j)")
        self.register_buffer('freq_cos',freq_cos,persistent=False)
        self.register_buffer('freq_sin',freq_sin,persistent=False)

    def rotate_half(self,x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[...,1::2]
        rotate_x = torch.stack((-x_odd,x_even),dim=-1)
        return einops.rearrange(rotate_x," ... d j -> ... (d j)")
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        freqs_cos = self.freq_cos[token_positions]
        freqs_sin = self.freq_sin[token_positions]
        # token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        return x * freqs_cos + self.rotate_half(x) * freqs_sin
        # 对 token_positions 中每个数字
        # 去 freq_cos 里查一行
        # 再按 token_positions 的形状摆回去

def softmax(x:torch.Tensor, dim:int) ->torch.Tensor:
    x_max = torch.max(x,dim,keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x,dim,keepdim=True)
    return exp_x/sum_exp

def scaled_dot_product_attention(Q,K,V,mask) ->torch.Tensor:
    d_k = Q.shape[-1]
    attention_score = einops.einsum(Q,K, " ... n d_k, ... m d_k -> ... n m")/math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.masked_fill(attention_score,~mask,float('-inf'))
    else:
        attention_scores = attention_score
    product = einops.einsum(softmax(attention_scores,-1),V," ... n m, ... m d_v -> ... n d_v")
    return product

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,theta,max_seq_len,device= None,dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta,self.d_k,max_seq_len)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # x = [b s d_model]
        b, s ,d_model = x.size()
        q = einops.rearrange(self.q_proj(x), "... seq (heads head_dim) -> ... heads seq head_dim",heads = self.num_heads)
        k = einops.rearrange(self.k_proj(x), "... seq (heads head_dim) -> ... heads seq head_dim",heads = self.num_heads) 
        v = einops.rearrange(self.v_proj(x), "... seq (heads head_dim) -> ... heads seq head_dim",heads = self.num_heads)
        # q_proj = [b s d_model]
        # q = [b h s d_k]
        if self.rope is not None:
            if token_positions is None:
                pos = torch.arange(s,device=x.device)
                token_positions = pos.expand(x.shape[:-1])
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)
        
        q_len = q.size(-2)
        k_len = k.size(-2)

        if q_len == 1:
            # 推理阶段：只需要 key 维 mask
            mask = torch.ones(
                1, k_len,
                device=x.device,
                dtype=torch.bool
            )
        else:
            # 训练阶段：标准 causal mask
            mask = torch.tril(
                torch.ones(q_len, k_len, device=x.device, dtype=torch.bool),
                0
            )
        out = scaled_dot_product_attention(q,k,v,mask)
        # out = [b h s d_v]
        result = self.output_proj(einops.rearrange(out, "... h s d_v -> ... s (h d_v)"))
        return result

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta, 
                device=None, dtype=None):
        super().__init__()
        self.d_model =d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.max_seq_len =max_seq_len
        self.theta = theta

        self.norm_pre_attention = RMSnorm(d_model)
        self.attention = CausalSelfAttention(d_model,num_heads,theta,max_seq_len)
        self.norm_pre_mlp = RMSnorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
    def forward(self,x: torch.Tensor):
        
        x_norm = self.norm_pre_attention(x)
        x = x + self.attention(x_norm)
        x_norm = self.norm_pre_mlp(x)
        x = x + self.mlp(x_norm)
        return x
    
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta, 
                device=None, dtype=None, use_rms_norm=True, norm_mode='pre', ffn_type='swiglu'):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,device,dtype) for _ in range(num_layers)])

        self.post_norm = RMSnorm(d_model,device=device, dtype=dtype)
        self.lm_head = Linear(d_model,vocab_size,device=device, dtype=dtype)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.post_norm(x)
        x = self.lm_head(x)
        return x
    
    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int, 
        eos_token_id: int = None, 
        temperature: float = 1.0, 
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        从模型生成文本 ID 序列。
        
        参数:
            prompt_ids: 提示词 ID (Batch, Seq_len)
            max_new_tokens: 最多生成的词数
            eos_token_id: 停止生成的 Token ID (如 <|endoftext|>)
            temperature: 温度系数 (越高越随机，越低越确定)
            top_p: 核采样阈值
        """
        # 设置为评估模式
        self.eval()
        
        # 将输入拷贝一份，避免修改原始数据
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # 1. 裁剪输入：模型只能处理 context_length 长度的内容
            # 如果生成的序列过长，只取最后的 context_length 个词
            idx_cond = generated[:, -self.context_length:]
            
            # 2. 前向传播得到 Logits
            # 我们只关心最后一个时间步的预测
            logits = self.forward(idx_cond) # (Batch, T, Vocab)
            logits = logits[:, -1, :]      # (Batch, Vocab)
            
            # 3. 应用温度 (Temperature)
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8) # 加个 epsilon 防止除以 0
            
            # 4. 应用 Top-P (Nucleus Sampling) 过滤
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)
            
            # 5. 归一化并采样
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            
            # 6. 拼接新词
            generated = torch.cat((generated, next_token), dim=1)
            
            # 7. 如果遇到了 EOS，提前结束生成
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数：执行 Top-P 截断"""
        # 对词表分值进行降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 计算累积概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        
        # 创建掩码：我们要去掉累积概率超过 p 的 Token
        # 逻辑：保留最小的集合 V(p)，使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True（需要移除）
        sorted_indices_to_remove = cumulative_probs > p
        
        # 关键修正：确保至少保留第一个词（最高概率词），
        # 并且我们要保留第一个“使概率超过 p”的那个词。
        # 做法是把标记位向右移动一格。
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove = indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits