import torch
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m = torch.max(logits,-1,keepdim=True).values
    # 假设输入形状：
    # logits     : (B, L, V)   # 模型原始输出，还没做 softmax
    # targets    : (B, L)      # 真实 token 的 id，每个位置一个整数
    # m          : (B, L, 1)   # 每个位置的最大 logit 值（用于数值稳定）
    shifted_logit = logits - m
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logit),-1))
    target_logits = torch.gather(logits,-1,index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    score = torch.mean(loss)

    return score