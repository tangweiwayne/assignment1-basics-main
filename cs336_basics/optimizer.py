import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 1. 获取梯度
                grad = p.grad.data
                
                # 2. 初始化或获取状态 (m, v, t)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # m
                    state['exp_avg_sq'] = torch.zeros_like(p.data) # v

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # m = β1 * m + (1 - β1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = β2 * v + (1 - β2) * g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4. 计算偏差修正后的学习率 (Bias Correction)
                # 注意：算法描述中 αt = α * sqrt(1-β2^t) / (1-β1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                # θ = θ - step_size * (m / (sqrt(v) + eps))
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 6. 应用解耦的权重衰减 (Decoupled Weight Decay)
                # θ = θ - α * λ * θ
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

        return loss