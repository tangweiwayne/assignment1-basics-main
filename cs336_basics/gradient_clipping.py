import torch

def gradient_clipping_impl(parameters, max_norm):
    epsilon = 1e-6
    # 1. 计算所有参数梯度的 L2 范数
    # 注意：需要把所有梯度的平方求和后再开方
    total_norm = 0.0
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    for p in params_with_grad:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # 2. 如果范数超过阈值，进行缩放
    if total_norm > max_norm:
        clip_coeff = max_norm / (total_norm + epsilon)
        for p in params_with_grad:
            p.grad.data.mul_(clip_coeff) # 原地乘法