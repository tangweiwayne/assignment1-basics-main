import math

def get_lr_cosine_schedule(t,alpha_max,alpha_min,Tw,Tc):
    if t < Tw:
        # 1. Warm-up 阶段
        return (t / Tw) * alpha_max
    elif t <= Tc:
        # 2. Cosine Annealing 阶段
        progress = (t - Tw) / (Tc - Tw)
        cosine_out = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_out * (alpha_max - alpha_min)
    else:
        # 3. Post-annealing 阶段
        return alpha_min