from torch.optim.lr_scheduler import LambdaLR
def build_cosine_with_warmup(optimizer, total_steps, warmup_steps, min_lr_ratio=1e-2):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step+1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        import math
        cosine = 0.5*(1.0 + math.cos(math.pi*progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)
