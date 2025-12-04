import torch, copy

class EMA:
    def __init__(self, model, decay=0.9995, warmup_steps=200):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0

        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = self.decay

        if self.num_updates < self.warmup_steps:
            decay = 1 - (1 - self.decay) * (self.num_updates / self.warmup_steps)

        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())

        for name, p in model_params.items():
            if name not in ema_params: continue
            ep = ema_params[name]
            if not p.requires_grad: continue
            ep.data.mul_(decay).add_(p.data, alpha=1-decay)

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "updates": self.num_updates}

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd["ema"])
        self.num_updates = sd.get("updates", 0)
