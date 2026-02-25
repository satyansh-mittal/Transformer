class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = (self.d_model ** -0.5) * min(self._step ** -0.5,
                                         self._step * (self.warmup ** -1.5))
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr