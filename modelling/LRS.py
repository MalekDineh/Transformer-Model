from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class TransformerScheduler(_LRScheduler):
    """
    Learning rate scheduler for the Transformer model
    """
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        return [lr for _ in self.base_lrs]

    def get_last_lr(self):
        return self.get_lr()

def get_optimizer(model):
    """
    Get the optimizer for the Transformer model
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1.0)
    return optimizer