from torch.optim import Adam

from exp import ex


class Adam(Adam):
    @ex.capture()
    def __init__(self, model, lr, weight_decay):
        options = {
            'params': model.parameters(),
            'lr': lr,
            'weight_decay': weight_decay
        }
        super().__init__(**options)