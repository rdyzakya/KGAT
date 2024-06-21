import torch

class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, **kwargs):
        pass

def context_manager(train=True):
    if train:
        return EmptyContextManager()
    return torch.no_grad()