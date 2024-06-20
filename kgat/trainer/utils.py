import torch

class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self):
        pass

def context_manager(train=True):
    if train:
        return torch.no_grad()
    return EmptyContextManager()