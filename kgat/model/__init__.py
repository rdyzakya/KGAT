from .graph import (
    GraphEncoderDecoder,
    SubgraphGenerator,
    VirtualTokenGenerator
)

from .text import load_model_lmkbc

import torch

class Pipeline(torch.nn.Module):
    def __init__(self, **modules):
        super().__init__()
        for k, v in modules.items():
            if isinstance(v, torch.nn.Module):
                self.__setattr__(k, v)