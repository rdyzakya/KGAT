import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    source https://github.com/pyg-team/pytorch_geometric/issues/92 :
    I evaluated this problem and see perfect reproducibility 
    on CPU when using the seed_everything code from you. 
    However, on GPU, we can not guarantee determinism because 
    we make heavy use of scatter ops to implement Graph Neural Networks 
    which are non-deterministic by nature on GPU.

    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)