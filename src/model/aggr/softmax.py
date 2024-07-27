import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class SoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::
        \mathrm{softmax}(\mathcal{X}|t) = \sum_{\mathbf{x}_i\in\mathcal{X}}
        \frac{\exp(t\cdot\mathbf{x}_i)}{\sum_{\mathbf{x}_j\in\mathcal{X}}
        \exp(t\cdot\mathbf{x}_j)}\cdot\mathbf{x}_{i},

    where :math:`t` controls the softness of the softmax when aggregating over
    a set of features :math:`\mathcal{X}`.

    Args:
        t (float, optional): Initial inverse temperature for softmax
            aggregation. (default: :obj:`1.0`)
        learn (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`t` for softmax aggregation dynamically.
            (default: :obj:`False`)
        semi_grad (bool, optional): If set to :obj:`True`, will turn off
            gradient calculation during softmax computation. Therefore, only
            semi-gradients are used during backpropagation. Useful for saving
            memory and accelerating backward computation when :obj:`t` is not
            learnable. (default: :obj:`False`)
        channels (int, optional): Number of channels to learn from :math:`t`.
            If set to a value greater than :obj:`1`, :math:`t` will be learned
            per input feature channel. This requires compatible shapes for the
            input to the forward calculation. (default: :obj:`1`)
    """
    def __init__(self, t: float = 1.0, learn: bool = False,
                 semi_grad: bool = False, channels: int = 1):
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_t = t
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        self.t = Parameter(torch.empty(channels)) if learn else t
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_t)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2,
                return_alpha: bool = False) -> Tensor:

        t = self.t
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            assert isinstance(t, Tensor)
            t = t.view(-1, self.channels)

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = x * t

        if not self.learn and self.semi_grad:
            with torch.no_grad():
                alpha = softmax(alpha, index, ptr, dim_size, dim)
        else:
            alpha = softmax(alpha, index, ptr, dim_size, dim)
        out = self.reduce(x * alpha, index, ptr, dim_size, dim, reduce='sum')
        
        if return_alpha:
            return out, alpha
        
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')