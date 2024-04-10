import torch
from torch_geometric.nn import conv
import torch.nn.functional as F

class SigmoidGATConv(conv.GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def edge_update(self, alpha_j, alpha_i,
                    edge_attr, index, ptr,
                    dim_size):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.sigmoid(alpha)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha