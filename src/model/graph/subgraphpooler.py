import torch
import torch_geometric

class LinearSubgraphPooler(torch.nn.Module):
    def __init__(self, lm, graphpooler):
        self.graphpooler = graphpooler
        self.lm = lm
        self.linear = torch.nn.Linear(in_features=graphpooler.out_channels,
                                      out_features=1,
                                      bias=True)
