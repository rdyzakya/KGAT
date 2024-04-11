import torch

class SubgraphVTTransformation(torch.nn.Module):
    def __init__(self, subgraph_emb_dim, hidden_dim, word_emb_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=subgraph_emb_dim,
                            out_channels=hidden_dim,
                            kernel_size=1,
                            stride=1)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim,
                            out_channels=word_emb_dim,
                            kernel_size=1,
                            stride=1)
    
    def forward(self, x):
        x = self.conv1(x.transpose(1,2))
        x = x.relu()
        x = self.conv2(x).transpose(1,2)
        return x