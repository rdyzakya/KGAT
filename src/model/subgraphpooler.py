import torch

class SubgraphPooler(torch.nn.Module):
    def __init__(self, text_emb_dim, hidden_dim, lm, graphpooler):
        super().__init__()
        self.graphpooler = graphpooler
        self.lm = lm # Part of AutoModel, not CausalLM or WithLMHead
        self.graph_emb_fuser = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=graphpooler.out_channels,
                            out_channels=hidden_dim,
                            kernel_size=1,
                            stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=hidden_dim,
                            out_channels=1,
                            kernel_size=1,
                            stride=1)
        )

        self.text_transform = torch.nn.Sequential(
            torch.nn.Linear(in_features=text_emb_dim,
                            out_features=hidden_dim,
                            bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim,
                            out_features=graphpooler.out_channels,
                            bias=True),
        )
    
    def forward(self, text_emb, edge_score, graph_emb, edge_batch):
        # text emb shape is (row, emb_dim)
        # transform text emb
        transformed_text_emb = self.text_transform(text_emb)
        # fuse word emb to graph_emb to create subgraph emb
        subgraph_emb = transformed_text_emb.unsqueeze(1) * graph_emb
        # fuse subgraph emb to edge_score
        transformed_subgraph_emb = self.graph_emb_fuser(subgraph_emb.transpose(1,2)).transpose(1,2)
        transformed_subgraph_emb = transformed_subgraph_emb[edge_batch]
        transformed_subgraph_emb = transformed_subgraph_emb.squeeze(-1)

        fused_score = edge_score * transformed_subgraph_emb
        mean_fused_score = fused_score.mean(-1)
        mean_fused_score = mean_fused_score.sigmoid()

        return mean_fused_score, subgraph_emb, edge_batch