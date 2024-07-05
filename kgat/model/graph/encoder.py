import torch
from torch_geometric.nn import GATv2Conv

class GNNSequential(torch.nn.Sequential):
    def forward(self, x, edge_index, relations, relation_index):
        for module in self._modules.values():
            x, relations = module(x, edge_index, relations, relation_index)
        return x, relations

class GATBlock(torch.nn.Module):
    def __init__(self, input_dim, out_dim, edge_dim, n_head=8, p=0.2):
        super().__init__()
        self.gnn = GATv2Conv(input_dim, out_dim, heads=n_head, concat=False, dropout=p, edge_dim=edge_dim, add_self_loops=True)
        # self.lin_edge = torch.nn.Linear(n_head * dim, dim)
    
    def forward(self, x, edge_index, relations, relation_index):
        edge_attr = relations[relation_index]
        x = self.gnn(x, edge_index, edge_attr=edge_attr)
        
        # relations = self.gnn.lin_edge(relations)
        # relations = relations.relu()
        # relations = self.lin_edge(relations)
        return x, relations
    
class GraphEncoder(torch.nn.Module):
    def __init__(self, n_features, h_dim, n_head=8, p=0.0, n_layers=4):
        super().__init__()
        # self.input_dim = input_dim
        # self.h_dim = h_dim
        # self.out_dim = out_dim
        self.n_features = n_features
        self.h_dim = h_dim
        # self.edge_dim = edge_dim
        self.n_head = n_head
        self.p = p
        self.n_layers = n_layers
        
        if n_layers == 1:
            self.gnn = GATBlock(n_features, n_features, n_features, n_head=n_head, p=p)
        elif n_layers == 2:
            self.gnn = GNNSequential(*[
                GATBlock(n_features, h_dim, n_features, n_head=n_head, p=p),
                GATBlock(h_dim, n_features, n_features, n_head=n_head, p=p)
            ])
        else:
            hidden_layer = [
                GATBlock(h_dim, h_dim, n_features, n_head=n_head, p=p) for _ in range(n_layers-2)
            ]
            gnn = [GATBlock(n_features, h_dim, n_features, n_head=n_head, p=p)] + hidden_layer + [GATBlock(h_dim, n_features, n_features, n_head=n_head, p=p)]
            self.gnn = GNNSequential(*gnn)

        # gnn = [
        #     GATBlock(dim, n_head=n_head, p=p) for _ in range(n_layers)
        # ]
        # self.gnn = GNNSequential(*gnn)
    
    def forward(self, x, edge_index, relations, relation_index):
        x, relations = self.gnn(x, edge_index, relations, relation_index)
        return x, relations

if __name__ == "__main__":
    # input_dim = 768
    # h_dim = 1024
    # out_dim = 768
    # dim = 768
    n_features = 4096
    h_dim = 128
    # edge_dim = 768
    n_head = 8
    p = 0.2
    n_layers = 8

    encoder = GraphEncoder(n_features, h_dim, n_head=n_head, p=p, n_layers=n_layers)

    print("Number of params :", sum(p.numel() for p in encoder.parameters()))

    entities = torch.randn(256, n_features)
    relations = torch.randn(16, n_features)

    edge_index = torch.randint(256, (2, 64))
    relation_index = torch.randint(16, (64,))

    out = encoder(entities, edge_index, relations, relation_index)

    assert out[0].shape[0] == entities.shape[0], f"{out[0].shape} | {entities.shape}"
    assert out[0].shape[1] == n_features, f"{out[0].shape} | {n_features}"
    assert out[1].shape[0] == relations.shape[0], f"{out[1].shape} | {relations.shape}"
    assert out[1].shape[1] == n_features, f"{out[1].shape} | {n_features}"