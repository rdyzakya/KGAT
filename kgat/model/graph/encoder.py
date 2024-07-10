import torch
from torch_geometric.nn import GATv2Conv, GATConv, TransformerConv

class GNNSequential(torch.nn.Sequential):
    def forward(self, x, edge_index, relations, relation_index):
        for module in self._modules.values():
            x, relations = module(x, edge_index, relations, relation_index)
        return x, relations

class GNNBlock(torch.nn.Module):
    def __init__(self, input_dim, out_dim, edge_dim, n_head=8, p=0.2, gnn_type="gatv2"):
        super().__init__()
        if gnn_type == "gatv2":
            self.gnn = GATv2Conv(input_dim, out_dim, heads=n_head, concat=False, dropout=p, edge_dim=edge_dim, add_self_loops=True)
        elif gnn_type == "gat":
            self.gnn = GATConv(input_dim, out_dim, heads=n_head, concat=False, dropout=p, edge_dim=edge_dim, add_self_loops=True)
        elif gnn_type == "unimp":
            self.gnn = TransformerConv(input_dim, out_dim, heads=n_head, concat=False, dropout=p, edge_dim=edge_dim)
        else:
            raise NotImplementedError("Only gat, gatv2, or unimp")
        # self.lin_edge = torch.nn.Linear(n_head * dim, dim)
        self.gnn_type = gnn_type
    
    def forward(self, x, edge_index, relations, relation_index):
        edge_attr = relations[relation_index]
        x = self.gnn(x, edge_index, edge_attr=edge_attr)
        
        # relations = self.gnn.lin_edge(relations)
        # relations = relations.relu()
        # relations = self.lin_edge(relations)
        return x, relations
    
class GraphEncoder(torch.nn.Module):
    def __init__(self, n_features, h_dim, n_head=8, p=0.0, n_layers=4, gnn_type="gatv2"):
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
            self.gnn = GNNBlock(n_features, n_features, n_features, n_head=n_head, p=p, gnn_type=gnn_type)
        elif n_layers == 2:
            self.gnn = GNNSequential(*[
                GNNBlock(n_features, h_dim, n_features, n_head=n_head, p=p, gnn_type=gnn_type),
                GNNBlock(h_dim, n_features, n_features, n_head=n_head, p=p, gnn_type=gnn_type)
            ])
        else:
            hidden_layer = [
                GNNBlock(h_dim, h_dim, n_features, n_head=n_head, p=p, gnn_type=gnn_type) for _ in range(n_layers-2)
            ]
            gnn = [GNNBlock(n_features, h_dim, n_features, n_head=n_head, p=p, gnn_type=gnn_type)] + hidden_layer + [GNNBlock(h_dim, n_features, n_features, n_head=n_head, p=p, gnn_type=gnn_type)]
            self.gnn = GNNSequential(*gnn)
        self.gnn_type = gnn_type

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