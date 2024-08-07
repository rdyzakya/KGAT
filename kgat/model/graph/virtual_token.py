import torch
from torch_geometric.nn import AttentionalAggregation

class DoubleLinear(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_features, n_features, bias=True)
        self.linear2 = torch.nn.Linear(n_features, n_features, bias=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x.relu()
        x = self.linear2(x)
        return x

class ToSequence(torch.nn.Module):
    def __init__(self, n_features, n_virtual_token=1):
        super().__init__()
        self.linear = torch.nn.ModuleList([
            DoubleLinear(n_features=n_features) for _ in range(n_virtual_token)
        ])
    
    def forward(self, x):
        result = []
        for lin in self.linear:
            o = lin(x) # N * Hin
            result.append(o.unsqueeze(0))
        result = torch.vstack(result)
        result = result.transpose(0,1) # N * L * Hin
        return result

class VirtualToken(torch.nn.Module):
    def __init__(self, n_features, n_virtual_token=1, gate_nn=None):
        super().__init__()
        gate_nn = gate_nn or torch.nn.Linear(n_features, n_features, bias=True)
        nn = torch.nn.Linear(n_features, n_features, bias=True)
        self.aggregation = AttentionalAggregation(gate_nn=gate_nn, nn=nn)
        self.to_sequence = ToSequence(n_features, n_virtual_token)

        self.n_virtual_token = n_virtual_token
    
    def forward(self, x, batch_index=None):
        x = self.aggregation(x, index=batch_index) # N * Hin
        x = self.to_sequence(x) # N * L * Hin

        return x

# class ToSequence(torch.nn.Module):
#     def __init__(self, n_features, gate_nn, n_virtual_token=1):
#         super().__init__()
#         self.aggr = torch.nn.ModuleList([
#             AttentionalAggregation(gate_nn=gate_nn, nn=torch.nn.Linear(n_features, n_features, bias=True)) for _ in range(n_virtual_token)
#         ])
    
#     def forward(self, x, batch_index=None):
#         result = []
#         for aggr in self.aggr:
#             o = aggr(x, index=batch_index) # N * Hin
#             result.append(o.unsqueeze(1))
#         result = torch.hstack(result) # N * L * Hin
#         return result

# class VirtualToken(torch.nn.Module):
#     def __init__(self, n_features, n_virtual_token=1, gate_nn=None):
#         super().__init__()
#         gate_nn = gate_nn or torch.nn.Linear(n_features, n_features, bias=True)
#         # nn = torch.nn.Linear(n_features, n_features, bias=True)
#         # self.aggregation = AttentionalAggregation(gate_nn=gate_nn, nn=nn)
#         self.to_sequence = ToSequence(n_features, gate_nn, n_virtual_token)

#         self.n_virtual_token = n_virtual_token
    
#     def forward(self, x, batch_index=None):
#         # x = self.aggregation(x, index=batch_index) # N * Hin
#         x = self.to_sequence(x, batch_index) # N * L * Hin

#         return x

if __name__ == "__main__":
    n_virtual_token = 3
    n_features = 768

    virtual_token = VirtualToken(n_features, n_virtual_token)

    entities = torch.randn(16, n_features) # 16 entities
    relations = torch.randn(4, n_features) # 4 relations

    batch_index = torch.arange(16)
    batch_index = torch.tensor([
        0,0,0,0,
        1,1,1,1,
        2,2,2,2,
        3,3,3,3,
    ])

    out = virtual_token(entities, batch_index=batch_index)

    print("Number of params :", sum(p.numel() for p in virtual_token.parameters()))

    assert out.shape[0] == batch_index.max()+1
    assert out.shape[1] == n_virtual_token
    assert out.shape[2] == entities.shape[1]