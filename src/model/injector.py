import torch

class Injector(torch.nn.Module):
    def __init__(self, edge_dim):
        super().__init__()
        self.edge_attr = torch.nn.parameter.Parameter(torch.randn(edge_dim))
    
    def forward(self, x, edge_index, relations, injection_node, node_batch=None, injection_node_batch=None):
        # node batch : [0,0,0,0,1,1,1]
        # injection node : [0,1] atau [0,0] atau [1,1] ,dst
        node_batch = torch.zeros(x.shape[0]) if node_batch is None else node_batch
        injection_node_batch = torch.arange(0, injection_node.shape[0]) if injection_node_batch is None else injection_node_batch
        
        x_out = torch.cat([x, injection_node], dim=0)
        relations_out = torch.cat([relations, self.edge_attr.unsqueeze(0)], dim=0)
        
        src_index = injection_node_batch[node_batch] + x.shape[0]
        rel_index = torch.full((x.shape[0],), fill_value=relations.shape[0], dtype=edge_index.dtype)
        tgt_index = torch.arange(0, x.shape[0])

        edge_index_out = torch.cat([edge_index, torch.stack([src_index, rel_index, tgt_index])], dim=1)

        x_is_injected = torch.cat([torch.zeros(x.shape[0]), torch.ones(injection_node.shape[0])], dim=0).int()
        edge_is_injected = torch.cat([torch.zeros(edge_index.shape[1]), torch.ones(x.shape[0])], dim=0).int()
        relations_is_injected = torch.cat([torch.zeros(relations.shape[0]), torch.ones(1)], dim=0).int() # only add 1 relation

        return x_out, edge_index_out, relations_out, x_is_injected, edge_is_injected, relations_is_injected

class Detach(torch.nn.Module):
    def forward(self, x, edge_index, relations, x_is_injected, edge_is_injected, relations_is_injected):
        x_not_injected = x_is_injected.logical_not()
        edge_not_injected = edge_is_injected.logical_not()
        relations_not_injected = relations_is_injected.logical_not()

        return x[x_not_injected], edge_index[:,edge_not_injected], relations[relations_not_injected]