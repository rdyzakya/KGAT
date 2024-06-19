import torch
from torch_geometric.nn import GATv2Conv

class Injector(torch.nn.Module):
    def __init__(self, input_dim, n_head=1, p=0.2):
        super().__init__()
        self.attention = GATv2Conv(input_dim, input_dim, heads=n_head, concat=False, dropout=p, add_self_loops=False, edge_dim=input_dim)
        # self.sum = SumAggregation()
        self.lin_edge = torch.nn.Linear(n_head * input_dim, input_dim, bias=True)
    
    def forward(self, queries, entities, edge_index, relations, relation_index, batch):

        assert batch.max()+1 == queries.shape[0], "Batch number should be equal to number of queries"
        # add the query node
        node_features = torch.vstack([entities, queries])
        # add new general relation type
        relation_features = torch.vstack([relations, torch.ones((1, relations.shape[1]))])
        # add connection between the query node and all other node (batch needed to know which)
        src_index = batch + entities.shape[0]
        tgt_index = torch.arange(0, entities.shape[0])
        added_edge_index = torch.vstack([src_index, tgt_index])

        added_relation_index = torch.full_like(batch, fill_value=relation_features.shape[0]-1) # general relation

        new_edge_index = torch.hstack([edge_index, added_edge_index])
        new_relation_index = torch.cat([relation_index, added_relation_index])

        out_node = self.attention(node_features, 
                                       new_edge_index,
                                       edge_attr=relation_features[new_relation_index])
        out_node = out_node[:entities.shape[0]]

        out_edge = self.attention.lin_edge(relations)
        out_edge = out_edge.relu()
        out_edge = self.lin_edge(out_edge)

        return out_node, out_edge



if __name__ == "__main__":
    input_dim = 768
    h_dim = 1024
    n_head=2

    queries = torch.randn(8, input_dim) # 8 queries with 768 dimension
    entities = torch.randn(256, input_dim) # 256 entities with 768 dimension
    relations = torch.randn(16, input_dim) # 16 relation types with 768 dimension

    injector = Injector(input_dim=input_dim,
                                   n_head=n_head,
                                   p=0.0)
    
    edge_index = torch.randint(256, (2,64))
    relation_index = torch.randint(16, (64,))
    batch = torch.arange(0, 8).repeat(256//8)
    batch, _ = batch.sort()
    
    out_node, out_edge = injector(queries, 
                                   entities, 
                                   edge_index, 
                                   relations, 
                                   relation_index, 
                                   batch)
    
    print("Number of params :", sum(p.numel() for p in injector.parameters()))

    assert out_node.shape == entities.shape, f"{out_node.shape} | {entities.shape}"
    assert out_edge.shape == relations.shape, f"{out_edge.shape} | {relations.shape}"