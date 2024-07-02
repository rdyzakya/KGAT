import torch

class OuterProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.unsqueeze(-1) * x.unsqueeze(-2)
        return x

class RESCAL(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.outer_product = OuterProduct()
        self.gate_nn = torch.nn.Linear(dim, dim, bias=True)
        self.dim = dim
    
    def forward(self, entities, relations):
        # entities shape : N_entity * h_dim
        # relations shape : N_relation * h_dim
        n_entity = entities.shape[0]
        n_relations = relations.shape[0]

        scores = torch.zeros(n_entity, n_relations, n_entity, device=entities.device)

        entities = self.gate_nn(entities)
        relations = self.gate_nn(relations)

        relation_matrices = self.outer_product(relations)
    
        for r in range(n_relations):
            R_k = relation_matrices[r]
            scores[:, r, :] = torch.matmul(entities, R_k).matmul(entities.transpose(0,1))
        
        return scores

if __name__ == "__main__":
    entities = torch.randn(256, 768) # 256 entities with 768 dimension
    relation_matrices = torch.randn(16, 768, 768) # 16 relations with 768 * 768 dimension

    decoder = RESCAL()

    scores = decoder(entities, relation_matrices)

    assert scores.shape[0] == entities.shape[0]
    assert scores.shape[1] == relation_matrices.shape[0]
    assert scores.shape[2] == entities.shape[0]