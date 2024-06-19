import torch

class RESCAL(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, entity_vectors, relation_matrices):
        # entities shape : N_entity * h_dim
        # relations shape : N_relation * h_dim * h_dim
        n_entity = entity_vectors.shape[0]
        n_relations = relation_matrices.shape[0]

        scores = torch.zeros(n_entity, n_relations, n_entity)
    
        for r in range(n_relations):
            R_k = relation_matrices[r]
            scores[:, r, :] = torch.matmul(entity_vectors, R_k).matmul(entity_vectors.transpose(0,1))
        
        return scores

if __name__ == "__main__":
    entities = torch.randn(256, 768) # 256 entities with 768 dimension
    relation_matrices = torch.randn(16, 768, 768) # 16 relations with 768 * 768 dimension

    decoder = RESCAL()

    scores = decoder(entities, relation_matrices)

    assert scores.shape[0] == entities.shape[0]
    assert scores.shape[1] == relation_matrices.shape[0]
    assert scores.shape[2] == entities.shape[0]