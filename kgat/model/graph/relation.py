import torch

class ReshapeRelation(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.w1 = torch.nn.Linear(input_dim, h_dim, bias=True)
        # self.w2 = torch.nn.Linear(h_dim, input_dim*input_dim, bias=True)
        self.lin = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.dim = input_dim
    
    def forward(self, x):
        # x = self.w1(x)
        # x = self.w2(x)
        # x = x.view(-1, self.dim, self.dim)
        x = self.lin(x)
        x = x.unsqueeze(-1) * x.unsqueeze(-2) # outer product
        return x

if __name__ == "__main__":
    input_dim = 768
    # h_dim = 64
    relations = torch.randn(16, input_dim)

    reshaper = ReshapeRelation(input_dim=input_dim)

    relation_matrices = reshaper(relations)

    print("Number of params :", sum(p.numel() for p in reshaper.parameters()))

    assert relation_matrices.shape[0] == relations.shape[0]
    assert relation_matrices.shape[1] == relations.shape[1]
    assert relation_matrices.shape[2] == relations.shape[1]