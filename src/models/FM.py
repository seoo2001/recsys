import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureLinear(nn.Module):
    def __init__(self, field_dims, out_dim=1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim,))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FM(nn.Module):
    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.linear = FeatureLinear(field_dims)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
    
    def forward(self, x: torch.Tensor):
        linear_term = self.linear(x)
        
        embed = self.embedding(x)
        
        square_of_sum = torch.sum(embed, dim=1) ** 2
        sum_of_square = torch.sum(embed ** 2, dim=1)
        interaction_term = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        output = linear_term + interaction_term
        
        return torch.sigmoid(output.squeeze(1))