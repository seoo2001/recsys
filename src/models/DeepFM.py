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


class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dims, batch_norm, dropout):
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, embed_dim in enumerate(mlp_dims):
            self.mlp.add_module(f'linear{idx}', nn.Linear(input_dim, embed_dim))
            if batch_norm:
                self.mlp.add_module(f'batchnorm{idx}', nn.BatchNorm1d(embed_dim))
            self.mlp.add_module(f'relu{idx}', nn.ReLU())
            if dropout > 0:
                self.mlp.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            input_dim = embed_dim
            
        self.mlp.add_module('output', nn.Linear(input_dim, 1))
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.mlp(x)
    

class DeepFM(nn.Module):
    def __init__(self, field_dims, embedding_dim, mlp_dims, batch_norm=True, dropout = 0.2):
        super().__init__()
        self.linear = FeatureLinear(field_dims)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.dnn = MLP(embedding_dim * len(field_dims), mlp_dims, batch_norm, dropout)
    
    def forward(self, x: torch.Tensor):
        linear_term = self.linear(x)
        
        embed = self.embedding(x)
        
        square_of_sum = torch.sum(embed, dim=1) ** 2
        sum_of_square = torch.sum(embed ** 2, dim=1)
        interaction_term = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        deep_out = self.dnn(embed.view(-1, embed.size(1) * embed.size(2))).squeeze(1)

        print("deepout: ", deep_out)
        
        output = linear_term + interaction_term + deep_out
        
        return torch.sigmoid(output.squeeze(1))