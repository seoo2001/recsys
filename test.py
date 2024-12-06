import pandas as pd
from src.models import *
from src.data.dataset import BaseDataset
from torch.utils.data import DataLoader
import torch

def main():
    data = pd.DataFrame({"user": [1, 2, 3, 4, 5, 6], "item": [1, 3, 2, 1, 4, 2], "rating": [0, 0, 1, 1, 1, 0]})
    dataset = BaseDataset(data)   
    feild_dims = [6, 6]
    model = FM(field_dims=feild_dims, embedding_dim=12)
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    for batch in dataloader:
        user, item = batch['user'], batch['item']
        user = user.view(-1, 1)
        item = item.view(-1, 1)
        input = torch.concat((user, item), dim=1)
        out = model(input)
        print("out:", out)

if __name__ == "__main__":
    main()