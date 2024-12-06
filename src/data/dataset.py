from torch.utils.data import Dataset, DataLoader
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.users = np.array(data["user"])
        self.items = np.array(data["item"])
        self.target = np.array(data["rating"])
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        return {
            'user': self.users[index],
            'item': self.items[index],
            'target': self.target[index],
        }
    

class PointwiseDataset(BaseDataset):
    def __init__(self):
        pass


class PairwiseDataset(BaseDataset):
    def __init__(self):
        pass