from torch.utils.data import dataloader, dataset

class BaseDataset(dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
    

class PointwiseDataset(BaseDataset):
    def __init__(self):
        pass


class PairwiseDataset(BaseDataset):
    def __init__(self):
        pass