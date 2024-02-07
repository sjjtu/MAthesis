import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class ECGDataset(Dataset):
    def __init__(self, path, hb_type="A") -> None:
        super().__init__()
        self.df = pd.read_csv(path, index_col=0).astype(np.float32).to_numpy().tolist()
        self.y = ['A']*len(self.df) if type=="A" else ['N']*len(self.df) # if none then just label everything as normal

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        seq, label = torch.tensor(self.df[index]).unsqueeze(1).float(), self.y[index]

        return seq, label