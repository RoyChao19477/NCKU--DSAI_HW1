import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

class df(Dataset):
    def __init__(self, train=True):
        df = pd.read_csv("data/1_dataset_elec.csv", index_col=0).drop(columns='date')

        if train:
            self.x = df.iloc[:-10]
            self.y = df.iloc[:-10]
        else:
            self.x = df.iloc[-10:]
            self.y = df.iloc[-10:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor( self.x.iloc[idx].values )[:-1]
        y = torch.tensor( self.y.iloc[idx].values )[-1]

        return x, y

class df_lstm(Dataset):
    def __init__(self, train=True):
        self.days = 7
        df = pd.read_csv("data/1_dataset_elec.csv", index_col=0).drop(columns='date')
        #df = pd.read_csv("data/3_dataset_outlier_elec.csv", index_col=0).drop(columns='date')
        
        if train:
            self.x = df[['supply', 'demand', 'industry', 'civil']].iloc[ :-10 - self.days]
            self.y = df['OR'].iloc[ :-10 - self.days]
        else:
            self.x = df[['supply', 'demand', 'industry', 'civil']].iloc[ -10 - self.days:]
            self.y = df['OR'].iloc[ -10 - self.days:]

    def __len__(self):
        return len(self.x) - self.days

    def __getitem__(self, idx):
        x = torch.tensor( self.x.iloc[idx:idx+self.days].values )
        y = torch.tensor( self.y.iloc[idx+self.days - 1] ).unsqueeze(0)
        #y = torch.tensor( self.y.iloc[idx-1:idx+self.days-1].values )
        # x = F.normalize(x, dim=1)

        return x, y

class df_lstm_v2(Dataset):
    def __init__(self, train=True):
        self.days = 7

        df = pd.read_csv("data/3_dataset_outlier_elec.csv", index_col=0).drop(columns='date')
        #df = pd.read_csv("data/3_dataset_outlier_elec.csv", index_col=0).drop(columns='date')
        

        if train:
            self.x = df[['supply', 'demand', 'industry', 'civil']].iloc[ :-10 - self.days]
            self.y = df['OR'].iloc[ :-10 - self.days]
        else:
            self.x = df[['supply', 'demand', 'industry', 'civil']].iloc[ -10 - self.days:]
            self.y = df['OR'].iloc[ -10 - self.days:]

    def __len__(self):
        return len(self.x) - self.days

    def __getitem__(self, idx):
        x = torch.tensor( self.x.iloc[idx:idx+self.days].values )
        y = torch.tensor( self.y.iloc[idx+self.days] ).unsqueeze(0)

        return x, y