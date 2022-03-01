from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, input_csv_file, start=None, end=None):
        self.data = pd.read_csv(input_csv_file)
        self.x = self.data.drop(["Output", "id"], axis=1).to_numpy()
        self.y = np.expand_dims(self.data["Output"].to_numpy(), axis=1)
        # if start == None or end == None:
        #     self.x = data
        #     self.y = output
        # else:
        #     self.x = data[start:end]
        #     self.y = output[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inp = torch.from_numpy(self.x[item])
        out = torch.from_numpy(self.y[item])
        #if self.y[item] == 0:
        #    out = torch.Tensor([1, 0])
        #else:
        #    out = torch.Tensor([0, 1])
        # inp = torch.from_numpy(np.float32(self.data.iloc[0].drop(["Output", "id"]).to_numpy()))
        # out = torch.tensor(self.data.iloc[0]["Output"]).unsqueeze_(0)
        return {"inp": inp, "out": out}

class NewMyDataset(Dataset):
    def __init__(self, input_csv_file, output_csv_file):
        self.x = np.load(input_csv_file)
        self.y = np.load(output_csv_file)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        inp = torch.from_numpy(self.x[item])
        out = torch.from_numpy(np.array([self.y[item]]))

        return {"inp": inp, "out": out}

class DS(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.x = df.drop(["Output"], axis=1)
        self.y = df["Ouput"]