import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

device = torch.device("cuda:1")


class MyDataset(Dataset):
    def __init__(self, input_excel_file, in_features):
        # self.data = pd.read_excel(input_excel_file)
        # self.data = self.data.drop(["methyl_code"], axis=1)
        # self.data = self.data.T
        # self.data.index.name = "id"
        # self.ids = self.data.index.values
        # self.x = self.data[list(range(in_features))].to_numpy()
        self.data = pd.read_csv(input_excel_file)
        self.ids = self.data.id.values
        self.x = self.data[list(map(str, list(map(str, list(range(in_features))))))].to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inp = torch.from_numpy(self.x[item])
        id = self.ids[item]
        return {"id": id, "inp": inp}

class VAE(nn.Module):
    def __init__(self, in_features):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_features, 500)
        self.fc21 = nn.Linear(500, 100)
        self.fc22 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 500)
        self.fc4 = nn.Linear(500, in_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_().to(device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == '__main__':
    for i in range(1, 6):
        minmaxscaler = MinMaxScaler()
        input_file = f"train_data_fold_{i}_with_pca_from_full_mms.csv"
        # if i == 1:
        #     in_features = 85
        # elif i == 2:
        #     in_features = 86
        # elif i == 3:
        #     in_features = 85
        # elif i == 4:
        #     in_features = 86
        # else:
        in_features = 443612
        batch_size = 1
        model = VAE(in_features)
        model.load_state_dict(torch.load(f"./pretrain_vae/vae_{i}_full.pth"))
        model.eval()
        model = model.to(device)
        # train
        dataset = MyDataset(input_file, in_features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        col_names = ["id"] + list(range(in_features))
        train_data = pd.DataFrame(columns=col_names)
        for batch_idx, data in enumerate(dataloader):
            id, inp = data["id"], data["inp"]
            inp = inp.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inp = inp.to(device)
            recon_batch, mu, logvar = model(inp)
            recon_batch_data = recon_batch.cpu().data
            data = id + torch.squeeze(recon_batch_data).tolist()
            train_data = train_data.append(pd.DataFrame([data], columns=col_names))
        # test
        # if i == 1:
        #     test_data_file = f"Testset1.csv"
        # else:
        test_data_file = f"test_data_fold_{i}_with_pca_from_full_mms.csv"
        dataset = MyDataset(test_data_file, in_features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        col_names = ["id"] + list(range(in_features))
        test_data = pd.DataFrame(columns=col_names)
        for batch_idx, data in enumerate(dataloader):
            id, inp = data["id"], data["inp"]
            inp = inp.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inp = inp.to(device)
            recon_batch, mu, logvar = model(inp)
            recon_batch_data = recon_batch.cpu().data
            data = id + torch.squeeze(recon_batch_data).tolist()
            test_data = test_data.append(pd.DataFrame([data], columns=col_names))
        minmaxscaler.fit(train_data[list(range(in_features))])
        train_data[list(range(in_features))] = minmaxscaler.transform(train_data[list(range(in_features))])
        test_data[list(range(in_features))] = minmaxscaler.transform(test_data[list(range(in_features))])
        train_data.to_csv(f"train_data_fold_{i}_with_pca_from_full_mms_vae.csv", index=False)
        output_df = pd.read_csv("Output.csv")
        new_test_data = test_data.merge(output_df, how="inner", on="id")
        new_test_data.to_csv(f"test_data_fold_{i}_with_pca_from_full_mms_vae.csv", index=False)