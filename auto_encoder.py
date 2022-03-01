import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import os
import pandas as pd
import argparse

argparser = argparse.ArgumentParser(description="Training Option")
argparser.add_argument("--in_features", "-i", type=int)
argparser.add_argument("--file", "-f", type=str)
args = argparser.parse_args()
in_features = args.in_features
input_file = args.file

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

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 1500
batch_size = 5
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MyDataset(input_file, in_features)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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


model = VAE(in_features)
if torch.cuda.is_available():
    model.to(device)

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE


optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        id, inp = data["id"], data["inp"]
        inp = inp.type(torch.FloatTensor)
        inp = Variable(inp)
        if torch.cuda.is_available():
            inp = inp.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inp)
        loss = loss_function(recon_batch, inp, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(inp),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.item() / len(inp)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    # if epoch % 10 == 0:
        # save = to_img(recon_batch.cpu().data)
        # print(recon_batch.cpu().data.shape)
        # save_image(save, './vae_img/image_{}.png'.format(epoch))
# number of features
# fold 1: 85
# fold 2: 86
# fold 3: 85
# fold 4: 86
# fold 5: 85
fold = input_file.split("_")[3]
torch.save(model.state_dict(), f'./pretrain_vae/vae_{fold}_full.pth')