import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from my_dataset import MyDataset
from torch.utils.data import SubsetRandomSampler
from torch import optim
from datetime import datetime

class NewModel(nn.Module):

    def __init__(self):
        super(NewModel, self).__init__()
        self.linear = nn.Linear(444417, 1000)
        self.batch = nn.BatchNorm1d(1000)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(1000, 100)
        self.batch_2 = nn.BatchNorm1d(100)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        x = x.type(dtype=torch.FloatTensor)
        x = x.to(torch.device("cuda:0"))
        out = self.linear(x)
        out = self.batch(out)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.batch_2(out)
        out = self.relu_2(out)
        out = self.linear_3(out)
        return self.sigmoid(out)[:, 0]
        # print(out.shape)

if __name__ == '__main__':
    import pandas as pd

    number_of_epochs = 200

    k_fold = KFold(n_splits=4, shuffle=True)

    train_data = MyDataset("msigcpg74420.csv",
                           "output.csv", 0, 110)
    val_data = MyDataset("msigcpg74420.csv",
                           "output.csv", 110, 120)
    test_data = MyDataset("msigcpg74420.csv",
                          "output.csv", 120, 150)
    device = torch.device("cuda:0")
    model = NewModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_func = nn.BCELoss()

    best_accu = 0
    best_model = ""

    #for fold, (train_ids, test_ids) in enumerate(k_fold.split(train_data)):
    #train_subsampler = SubsetRandomSampler(train_ids)
    #test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
    print(len(train_loader))
    print(len(val_loader))
    for epoch in range(number_of_epochs):
        model.train()
        training_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            inp = data["inp"].to(device, dtype=torch.float)
            output = data["out"].to(device, dtype=torch.float)
            # inp = torch.unsqueeze(inp, 1)
            predict = model(inp)
            # output = torch.squeeze(output)
            # print(predict.shape)
            # print(output.shape)
            loss = loss_func(predict, output)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        #print(f" {epoch + 1} / {number_of_epochs} loss: {training_loss / 18}")
        if (epoch % 1) == 0:
            correct = 0
            for data_1 in val_loader:
                model.eval()
                inp = data_1["inp"].to(device, dtype=torch.float)
                output = data_1["out"].to(device, dtype=torch.float)
                # inp = torch.unsqueeze(inp, 1)
                predict_val = model(inp)
                predict_val = predict_val > 0.5
                if predict_val[0][0] == output[0][0]:
                    correct += 1
            # print(correct / len(val_loader))
            accu = correct / len(val_loader)
            if accu > best_accu:
                print(f"current accu: {accu}")
                # print(f"best accu: {best_accu}")
                best_accu = accu
                timet = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                best_model = f"save_point/model_{timet}.pth"
                print(f"best model: {best_model}")
                torch.save(model.state_dict(), best_model)
    print(f"best_model: {best_model}")

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    test_model = NewModel()
    test_model.load_state_dict(torch.load(best_model))
    test_model.to(device)

    correct = 0
    for data_2 in test_loader:
        test_model.eval()
        inp = data_2["inp"].to(device, dtype=torch.float)
        output = data_2["out"].to(device, dtype=torch.float)
        # inp = torch.unsqueeze(inp, 1)
        predict_val = test_model(inp)
        predict_val = predict_val > 0.5
        if predict_val[0][0] == output[0][0]:
            correct += 1
    print(f"accurate: {correct / len(test_loader)}")

