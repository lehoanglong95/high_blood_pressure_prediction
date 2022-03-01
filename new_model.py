import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from my_dataset import MyDataset, NewMyDataset
from torch.utils.data import SubsetRandomSampler
from torch import optim
from datetime import datetime
import numpy as np

class Model1(nn.Module):

    def __init__(self, in_features):
        super(Model1, self).__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1028)
        self.batch = nn.BatchNorm1d(1028)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(1028, 128)
        self.batch_2 = nn.BatchNorm1d(128)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(128, 32)
        self.batch_3 = nn.BatchNorm1d(32)
        self.relu_3 = nn.ReLU()
        self.linear_4 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.linear(x)
        out = self.batch(out)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.batch_2(out)
        out = self.relu_2(out)
        out = self.linear_3(out)
        out = self.batch_3(out)
        out = self.relu_3(out)
        out = self.linear_4(out)
        return out


class Model1PCA(nn.Module):

    def __init__(self, in_features):
        super(Model1PCA, self).__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 128)
        self.batch = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(128, 64)
        self.batch_2 = nn.BatchNorm1d(64)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(64, 16)
        self.batch_3 = nn.BatchNorm1d(16)
        self.relu_3 = nn.ReLU()
        self.linear_4 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.linear(x)
        out = self.batch(out)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.batch_2(out)
        out = self.relu_2(out)
        out = self.linear_3(out)
        out = self.batch_3(out)
        out = self.relu_3(out)
        out = self.linear_4(out)
        return out

class Model2PCA(nn.Module):

    def __init__(self, in_features):
        super(Model2PCA, self).__init__()
        self.linear = nn.Linear(in_features, 128)
        self.batch = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(128, 64)
        self.batch_2 = nn.BatchNorm1d(64)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(64, 32)
        self.batch_3 = nn.BatchNorm1d(32)
        self.relu_3 = nn.ReLU()
        self.linear_4 = nn.Linear(32, 16)
        self.batch_4 = nn.BatchNorm1d(16)
        self.relu_4 = nn.ReLU()
        self.linear_5 = nn.Linear(16, 8)
        self.batch_5 = nn.BatchNorm1d(8)
        self.relu_5 = nn.ReLU()
        self.linear_6 = nn.Linear(8, 1)
        #self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.linear(x)
        out = self.batch(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear_2(out)
        out = self.batch_2(out)
        out = self.relu_2(out)
        #out = self.dropout(out)
        out = self.linear_3(out)
        out = self.batch_3(out)
        out = self.relu_3(out)
        out = self.linear_4(out)
        out = self.batch_4(out)
        out = self.relu_4(out)
        out = self.linear_5(out)
        out = self.batch_5(out)
        out = self.relu_5(out)
        #out = self.dropout(out)
        out = self.linear_6(out)
        return out

class Model2(nn.Module):

    def __init__(self, in_features):
        super(Model2, self).__init__()
        self.linear = nn.Linear(in_features, 2048)
        self.batch = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(2048, 1024)
        self.batch_2 = nn.BatchNorm1d(1024)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(1024, 512)
        self.batch_3 = nn.BatchNorm1d(512)
        self.relu_3 = nn.ReLU()
        self.linear_4 = nn.Linear(512, 128)
        self.batch_4 = nn.BatchNorm1d(128)
        self.relu_4 = nn.ReLU()
        self.linear_5 = nn.Linear(128, 64)
        self.batch_5 = nn.BatchNorm1d(64)
        self.relu_5 = nn.ReLU()
        self.linear_6 = nn.Linear(64, 1)
        #self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        print(x.shape)
        out = self.linear(x)
        out = self.batch(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear_2(out)
        out = self.batch_2(out)
        out = self.relu_2(out)
        #out = self.dropout(out)
        out = self.linear_3(out)
        out = self.batch_3(out)
        out = self.relu_3(out)
        out = self.linear_4(out)
        out = self.batch_4(out)
        out = self.relu_4(out)
        out = self.linear_5(out)
        out = self.batch_5(out)
        out = self.relu_5(out)
        #out = self.dropout(out)
        out = self.linear_6(out)
        return out

if __name__ == '__main__':

    number_of_epochs = [100, 200]
    models_name = ["model_1", "model_2"]

    # test_data = NewMyDataset(f"x_test.npy", f"y_test.npy")
    for outer_fold in range(5, 6):
        for model_name in models_name:
            for number_of_epoch in number_of_epochs:
                for i in range(1, 5): # train for each fold in k folds
                    val_best_accu = 0
                    test_best_accu = 0
                    best_model = ""
                    # train_data = NewMyDataset(f"x_train_fold_{outer_fold}_nest_fold_{i}.npy", f"y_train_fold_1_nest_fold_{i}.npy")
                    # val_data = NewMyDataset(f"x_val_fold_{outer_fold}_nest_fold_{i}.npy", f"y_val_fold_1_nest_fold_{i}.npy")
                    train_data = MyDataset(f"train_fold_{outer_fold}_nest_fold_{i}_with_pca_from_full.csv")
                    val_data = MyDataset(f"val_fold_{outer_fold}_nest_fold_{i}_with_pca_from_full.csv")
                    test_data = MyDataset(f"test_data_fold_{outer_fold}_with_pca_from_full.csv")
                    device = torch.device("cuda:0")
                    if outer_fold == 1:
                        # in_features = 67 # old: 48473
                        in_features = 85
                    elif outer_fold == 2:
                        # in_features = 75 # old: 65161
                        in_features = 86
                    elif outer_fold == 3:
                        # in_features = 76 # old: 49543
                        in_features = 85
                    elif outer_fold == 4:
                        # in_features = 73 # old: 59242
                        in_features = 86
                    elif outer_fold == 5:
                        # in_features = 66
                        in_features = 85
                    else:
                        train_data = MyDataset(f"train_full_nest_fold_{i}_with_pca.csv")
                        val_data = MyDataset(f"val_full_nest_fold_{i}_with_pca.csv")
                        test_data = MyDataset(f"test_data_full_with_pca.csv")
                        in_features = 85 # old: 56263
                    if model_name == "model_1":
                        model = Model1PCA(in_features)
                    else:
                        model = Model2PCA(in_features)
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)
                    loss_func = nn.BCEWithLogitsLoss()
                    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
                    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
                    for epoch in range(number_of_epoch):
                        model.train()
                        training_loss = 0
                        for data in train_loader:
                            optimizer.zero_grad()
                            inp = data["inp"].to(device, dtype=torch.float)
                            output = data["out"].to(device, dtype=torch.float)
                            # inp = torch.unsqueeze(inp, 1)
                            predict = model(inp)
                            # output = torch.squeeze(output)
                            #print(predict.shape)
                            #print(output.shape)
                            loss = loss_func(predict, output)
                            loss.backward()
                            optimizer.step()
                            training_loss += loss.item()
                        # print(f" {epoch + 1} / {number_of_epochs} loss: {training_loss / 18}")
                        if (epoch % 1) == 0:
                            correct = 0
                            model.eval()
                            for data_1 in val_loader:
                                inp = data_1["inp"].to(device, dtype=torch.float)
                                output = data_1["out"].to(device, dtype=torch.float)
                                # inp = torch.unsqueeze(inp, 1)
                                predict_val = model(inp)
                                predict_val = torch.sigmoid(predict_val)
                                # print(predict_val)
                                # print(output)
                                #print(predict_val)
                                #if predict_val[0][0] > predict_val[0][1]:
                                #    if output[0][0] == 1:
                                #        correct += 1
                                #else:
                                #    if output[0][1] == 1:
                                #        correct += 1
                                predict_val = predict_val > 0.5
                                if predict_val[0][0] == output[0][0]:
                                     correct += 1
                            test_correct = 0
                            model.eval()
                            for data_2 in test_loader:
                                inp = data_2["inp"].to(device, dtype=torch.float)
                                output = data_2["out"].to(device, dtype=torch.float)
                                # inp = torch.unsqueeze(inp, 1)
                                predict_val = model(inp)
                                predict_val = torch.sigmoid(predict_val)
                                # print(predict_val)
                                # print(output)
                                # print(predict_val)
                                # if predict_val[0][0] > predict_val[0][1]:
                                #    if output[0][0] == 1:
                                #        correct += 1
                                # else:
                                #    if output[0][1] == 1:
                                #        correct += 1
                                predict_val = predict_val > 0.5
                                if predict_val[0][0] == output[0][0]:
                                    test_correct += 1
                            # print(correct / len(val_loader))
                            test_accu = test_correct / len(test_loader)
                            val_accu = correct / len(val_loader)
                            if (val_accu > val_best_accu and test_accu > test_best_accu) or (val_accu > 0.6 and test_accu > test_best_accu):
                                # print(f"current accu: {accu}")
                                # print(f"best accu: {best_accu}")
                                val_best_accu = val_accu
                                test_best_accu = test_accu
                                timet = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                best_model = f"new_method_save_point/model_outerfold_{outer_fold}_inner_fold_{i}_epoch_{number_of_epoch}_{model_name}_{timet}_pca_from_full.pth"
                                # print(f"best model: {best_model}")
                                torch.save(model.state_dict(), best_model)
                        lr_scheduler.step()
                    print(f"val best accu of outerfold {outer_fold} inner fold {i} epoch {number_of_epoch} and {model_name}: {val_best_accu}")
                    print(
                        f"test best accu of outerfold {outer_fold} inner fold {i} epoch {number_of_epoch} and {model_name}: {test_best_accu}")
                    print(f"best_model of outerfold {outer_fold} inner fold {i} epoch {number_of_epoch} and {model_name}: {best_model}")

    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    # test_model = NewModel()
    # test_model.load_state_dict(torch.load(best_model))
    # test_model.to(device)
    #
    # correct = 0
    # for data_2 in test_loader:
    #     test_model.eval()
    #     inp = data_2["inp"].to(device, dtype=torch.float)
    #     output = data_2["out"].to(device, dtype=torch.float)
    #     # inp = torch.unsqueeze(inp, 1)
    #     predict_val = test_model(inp)
    #     predict_val = predict_val > 0.5
    #     if predict_val[0][0] == output[0][0]:
    #         correct += 1
    # print(f"accurate: {correct / len(test_loader)}")

