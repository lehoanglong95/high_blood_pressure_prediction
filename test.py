import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from my_dataset import MyDataset, NewMyDataset
from torch.utils.data import SubsetRandomSampler
from torch import optim
from datetime import datetime
from temp.new_model import NewModel
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = torch.device("cuda:0")
    # model = NewModel().to(device)
    # best_model = "save_point/model_2021-10-03-23-45-47.pth"
    # pretrain = "/home/longle/PycharmProjects/miRNA_DL/pretrain/best_model_fold_1_epoch_41.pth"
    pretrain = ""
    model = torch.load(pretrain)
    # model.load_state_dict(torch.load(best_model))
    # test_data = MyDataset("msigcpg74420.csv",
    #                       "output.csv", 90, 120)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    test_data = MyDataset(f"./test_combination_data.csv",
                        f"./test_combination_data.csv", type=2)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for inp, output in test_loader:
        model.eval()
        print(output.shape)
        inp = inp.to(device, dtype=torch.float)
        output = output.to(device, dtype=torch.long)
        # inp = torch.unsqueeze(inp, 1)
        predict_val = model(inp)
        predict_val = predict_val > 0.5
        if predict_val[0] == 1 and output[0] == 1:
            TP += 1
        elif predict_val[0] == 0 and output[0] == 0:
            TN += 1
        elif predict_val[0] == 1 and output[0] == 0:
            FP += 1
        elif predict_val[0] == 0 and output[0] == 1:
            FN += 1
        # if predict_val[0][0] == output[0][0]:
        #     correct += 1
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print(f"accuracy: {(TP + TN) / (TP + FP + FN + TN)}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {2 * precision * recall / (precision + recall)}")
    print(f"specificity: {TN / (TN + FP)}")