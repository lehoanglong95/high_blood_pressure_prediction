from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MyDataset(Dataset):

    def __init__(self, data_file, result_file, type):
        if data_file == result_file:
            data = pd.read_csv(data_file)
            if type == 0 or type == 1:
                data = data[data["type"] == float(type)]
                data = data.drop("type", axis=1)
            self.id1 = data.id.values
            self.id2 = self.id1
            data.drop("id", axis=1, inplace=True)
            self.data = np.array(data.drop(["results"], axis=1))
            self.result = np.array(data.results.values)
        else:
            data = pd.read_csv(data_file)
            result = pd.read_csv(result_file)
            if type == 0 or type == 1:
                data = data[data["type"] == float(type)]
                result = result[result["type"] == float(type)]
                data = data.drop("type", axis=1)
                result = result.drop("type", axis=1)
            self.id1 = data.id.values
            self.id2 = result.id.values
            data.drop("id", axis=1, inplace=True)
            result.drop("id", axis=1, inplace=True)
            self.data = np.array(data)
            self.result = np.array(result)

    def __getitem__(self, index):
        assert self.id1[index] == self.id2[index]
        # assert self.t_data["id"].loc[index] == self.t_result["id"].loc[index]
        # assert self.data["id"].loc[index] == self.result["id"].loc[index]
        # a = self.data.drop("id", axis=1, inplace=False)
        # b = self.result.drop("id", axis=1, inplace=False)
        # temp_data = np.array(a.loc[index])
        # temp_result = np.array(b.loc[index])

        return self.data[index], self.result[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_dataset = MyDataset("./train_data_fold_1.csv", "./train_result_fold_1.csv", type=0)
    val_dataset = MyDataset("./train_data_fold_2.csv", "./train_result_fold_2.csv", type=1)
    train_dataloader = DataLoader(train_dataset, batch_size=9, shuffle=False, num_workers=3)
    for data, result in train_dataset:
        print("A")
        break
        # print(data.shape)
        # print(result.shape)