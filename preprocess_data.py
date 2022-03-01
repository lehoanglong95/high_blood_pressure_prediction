import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # minmaxscaler = MinMaxScaler()
    # pca = PCA(.90)
    # # scaler = StandardScaler()
    # full_data = pd.read_excel(f"participants_dataset_for_methylation.xlsx")
    # full_data = full_data.drop(["methyl_code", "cgname"], axis=1)
    # full_data = full_data.T
    # full_data.index.name = "id"
    # fold_1 = pd.read_csv("raw_fold_1.csv")
    # fold_2 = pd.read_csv("raw_fold_2.csv")
    # fold_3 = pd.read_csv("raw_fold_3.csv")
    # fold_4 = pd.read_csv("raw_fold_4.csv")
    # fold_5 = pd.read_csv("raw_fold_5.csv")
    # train_fold = pd.concat([fold_1, fold_2, fold_3, fold_4])
    # train_data = full_data.merge(train_fold, how="inner", on="id")
    # test_data = full_data.merge(fold_5, how="inner", on="id")
    # # test_data = test_data.drop(["methyl_code"], axis=1)
    # # test_data = test_data.T
    # # test_data.index.name = "id"
    # output_df = pd.read_csv("Output.csv")
    # # new_test_data = test_data.merge(output_df, how="inner", on="id")
    # num_features = 443612
    # minmaxscaler.fit(train_data[list(map(str, list(range(num_features))))])
    # train_data[list(map(str, list(range(num_features))))] = minmaxscaler.transform(train_data[list(map(str, list(range(num_features))))])
    # test_data[list(map(str, list(range(num_features))))] = minmaxscaler.transform(test_data[list(map(str, list(range(num_features))))])
    # pca.fit(train_data[list(map(str, list(range(num_features))))])
    # new_train_data = pd.DataFrame(data=pca.transform(train_data[list(map(str, list(range(num_features))))]), index=train_data.id)
    # new_test_data = pd.DataFrame(data=pca.transform(test_data[list(map(str, list(range(num_features))))]), index=test_data.id)
    # new_train_data.to_csv(f"train_data_full_with_pca.csv")
    # new_new_test_data = new_test_data.merge(output_df, how="inner", on="id")
    # new_new_test_data.to_csv(f"test_data_full_with_pca.csv", index=False)
    for i in range(1, 6):
        minmaxscaler = MinMaxScaler()
        # pca = PCA(.90)
        # scaler = StandardScaler()
        train_data = pd.read_csv(f"train_data_fold_{i}_full.csv")
        # train_data = train_data.drop(["methyl_code"], axis=1)
        # train_data = train_data.T
        # train_data.index.name = "id"
        # if i == 1:
        test_data = pd.read_csv(f"test_data_fold_{i}_full.csv")
        # else:
        # test_data = pd.read_excel(f"Testset{i}.xlsx")
        # test_data = test_data.drop(["methyl_code"], axis=1)
        # test_data = test_data.T
        # test_data.index.name = "id"
        output_df = pd.read_csv("Output.csv")
        # new_test_data = test_data.merge(output_df, how="inner", on="id")
        # if i == 1:
        #     num_features = 48473
        # elif i == 2:
        #     num_features = 65161
        # elif i == 3:
        #     num_features = 49543
        # elif i == 4:
        #     num_features = 59242
        # else:
        num_features = 443612
        minmaxscaler.fit(train_data[list(map(str, list(map(str, list(range(num_features))))))])
        train_data[list(map(str, list(range(num_features))))] = minmaxscaler.transform(train_data[list(map(str, list(range(num_features))))])
        test_data[list(map(str, list(range(num_features))))] = minmaxscaler.transform(test_data[list(map(str, list(range(num_features))))])
        # train_data.to_csv(f"train_data_fold_{i}.csv")
        # mew_test_data = test_data.merge(output_df, how="inner", on="id")
        # new_test_data.to_csv(f"test_data_fold_{i}.csv", index=False)
        # pca.fit(train_data[list(map(str, list(range(num_features))))])
        # new_train_data = pd.DataFrame(data=pca.transform(train_data[list(map(str, list(range(num_features))))]), index=train_data.id)
        # new_test_data = pd.DataFrame(data=pca.transform(test_data[list(map(str, list(range(num_features))))]), index=test_data.id)
        train_data.to_csv(f"train_data_fold_{i}_with_pca_from_full_mms.csv")
        new_test_data = test_data.merge(output_df, how="inner", on="id")
        new_test_data.to_csv(f"test_data_fold_{i}_with_pca_from_full_mms.csv")
    # input_data = input_data

    # output_data = new_df["Output"].to_numpy()
    # np.save("x_test_fold_1", input_data)
    # np.save("y_test_fold_1", output_data)