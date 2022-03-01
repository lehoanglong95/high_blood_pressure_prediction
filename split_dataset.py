import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from my_dataset import MyDataset
import numpy as np

def split_dataseet():
    """split dataset into train set and test set with the proportion between negative and positive of train set equal
    this of test set"""
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    train_data = MyDataset("msigcpg74420.csv",
                           "Outputnew.csv")
    for train_index, test_index in skf.split(train_data.x, train_data.y):
        X_train, X_test = train_data.x[train_index], train_data.x[test_index]
        y_train, y_test = train_data.y[train_index], train_data.y[test_index]
        np.save("x_train.npy", X_train)
        np.save("x_test.npy", X_test)
        np.save("y_train.npy", y_train)
        np.save("y_test.npy", y_test)
        break

def split_train_set():
    """split train set into 4 folds with each fold contains 30 subjects"""
    # input = pd.read_csv(f"train_data_full_with_pca.csv")
    # # out = pd.read_csv("Output.csv")
    # fold_1 = pd.read_csv("raw_fold_1.csv")
    # fold_2 = pd.read_csv("raw_fold_2.csv")
    # fold_3 = pd.read_csv("raw_fold_3.csv")
    # fold_4 = pd.read_csv("raw_fold_4.csv")
    # # fold_5 = pd.read_csv("raw_fold_5.csv")
    # fold_1_df = input.merge(fold_1, how="inner", on="id")
    # fold_2_df = input.merge(fold_2, how="inner", on="id")
    # fold_3_df = input.merge(fold_3, how="inner", on="id")
    # fold_4_df = input.merge(fold_4, how="inner", on="id")
    # pd.concat([fold_1_df, fold_2_df, fold_3_df]).to_csv(f"train_full_nest_fold_1_with_pca.csv", index=False)
    # fold_4_df.to_csv(f"val_full_nest_fold_1_with_pca.csv", index=False)
    # pd.concat([fold_1_df, fold_2_df, fold_4_df]).to_csv(f"train_full_nest_fold_2_with_pca.csv", index=False)
    # fold_3_df.to_csv(f"val_full_nest_fold_2_with_pca.csv", index=False)
    # pd.concat([fold_1_df, fold_4_df, fold_3_df]).to_csv(f"train_full_nest_fold_3_with_pca.csv", index=False)
    # fold_2_df.to_csv(f"val_full_nest_fold_3_with_pca.csv", index=False)
    # pd.concat([fold_4_df, fold_2_df, fold_3_df]).to_csv(f"train_full_nest_fold_4_with_pca.csv", index=False)
    # fold_1_df.to_csv(f"val_full_nest_fold_4_with_pca.csv", index=False)
    for i in range(1, 6):
        skf = StratifiedKFold(n_splits=4, shuffle=True)
        input = pd.read_csv(f"train_data_fold_{i}_with_pca_from_full.csv")
        # out = pd.read_csv("Output.csv")
        if i == 1:
            fold_1 = pd.read_csv("raw_fold_1.csv")
            fold_2 = pd.read_csv("raw_fold_2.csv")
            fold_3 = pd.read_csv("raw_fold_3.csv")
            fold_4 = pd.read_csv("raw_fold_4.csv")
        elif i == 2:
            fold_1 = pd.read_csv("raw_fold_1.csv")
            fold_2 = pd.read_csv("raw_fold_2.csv")
            fold_3 = pd.read_csv("raw_fold_3.csv")
            fold_4 = pd.read_csv("raw_fold_5.csv")
        elif i == 3:
            fold_1 = pd.read_csv("raw_fold_1.csv")
            fold_2 = pd.read_csv("raw_fold_2.csv")
            fold_3 = pd.read_csv("raw_fold_4.csv")
            fold_4 = pd.read_csv("raw_fold_5.csv")
        elif i == 4:
            fold_1 = pd.read_csv("raw_fold_1.csv")
            fold_2 = pd.read_csv("raw_fold_3.csv")
            fold_3 = pd.read_csv("raw_fold_4.csv")
            fold_4 = pd.read_csv("raw_fold_5.csv")
        else:
            fold_1 = pd.read_csv("raw_fold_2.csv")
            fold_2 = pd.read_csv("raw_fold_3.csv")
            fold_3 = pd.read_csv("raw_fold_4.csv")
            fold_4 = pd.read_csv("raw_fold_5.csv")
    #     #fold_5 = pd.read_csv("raw_fold_5.csv")
        fold_1_df = input.merge(fold_1, how="inner", on="id")
        fold_2_df = input.merge(fold_2, how="inner", on="id")
        fold_3_df = input.merge(fold_3, how="inner", on="id")
        fold_4_df = input.merge(fold_4, how="inner", on="id")
        pd.concat([fold_1_df, fold_2_df, fold_3_df]).to_csv(f"train_fold_{i}_nest_fold_1_with_pca_from_full.csv", index=False)
        fold_4_df.to_csv(f"val_fold_{i}_nest_fold_1_with_pca_from_full.csv", index=False)
        pd.concat([fold_1_df, fold_2_df, fold_4_df]).to_csv(f"train_fold_{i}_nest_fold_2_with_pca_from_full.csv", index=False)
        fold_3_df.to_csv(f"val_fold_{i}_nest_fold_2_with_pca_from_full.csv", index=False)
        pd.concat([fold_1_df, fold_4_df, fold_3_df]).to_csv(f"train_fold_{i}_nest_fold_3_with_pca_from_full.csv", index=False)
        fold_2_df.to_csv(f"val_fold_{i}_nest_fold_3_with_pca_from_full.csv", index=False)
        pd.concat([fold_4_df, fold_2_df, fold_3_df]).to_csv(f"train_fold_{i}_nest_fold_4_with_pca_from_full.csv", index=False)
        fold_1_df.to_csv(f"val_fold_{i}_nest_fold_4_with_pca_from_full.csv", index=False)
    # new_df = input.merge(out, how="inner", on="id")
    # X = new_df.drop(["id", "Output"], axis=1).to_numpy()
    # Y = new_df["Output"].to_numpy()
    # for idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]
    #     np.save(f"x_train_fold_1_nest_fold_{idx + 1}.npy", X_train)
    #     np.save(f"x_val_fold_1_nest_fold_{idx + 1}.npy", X_test)
    #     np.save(f"y_train_fold_1_nest_fold_{idx + 1}.npy", y_train)
    #     np.save(f"y_val_fold_1_nest_fold_{idx + 1}.npy", y_test)

if __name__ == '__main__':
    split_train_set()
    # skf = StratifiedKFold(n_splits=4, shuffle=True)
    # train_data = MyDataset("msigcpg74420.csv",
    #                        "Outputnew.csv")
    # for train_index, test_index in skf.split(train_data.x, train_data.y):
    #     X_train, X_test = train_data.x[train_index], train_data.x[test_index]
    #     y_train, y_test = train_data.y[train_index], train_data.y[test_index]
    #     np.save("x_train.npy", X_train)
    #     np.save("x_test.npy", X_test)
    #     np.save("y_train.npy", y_train)
    #     np.save("y_test.npy", y_test)
    #     break
    # train_data = MyDataset("msigcpg74420.csv",
    #                        "output.csv", 0, 110)
    # for fold, (train_ids, test_ids) in enumerate(k_fold.split(train_data)):
    #     train_subsampler = SubsetRandomSampler(train_ids)
    #     test_subsampler = SubsetRandomSampler(test_ids)