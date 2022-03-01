from sklearn.metrics import roc_curve, roc_auc_score
import torch
from new_model import Model1, Model2, Model2PCA, Model1PCA
import matplotlib.pyplot as plt
from my_dataset import MyDataset, NewMyDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
import torch.nn as nn
from statistics import mean

# class NewModel(nn.Module):
#
#     def __init__(self):
#         super(NewModel, self).__init__()
#         self.linear = nn.Linear(65161, 1024)
#         self.batch = nn.BatchNorm1d(1024)
#         self.relu = nn.ReLU()
#         self.linear_2 = nn.Linear(1024, 512)
#         self.batch_2 = nn.BatchNorm1d(512)
#         self.relu_2 = nn.ReLU()
#         self.linear_3 = nn.Linear(512, 64)
#         self.batch_3 = nn.BatchNorm1d(64)
#         self.relu_3 = nn.ReLU()
#         self.linear_4 = nn.Linear(64, 1)
#         #self.sigmoid = nn.Sigmoid()
#         #self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         out = self.linear(x)
#         out = self.batch(out)
#         out = self.relu(out)
#         #out = self.dropout(out)
#         out = self.linear_2(out)
#         out = self.batch_2(out)
#         out = self.relu_2(out)
#         #out = self.dropout(out)
#         out = self.linear_3(out)
#         out = self.batch_3(out)
#         out = self.relu_3(out)
#         #out = self.dropout(out)
#         out = self.linear_4(out)
#         return out
#         # return self.sigmoid(out)
#         # print(out.shape)
#
# class NewModel2(nn.Module):
#
#     def __init__(self):
#         super(NewModel2, self).__init__()
#         self.linear = nn.Linear(65161, 2048)
#         self.batch = nn.BatchNorm1d(2048)
#         self.relu = nn.ReLU()
#         self.linear_2 = nn.Linear(2048, 1024)
#         self.batch_2 = nn.BatchNorm1d(1024)
#         self.relu_2 = nn.ReLU()
#         self.linear_3 = nn.Linear(1024, 512)
#         self.batch_3 = nn.BatchNorm1d(512)
#         self.relu_3 = nn.ReLU()
#         self.linear_4 = nn.Linear(512, 128)
#         self.batch_4 = nn.BatchNorm1d(128)
#         self.relu_4 = nn.ReLU()
#         self.linear_5 = nn.Linear(128, 64)
#         self.batch_5 = nn.BatchNorm1d(64)
#         self.relu_5 = nn.ReLU()
#         self.linear_6 = nn.Linear(64, 1)
#         #self.sigmoid = nn.Sigmoid()
#         #self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         out = self.linear(x)
#         out = self.batch(out)
#         out = self.relu(out)
#         #out = self.dropout(out)
#         out = self.linear_2(out)
#         out = self.batch_2(out)
#         out = self.relu_2(out)
#         #out = self.dropout(out)
#         out = self.linear_3(out)
#         out = self.batch_3(out)
#         out = self.relu_3(out)
#         out = self.linear_4(out)
#         out = self.batch_4(out)
#         out = self.relu_4(out)
#         out = self.linear_5(out)
#         out = self.batch_5(out)
#         out = self.relu_5(out)
#         #out = self.dropout(out)
#         out = self.linear_6(out)
#         return out

if __name__ == '__main__':
    device = torch.device("cuda:1")
    # model = NewModel().to(device)
    # best_model = "save_point/model_2021-10-03-23-45-47.pth"
    # model.load_state_dict(torch.load(best_model))
    # test_data = MyDataset("msigcpg74420.csv",
    #                       "output.csv", 90, 120)
    # pretrain = "save_point/model_fold_2_2021-11-04-17-19-11.pth"
    # pretrain = "save_point/model_fold_2_2021-11-04-18-34-32.pth"
    # pretrains = ["new_method_save_point/model_fold_1_2021-11-20-16-24-22.pth",
    #              "new_method_save_point/model_fold_2_2021-11-20-16-24-54.pth",
    #              "new_method_save_point/model_fold_3_2021-11-20-16-26-43.pth",
    #              "new_method_save_point/model_fold_4_2021-11-20-16-27-43.pth"]
    # output 2 layers
    # pretrains = ["new_method_save_point/model_fold_4_2021-11-21-22-46-21.pth",
    #              "new_method_save_point/model_fold_4_2021-11-21-22-57-13.pth",
    #              "new_method_save_point/model_fold_4_2021-11-21-23-16-06.pth",
    #              "new_method_save_point/model_fold_2_2021-11-21-23-22-36.pth"]
    # pretrains = ["new_method_save_point/model_outerfold_1_inner_fold_4_epoch_200_model_2_2021-11-23-16-02-31.pth",
    #              "new_method_save_point/model_outerfold_2_inner_fold_4_epoch_200_model_2_2021-11-23-16-37-40.pth",
    #              "new_method_save_point/model_outerfold_3_inner_fold_1_epoch_200_model_2_2021-11-23-16-53-07.pth",
    #              "new_method_save_point/model_outerfold_4_inner_fold_2_epoch_200_model_2_2021-11-23-17-23-34.pth",
    #              "new_method_save_point/model_outerfold_5_inner_fold_1_epoch_100_model_2_2021-11-23-18-16-29.pth"]
    pretrains_PCA = ["new_method_save_point/model_outerfold_1_inner_fold_3_epoch_200_model_1_2022-01-04-21-35-09.pth",
                 "new_method_save_point/model_outerfold_2_inner_fold_4_epoch_100_model_1_2021-12-28-20-38-22.pth",
                 "new_method_save_point/model_outerfold_3_inner_fold_2_epoch_100_model_1_2021-12-28-20-42-15.pth",
                 "new_method_save_point/model_outerfold_4_inner_fold_4_epoch_100_model_2_2021-12-28-20-48-52.pth",
                 "new_method_save_point/model_outerfold_5_inner_fold_3_epoch_200_model_1_2021-12-28-20-51-54.pth"]
    rf_pretrains_PCA = ["random_forest/model_fold_1_inner_1_2021-12-28-21-22-51.pth",
                    "random_forest/model_fold_2_inner_1_2021-12-28-21-22-51.pth",
                    "random_forest/model_fold_3_inner_1_2021-12-28-21-22-52.pth",
                    "random_forest/model_fold_4_inner_3_2021-12-28-21-22-52.pth",
                    "random_forest/model_fold_5_inner_1_2021-12-28-21-22-52.pth"]
    svm_pretrains_PCA = ["svm/model_fold_1_inner_2_2021-12-28-21-00-35.pth",
                     "svm/model_fold_2_inner_2_2021-12-28-21-00-35.pth",
                     "svm/model_fold_3_inner_4_2021-12-28-21-00-35.pth",
                     "svm/model_fold_4_inner_3_2021-12-28-21-00-35.pth",
                     "svm/model_fold_5_inner_2_2021-12-28-21-00-36.pth"]

    pretrains_PCA_new = ["new_method_save_point/model_outerfold_1_inner_fold_2_epoch_200_model_2_2022-01-24-01-01-36_pca_from_full.pth", #0.63
                     "new_method_save_point/model_outerfold_2_inner_fold_4_epoch_200_model_1_2022-01-24-01-04-50_pca_from_full.pth", # 0.7
                     "new_method_save_point/model_outerfold_3_inner_fold_2_epoch_200_model_2_2022-01-24-01-11-24_pca_from_full.pth", # 0.66
                     "new_method_save_point/model_outerfold_4_inner_fold_2_epoch_100_model_1_2022-01-24-01-13-04_pca_from_full.pth",  #0.76
                     "new_method_save_point/model_outerfold_5_inner_fold_1_epoch_100_model_1_2022-01-24-01-53-47_pca_from_full.pth"] #0.66
    rf_pretrains_PCA_new = ["random_forest/model_fold_1_inner_4_2022-01-24-08-59-19_pca_from_full.pth", #0.53
                        "random_forest/model_fold_2_inner_4_2022-01-24-08-59-20_pca_from_full.pth", #0.66
                        "random_forest/model_fold_3_inner_3_2022-01-24-08-59-20_pca_from_full.pth", #0.6
                        "random_forest/model_fold_4_inner_3_2022-01-24-08-59-21_pca_from_full.pth", #0.7
                        "random_forest/model_fold_5_inner_3_2022-01-24-08-59-21_pca_from_full.pth"] #0.6
    svm_pretrains_PCA_new = ["svm/model_fold_1_inner_2_2022-01-24-09-04-17_pca_from_full.pth", #0.5
                         "svm/model_fold_2_inner_4_2022-01-24-09-04-17_pca_from_full.pth", #0.53
                         "svm/model_fold_3_inner_2_2022-01-24-09-04-17_pca_from_full.pth", #0.43
                         "svm/model_fold_4_inner_3_2022-01-24-09-04-17_pca_from_full.pth", #0.7
                         "svm/model_fold_5_inner_4_2022-01-24-09-04-17_pca_from_full.pth"] #0.6

    pretrains_PCA_VAE = ["new_method_save_point/model_outerfold_1_inner_fold_3_epoch_200_model_2_2022-01-23-16-17-31.pth",
                 "new_method_save_point/model_outerfold_2_inner_fold_2_epoch_200_model_1_2022-01-23-16-19-18.pth",
                 "new_method_save_point/model_outerfold_3_inner_fold_3_epoch_100_model_1_2022-01-23-16-23-28.pth",
                 "new_method_save_point/model_outerfold_4_inner_fold_4_epoch_100_model_2_2022-01-23-16-30-18.pth",
                 "new_method_save_point/model_outerfold_5_inner_fold_4_epoch_200_model_1_2022-01-23-16-33-48.pth"]
    rf_pretrains_PCA_VAE = ["random_forest/model_fold_1_inner_3_2022-01-23-21-22-37.pth",
                    "random_forest/model_fold_2_inner_1_2022-01-23-21-22-37.pth",
                    "random_forest/model_fold_3_inner_3_2022-01-23-21-22-37.pth",
                    "random_forest/model_fold_4_inner_3_2022-01-23-21-22-38.pth",
                    "random_forest/model_fold_5_inner_3_2022-01-23-21-22-38.pth"]
    svm_pretrains_PCA_VAE = ["svm/model_fold_1_inner_3_2022-01-23-21-18-38.pth",
                     "svm/model_fold_2_inner_1_2022-01-23-21-18-38.pth",
                     "svm/model_fold_3_inner_3_2022-01-23-21-18-38.pth",
                     "svm/model_fold_4_inner_4_2022-01-23-21-18-38.pth",
                     "svm/model_fold_5_inner_4_2022-01-23-21-18-38.pth"]

    # VAE
    pretrains = ["new_method_save_point/model_outerfold_1_inner_fold_4_epoch_200_model_2_2022-01-10-00-17-46.pth",
                  "new_method_save_point/model_outerfold_2_inner_fold_4_epoch_200_model_2_2022-01-10-00-30-57.pth",
                  "new_method_save_point/model_outerfold_3_inner_fold_1_epoch_200_model_2_2022-01-10-00-33-20.pth",
                  "new_method_save_point/model_outerfold_4_inner_fold_2_epoch_200_model_2_2022-01-10-00-47-23.pth",
                  "new_method_save_point/model_outerfold_5_inner_fold_2_epoch_200_model_2_2022-01-10-00-57-07.pth"]
    rf_pretrains = ["random_forest/model_fold_1_inner_4_2022-01-10-08-41-51.pth",
                    "random_forest/model_fold_2_inner_4_2022-01-10-08-42-48.pth",
                    "random_forest/model_fold_3_inner_1_2022-01-10-08-42-55.pth",
                    "random_forest/model_fold_4_inner_1_2022-01-10-08-43-23.pth",
                    "random_forest/model_fold_5_inner_1_2022-01-10-08-43-55.pth"]
    svm_pretrains = ["svm/model_fold_1_inner_3_2022-01-10-08-46-13.pth",
                     "svm/model_fold_2_inner_4_2022-01-10-08-47-18.pth",
                     "svm/model_fold_3_inner_1_2022-01-10-08-47-25.pth",
                     "svm/model_fold_4_inner_1_2022-01-10-08-47-54.pth",
                     "svm/model_fold_5_inner_2_2022-01-10-08-48-35.pth"]
    # pretrains = ["new_method_save_point/model_outerfold_1_inner_fold_3_epoch_100_model_1_2022-01-04-21-34-06.pth",
                 # "new_method_save_point/model_outerfold_1_inner_fold_3_epoch_200_model_1_2022-01-04-21-35-09.pth"]

    dl_models = []
    auc_scores = []
    auprc_scores = []
    accuracy_scores = []
    recalls = []
    specificities = []
    precisions = []
    f1_scores = []
    for idx, (pretrain_PCA, rf_PCA, svm_PCA) in enumerate(zip(pretrains_PCA_new, rf_pretrains_PCA_new, svm_pretrains_PCA_new)):
    # for idx, pretrain in enumerate(pretrains):
        outer_fold = int(svm_PCA.split("_")[2])
        # inner_fold = int(pretrain.split("_")[4])
        # print(outer_fold)
        # print(inner_fold)

        if outer_fold == 1:
            # in_features = 67  # old: 48473
            in_features = 48473
            in_features_pca = 85
        elif outer_fold == 2:
            # in_features = 75  # old: 65161
            in_features = 65161
            in_features_pca = 86
        elif outer_fold == 3:
            # in_features = 76  # old: 49543
            in_features = 49543
            in_features_pca = 85
        elif outer_fold == 4:
            # in_features = 73  # old: 59242
            in_features = 59242
            in_features_pca = 86
        elif outer_fold == 5:
            # in_features = 66
            in_features = 56263
            in_features_pca = 85
        if "model_1" in pretrain_PCA:
            dl_model = Model1PCA(in_features_pca)
        else:
            dl_model = Model2PCA(in_features_pca)
        # if "model_1" in pretrain_PCA:
        #     dl_model_pca = Model1(in_features_pca)
        # else:
        #     dl_model_pca = Model2PCA(in_features_pca)
        # rf_model = pickle.load(open(rf, "rb"))
        # svm_model = pickle.load(open(svm, "rb"))
        rf_model_pca = pickle.load(open(rf_PCA, "rb"))
        svm_model_pca = pickle.load(open(svm_PCA, "rb"))
        # if idx % 4 == 0 and idx != 0:
        #     print("################ DL MODEL #####################")
        #     print(f"auc score: {mean(auc_scores)}")
        #     print(f"auprc score: {mean(auprc_scores)}")
        #     print(f"accuracy: {mean(accuracy_scores)}")
        #     print(f"recall: {mean(recalls)}")
        #     print(f"specificity: {mean(specificities)}")
        #     print(f"precision: {mean(precisions)}")
        #     print(f"f1 score: {mean(f1_scores)}")
        #     dl_models = []
        #     auc_scores = []
        #     auprc_scores = []
        #     accuracy_scores = []
        #     recalls = []
        #     specificities = []
        #     precisions = []
        #     f1_scores = []
        dl_model.load_state_dict(torch.load(pretrain_PCA))
        dl_model.to(device)
        # dl_model_pca.load_state_dict(torch.load(pretrain_PCA))
        # dl_model_pca.to(device)
        #dl_models.append(dl_model)
    # svm = pickle.load(open("svm/model_fold_2_2021-11-04-18-22-42.pth", "rb"))
    # random_forest = pickle.load(open("random_forest/model_fold_3_2021-11-04-17-34-47.pth", "rb"))
    # model.load_state_dict(torch.load(best_model))
    # test_data = MyDataset("msigcpg74420.csv",
    #                       "output.csv", 90, 120)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
    # test_data = NewMyDataset(f"x_test_fold_1.npy", f"y_test_fold_1.npy")
        test_data = MyDataset(f"test_data_fold_{outer_fold}_with_pca_from_full.csv")
        # test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # output = pd.read_csv("output.csv", header=None).to_numpy()
        # y_test = output[90:120].squeeze()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
        # test_data_pca = MyDataset(f"test_data_fold_{outer_fold}_with_pca.csv")
        # test_loader_pca = torch.utils.data.DataLoader(test_data_pca, batch_size=1)

    #for idx, model in enumerate(dl_models):
            # if idx == 0:
        # probs = []
        # predicts = []
        # y_test = []
        # dl_model.eval()
        probs_pca = []
        predicts_pca = []
        y_test_pca = []
        dl_model.eval()
        for data_2 in test_loader:
            inp = data_2["inp"].to(device, dtype=torch.float)
            output = data_2["out"]
            # output = output.to(device, dtype=torch.long)
            # inp = torch.unsqueeze(inp, 1)
            predict_val = dl_model(inp)
            predict_val = torch.sigmoid(predict_val)
            #predict_val = torch.softmax(predict_val, dim=1)
            # print(predict_val[0][0])
            # print(predict_val[0][1])
            #if predict_val[0][0] > predict_val[0][1]:
            #    predicts.append(0)
            #    probs.append(predict_val[0][0].detach().cpu().numpy())
            #else:
            #    predicts.append(1)
            #    probs.append(predict_val[0][1].detach().cpu().numpy())

            probs_pca.append(predict_val[0][0].detach().cpu().numpy())
            if predict_val[0][0].detach().cpu().numpy() > 0.5:
                 predicts_pca.append(1)
            else:
                 predicts_pca.append(0)
            if output[0][0] == 1:
                y_test_pca.append(1)
            else:
                y_test_pca.append(0)
            #print(predicts)
            #print(y_test)
            #print(probs)
        # for data_2 in test_loader_pca:
        #     inp = data_2["inp"].to(device, dtype=torch.float)
        #     output = data_2["out"]
        #     # output = output.to(device, dtype=torch.long)
        #     # inp = torch.unsqueeze(inp, 1)
        #     predict_val = dl_model_pca(inp)
        #     predict_val = torch.sigmoid(predict_val)
        #     #predict_val = torch.softmax(predict_val, dim=1)
        #     # print(predict_val[0][0])
        #     # print(predict_val[0][1])
        #     #if predict_val[0][0] > predict_val[0][1]:
        #     #    predicts.append(0)
        #     #    probs.append(predict_val[0][0].detach().cpu().numpy())
        #     #else:
        #     #    predicts.append(1)
        #     #    probs.append(predict_val[0][1].detach().cpu().numpy())
        #
        #     probs_pca.append(predict_val[0][0].detach().cpu().numpy())
        #     if predict_val[0][0].detach().cpu().numpy() > 0.5:
        #          predicts_pca.append(1)
        #     else:
        #          predicts_pca.append(0)
        #     if output[0][0] == 1:
        #         y_test_pca.append(1)
        #     else:
        #         y_test_pca.append(0)
        # svm_predicts = svm_model.predict(test_data.x)
        # svm_probs = svm_model.predict_proba(test_data.x)[:, 1]
        # rf_predicts = rf_model.predict(test_data.x)
        # rf_probs = rf_model.predict_proba(test_data.x)[:, 1]
        svm_predicts_pca = svm_model_pca.predict(test_data.x)
        svm_probs_pca = svm_model_pca.predict_proba(test_data.x)[:, 1]
        rf_predicts_pca = rf_model_pca.predict(test_data.x)
        rf_probs_pca = rf_model_pca.predict_proba(test_data.x)[:, 1]
        # print(y_test)
        cf_matrix = confusion_matrix(y_test_pca, predicts_pca)
        precision = precision_score(y_test_pca, predicts_pca)
        recall = recall_score(y_test_pca, predicts_pca)
        # print(cf_matrix)
        auc_score = roc_auc_score(np.array(y_test_pca), probs_pca)
        # auc_score_pca = roc_auc_score(np.array(y_test_pca), probs_pca)
        auc_scores.append(auc_score)
        auprc_scores.append(average_precision_score(y_test_pca, probs_pca))
        accuracy_scores.append(accuracy_score(y_test_pca, predicts_pca))
        recalls.append(recall_score(y_test_pca, predicts_pca))
        specificities.append(cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[0][1]))
        precisions.append(precision_score(y_test_pca, predicts_pca))
        f1_scores.append((2 * precision * recall) / (precision + recall))
        print(outer_fold)
        print(f"auc score: {auc_score}")
        print(f"auprc score: {average_precision_score(y_test_pca, probs_pca)}")
        print(f"accuracy: {accuracy_score(y_test_pca, predicts_pca)}")
        print(f"recall: {recall_score(y_test_pca, predicts_pca)}")
        print(f"specificity: {cf_matrix[1][1] / (cf_matrix[1][1] + cf_matrix[0][1])}")
        print(f"precision: {precision_score(y_test_pca, predicts_pca)}")
        print(f"f1 score: {(2 * precision * recall) / (precision + recall)}")
        fpr, tpr, _ = roc_curve(y_test_pca, probs_pca)
        fpr1, tpr1, _ = roc_curve(y_test_pca, rf_probs_pca)
        fpr2, tpr2, _ = roc_curve(y_test_pca, svm_probs_pca)
        # fpr_pca, tpr_pca, _ = roc_curve(y_test_pca, probs)
        # fpr1_pca, tpr1_pca, _ = roc_curve(y_test_pca, rf_probs_pca)
        # fpr2_pca, tpr2_pca, _ = roc_curve(y_test_pca, svm_probs_pca)
        plt.plot(fpr, tpr, label="Deep Learning Model (AUROC = {:.2f})".format(auc_score))
        plt.plot(fpr1, tpr1, label="Random Forest (AUROC = {:.2f})".format(roc_auc_score(np.array(y_test_pca), rf_probs_pca)))
        plt.plot(fpr2, tpr2, label="SVM (AUROC = {:.2f})".format(roc_auc_score(np.array(y_test_pca), svm_probs_pca)))
        # plt.plot(fpr_pca, tpr_pca, label="Deep Learning Model With PCA (AUROC = {:.2f})".format(auc_score_pca))
        # plt.plot(fpr1_pca, tpr1_pca, label="Random Forest With PCA (AUROC = {:.2f})".format(roc_auc_score(np.array(y_test_pca), rf_probs_pca)))
        # plt.plot(fpr2_pca, tpr2_pca, label="SVM With PCA (AUROC = {:.2f})".format(roc_auc_score(np.array(y_test_pca), svm_probs_pca)))
        print(f"svm auc score: {roc_auc_score(np.array(y_test_pca), svm_probs_pca)}")
        print(f"svm auprc score: {average_precision_score(y_test_pca, svm_probs_pca)}")
        print(f"svm accuracy: {accuracy_score(y_test_pca, svm_predicts_pca)}")
        print(f"svm recall: {recall_score(y_test_pca, svm_predicts_pca)}")
        svm_cf_matrix = confusion_matrix(y_test_pca, svm_predicts_pca)
        print(f"svm specificity: {svm_cf_matrix[1][1] / (svm_cf_matrix[1][1] + svm_cf_matrix[0][1])}")
        print(f"svm precision: {precision_score(y_test_pca, svm_predicts_pca)}")
        svm_precision = precision_score(y_test_pca, svm_predicts_pca)
        svm_recall = recall_score(y_test_pca, svm_predicts_pca)
        print(f"svm f1 score: {(2 * svm_precision * svm_recall) / (svm_precision + svm_recall)}")

        print(f"random forest auc score: {roc_auc_score(np.array(y_test_pca), rf_probs_pca)}")
        print(f"random forest auprc score: {average_precision_score(y_test_pca, rf_probs_pca)}")
        print(f"random forest accuracy: {accuracy_score(y_test_pca, rf_predicts_pca)}")
        print(f"random forest recall: {recall_score(y_test_pca, rf_predicts_pca)}")
        rf_cf_matrix = confusion_matrix(y_test_pca, rf_predicts_pca)
        print(f"srandom forest pecificity: {rf_cf_matrix[1][1] / (rf_cf_matrix[1][1] + rf_cf_matrix[0][1])}")
        print(f"random forest precision: {precision_score(y_test_pca, rf_predicts_pca)}")
        rf_precision = precision_score(y_test_pca, rf_predicts_pca)
        rf_recall = recall_score(y_test_pca, rf_predicts_pca)
        print(f"random forest f1 score: {(2 * rf_precision * rf_recall) / (rf_precision + rf_recall)}")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f"/home/longle/auroc_fold_{outer_fold}_pca_from_full.png")
        plt.clf()
        # plt.show()
        from sklearn.metrics import precision_recall_curve
        dl_precision, dl_recall,  _ = precision_recall_curve(test_data.y, probs_pca)
        rf_precision, rf_recall, _ = precision_recall_curve(test_data.y, rf_probs_pca)
        svm_precision, svm_recall, _ = precision_recall_curve(test_data.y, svm_probs_pca)
        # dl_precision_pca, dl_recall_pca, _ = precision_recall_curve(test_data_pca.y, probs_pca)
        # rf_precision_pca, rf_recall_pca, _ = precision_recall_curve(test_data_pca.y, rf_probs_pca)
        # svm_precision_pca, svm_recall_pca, _ = precision_recall_curve(test_data_pca.y, svm_probs_pca)
        plt.plot(dl_recall, dl_precision, label="Deep Learning Model (AUPRC = {:.2f})".format(average_precision_score(y_test_pca, probs_pca)))
        plt.plot(rf_recall, rf_precision, label="Random Forest (AUPRC = {:.2f})".format(average_precision_score(y_test_pca, rf_probs_pca)))
        plt.plot(svm_recall, svm_precision, label="SVM (AUPRC = {:.2f})".format(average_precision_score(y_test_pca, svm_probs_pca)))
        # plt.plot(dl_precision_pca, dl_precision_pca,
        #          label="Deep Learning Model With PCA (AUPRC = {:.2f})".format(average_precision_score(y_test_pca, probs_pca)))
        # plt.plot(rf_recall_pca, rf_precision_pca,
        #          label="Random Forest With PCA (AUPRC = {:.2f})".format(average_precision_score(y_test_pca, rf_probs_pca)))
        # plt.plot(svm_recall_pca, svm_precision_pca, label="SVM With PCA (AUPRC = {:.2f})".format(average_precision_score(y_test_pca, svm_probs_pca)))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(f"/home/longle/auprc_fold_{outer_fold}_pca_from_full.png")
        plt.clf()
    # plt.savefig("dl_model.png")
        # elif idx == 1:
        #     print("################ SVM #####################")
        #     predicts = svm.predict(test_data.x)
        #     probs = svm.predict_proba(test_data.x)[:, 1]
        #     print(confusion_matrix(test_data.y, predicts))
        #     print(f"precision: {precision_score(test_data.y, predicts)}")
        #     print(f"sensitivity: {recall_score(test_data.y, predicts)}")
        #     print(f"accuracy: {accuracy_score(test_data.y, predicts)}")
        #     auc_score = roc_auc_score(test_data.y, probs)
        #     print(f"auc score: {auc_score}")
        #     # fpr, tpr, _ = roc_curve(test_data.y, probs)
        #     # plt.plot(fpr, tpr)
        #     # plt.xlabel("False Positive Rate")
        #     # plt.ylabel("True Positive Rate")
        #     # plt.savefig("svm.png")
        #     display = PrecisionRecallDisplay.from_predictions(test_data.y, probs)
        #     plt.show()
        # else:
        #     print("################ Random Forest #####################")
        #     predicts = random_forest.predict(test_data.x)
        #     probs = random_forest.predict_proba(test_data.x)[:, 1]
        #     print(confusion_matrix(test_data.y, predicts))
        #     print(f"precision: {precision_score(test_data.y, predicts)}")
        #     print(f"sensitivity: {recall_score(test_data.y, predicts)}")
        #     print(f"accuracy: {accuracy_score(test_data.y, predicts)}")
        #     auc_score = roc_auc_score(test_data.y, probs)
        #     print(f"auc score: {auc_score}")
        #     # fpr, tpr, _ = roc_curve(test_data.y, probs)
        #     # plt.plot(fpr, tpr)
        #     # plt.xlabel("False Positive Rate")
        #     # plt.ylabel("True Positive Rate")
        #     # plt.savefig("rf.png")
        #     display = PrecisionRecallDisplay.from_predictions(test_data.y, probs)
        #     plt.show()

