from sklearn.metrics import roc_curve, roc_auc_score
import torch
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

if __name__ == '__main__':
    for pretrain in ["random_forest/model_fold_1_inner_1_2021-12-28-21-22-51.pth",
                     "random_forest/model_fold_2_inner_1_2021-12-28-21-22-51.pth",
                     "random_forest/model_fold_3_inner_1_2021-12-28-21-22-52.pth",
                     "random_forest/model_fold_4_inner_3_2021-12-28-21-22-52.pth",
                     "random_forest/model_fold_5_inner_1_2021-12-28-21-22-52.pth"]:
        svm = pickle.load(open(pretrain, "rb"))
        outerfold = pretrain.split("_")[3]
        test_data = MyDataset(f"test_data_fold_{outerfold}_with_pca.csv")
        print("################ SVM #####################")
        predicts = svm.predict(test_data.x)
        probs = svm.predict_proba(test_data.x)[:, 1]
        print(confusion_matrix(test_data.y, predicts))
        precision = precision_score(test_data.y, predicts)
        recall = recall_score(test_data.y, predicts)
        print(f"precision: {precision_score(test_data.y, predicts)}")
        print(f"recall: {recall_score(test_data.y, predicts)}")
        print(f"f1 score: {(2 * precision * recall) / (precision + recall)}")
        print(f"accuracy: {accuracy_score(test_data.y, predicts)}")
        auc_score = roc_auc_score(test_data.y, probs)
        print(f"auc score: {auc_score}")