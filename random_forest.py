from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from my_dataset import NewMyDataset, MyDataset
import pickle
import torch
from datetime import datetime
from sklearn.svm import SVC


if __name__ == '__main__':
    for outer_fold in range(1, 6):
        for i in range(1, 5):
            best_accu = 0
            best_model = ""
            # clf = RandomForestClassifier(max_depth=500, random_state=0)
            clf = SVC(kernel="linear", probability=True)
            train_data = MyDataset(f"train_fold_{outer_fold}_nest_fold_{i}_with_pca_from_full.csv")
            val_data = MyDataset(f"test_data_fold_{outer_fold}_with_pca_from_full.csv")
            clf.fit(train_data.x, train_data.y)
            predicts = clf.predict(val_data.x)
            temp_best_accu = 0
            for predict, gt in zip(predicts, val_data.y):
                if predict == gt[0]:
                    temp_best_accu += 1
            best_accu = temp_best_accu / len(val_data)
            # if temp_best_accu / len(val_data) > best_accu:
            #     best_accu = temp_best_accu / len(val_data)
            timet = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            best_model_name = f"svm/model_fold_{outer_fold}_inner_{i}_{timet}_pca_from_full.pth"
            best_model = pickle.dump(clf, open(best_model_name, "wb"))
            print(
                f"test best accu of outerfold {outer_fold} inner fold {i}: {best_accu}")
            print(
                f"best_model of outerfold {outer_fold} inner fold {i}: {best_model_name}")
        # print(type(predict))

        # print(predict.shape)
        # break

