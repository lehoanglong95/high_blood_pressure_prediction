from sklearn.svm import SVC
from my_dataset import NewMyDataset, MyDataset
import pickle
import torch
from datetime import datetime

if __name__ == '__main__':
    for outer_fold in range(1, 5):
        for i in range(1, 5):
            best_accu = 0
            best_model = ""
            clf = SVC(kernel="linear", probability=True)
            # train_data = NewMyDataset(f"x_train_fold_{i}.npy", f"y_train_{i}.npy")
            # val_data = NewMyDataset(f"x_val_fold_{i}.npy", f"y_val_{i}.npy")
            train_data = MyDataset(f"train_fold_{outer_fold}_nest_fold_{i}.csv")
            val_data = MyDataset(f"val_fold_{outer_fold}_nest_fold_{i}.csv")
            clf.fit(train_data.x, train_data.y)
            predicts = clf.predict(val_data.x)
            print(list(predicts))
            gt = []
            for yy in val_data.y:
                gt.append(yy[0])
            print(gt)
            temp_best_accu = 0
            for predict, gt in zip(predicts, val_data.y):
                if predict == gt[0]:
                    temp_best_accu += 1
            # print(temp_best_accu / len(val_data))
            if temp_best_accu / len(val_data) > best_accu:
                best_accu = temp_best_accu / len(val_data)
                timet = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                best_model = f"svm/model_outter_fold_{outer_fold}_inner_fold_{i}_{timet}.pth"
                pickle.dump(clf, open(best_model, "wb"))
            print(f"outerfold {outer_fold} inner fold {i}: {best_accu} {best_model}")
        # print(type(predict))

        # print(predict.shape)
        # break

