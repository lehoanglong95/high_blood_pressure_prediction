from sklearn import linear_model
from my_dataset import NewMyDataset
import pickle
import torch
from datetime import datetime

if __name__ == '__main__':
    best_accu = 0
    best_model = ""
    for i in range(1, 5):
        clf = linear_model.Lasso(alpha=0.5)
        train_data = NewMyDataset(f"x_train_fold_{i}.npy", f"y_train_{i}.npy")
        val_data = NewMyDataset(f"x_val_fold_{i}.npy", f"y_val_{i}.npy")
        clf.fit(train_data.x, train_data.y)
        predicts = clf.predict(val_data.x)
        temp_best_accu = 0
        for predict, gt in zip(predicts, val_data.y):
            if predict == gt[0]:
                temp_best_accu += 1
        print(temp_best_accu / len(val_data))
        if temp_best_accu / len(val_data) > best_accu:
            best_accu = temp_best_accu / len(val_data)
            timet = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            best_model = f"lasso/model_fold_{i}_{timet}.pth"
            best_model = pickle.dump(clf, open(best_model, "wb"))
        # print(type(predict))

        # print(predict.shape)
        # break

