import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 导入数据集并拼接为一个data_frame, 颜色标签：red=1，white=0, 并随即生成训练集与测试集
def load_data(df1, df2):
    df_red = pd.read_csv(df1, sep=";")
    df_red.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                      "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    df_red["type"] = 1

    df_white = pd.read_csv(df2, sep=";")
    df_white.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
                        "quality"]
    df_white["type"] = 0

    df_red = np.array(df_red)
    df_white = np.array(df_white)

    df_red_train = df_red[:int(df_red.shape[0] * 0.8), :]
    df_red_test = df_red[int(df_red.shape[0] * 0.8) + 1:-1, :]
    df_white_train = df_white[:int(df_white.shape[0] * 0.8), :]
    df_white_test = df_white[int(df_white.shape[0] * 0.8) + 1:-1, :]

    train_set = np.concatenate((df_red_train, df_white_train))
    test_set = np.concatenate((df_red_test, df_white_test))

    return train_set, test_set


# 特征归一化
mu = []
std = []
def normalize(data):
    for i in range(0, data.shape[1] - 2):
        data[:, i] = ((data[:, i] - np.mean(data[:, i])) / np.std(data[:, i]))
        mu.append(np.mean(data[:, i]))
        std.append(np.std(data[:, i]))


# 定义逻辑回归模型
class logistic_regression:
    def __init__(self, lr, epoch, data):
        self.lr = lr
        self.epoch = epoch
        self.data = np.hstack([np.ones((len(data), 1)), data])
        self.weight = np.random.randn(self.data.shape[1])
        self.loss = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 定义loss function
    def j_loss(self, label):
        pred = self.sigmoid(np.dot(self.data, self.weight))
        return - np.sum(label * np.log(pred) + (1 - label) * np.log(1 - pred)) / label.shape[0]

    def d_j(self, label):
        pred = self.sigmoid(np.dot(self.data, self.weight))
        return np.dot(self.data.T, (pred-label))/label.shape

    # 权重梯度计算
    def train(self, label):
        """

        :rtype: object
        """
        for i in range(self.epoch):
            self.weight = self.weight - self.lr*self.d_j(label)
            self.loss.append(self.j_loss(label))
        return self

    # 分类结果
    def prediction_classification(self, test):
        test_in = np.hstack([np.ones((len(test), 1)), test])
        pred_out = self.sigmoid(np.dot(test_in, self.weight))
        pred_out = np.array(pred_out >= 0.5, dtype='int')
        return pred_out

    def f_acc(self, test, t_label):
        pred_out = self.prediction_classification(test)
        acc = np.sum(pred_out == t_label)/t_label.shape
        return acc


# 导入数据,进行分类问题处理
red_wine = r".\winequality-red.csv"
white_wine = r".\winequality-white.csv"

train_data, test_data = load_data(red_wine, white_wine)

tr_data = train_data[:, :train_data.shape[1]-2]
tr_label = train_data[:, train_data.shape[1]-1:]
te_data = test_data[:, :test_data.shape[1]-2]
te_label = test_data[:, test_data.shape[1]-1:]

tr_label = tr_label.T[0, :]
te_label = te_label.T[0, :]
# 特征归一化
normalize(tr_data)
normalize(te_data)

# 模型计算
demo = logistic_regression(lr=0.01, epoch=1000, data=tr_data).train(tr_label)
result = demo.prediction_classification(te_data)
acc = demo.f_acc(te_data, te_label)
print(acc)


# 画图
# acc = []
# e_count = []
# for i in range(50):
#     demo = logistic_regression(lr=0.01, epoch=i*100, data=tr_data).train(tr_label)
#     result = demo.prediction(te_data)
#     c_acc = demo.f_acc(te_data, te_label)
#     acc.append(c_acc)
#     e_count.append(i*100)

# loss = demo.loss
# loss_c = []
# e_count = []
# for i in range(25):
#     loss_c.append(loss[i*100])
#     e_count.append(i*100)
# plt.figure(figsize=(20, 10), dpi=100)
# plt.plot(e_count, loss_c, c='blue')
# plt.scatter(e_count, loss_c, c='blue')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("epoch", fontdict={'size': 16})
# plt.ylabel("loss", fontdict={'size': 16})
# plt.title("logistic_regression_loss", fontdict={'size': 20})
# plt.show()



