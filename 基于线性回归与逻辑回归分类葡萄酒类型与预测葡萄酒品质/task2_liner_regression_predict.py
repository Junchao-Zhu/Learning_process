import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 导入数据集并拼接为一个data_frame, 颜色标签：red=1，white=0, 并随即生成训练集与测试集
def load_data(df1):
    df = pd.read_csv(df1, sep=";")
    df.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    df = np.array(df)

    train_set = df[:int(df.shape[0] * 0.8), :]
    test_set = df[int(df.shape[0] * 0.8) + 1:-1, :]

    return train_set, test_set


# 特征归一化
mu = []
std = []


def normalize(data):
    for i in range(0, data.shape[1] - 2):
        data[:, i] = ((data[:, i] - np.mean(data[:, i])) / np.std(data[:, i]))
        mu.append(np.mean(data[:, i]))
        std.append(np.std(data[:, i]))


# 回归预测
class liner_regression:
    def __init__(self, data, lr, epoch, label):
        self.lr = lr
        self.epoch = epoch
        self.data = np.hstack((np.ones((data.shape[0], 1)), data))
        self.weight = np.zeros((self.data.shape[1], 1))
        self.loss = []
        self.label = np.reshape(label, (label.shape[0], 1))

    def c_loss(self):
        return np.sum(np.dot((np.dot(self.data, self.weight) - self.label).T,
                      (np.dot(self.data, self.weight) - self.label)) / (2 * self.label.shape[0]))

    def train(self):
        for i in range(self.epoch):
            temp = np.dot(self.data, self.weight)
            cost = (1 / self.data.shape[0]) * np.dot(self.data.T, (temp - self.label))
            self.weight = self.weight - self.lr * cost
            self.loss.append(self.c_loss())
        return self

    def prediction(self, test):
        result = []
        temp = np.dot(test, self.weight[1:]) + self.weight[0]
        result.append(temp)
        return result


# 性能测试函数，包括：平均绝对误差(MAE）,均方误差（MSE）,均方根误差（RMSE）与R2_score
def MAE(label, pre):
    return np.mean(np.abs(label - pre))


def MSE(label, pre):
    return np.mean((label - pre) ** 2)


def RMSE(label, pre):
    return np.sqrt(MSE(label, pre))


def R2(label, pre):
    u = np.sum((label - pre) ** 2)
    v = np.sum((label - np.mean(label)) ** 2)
    return 1 - (u / v)


# 导入数据,进行回归问题处理
red_wine = r".\winequality-red.csv"
train_data, test_data = load_data(red_wine)

tr_data = train_data[:, :train_data.shape[1] - 1]
tr_label = train_data[:, train_data.shape[1] - 1:train_data.shape[1]]
te_data = test_data[:, :test_data.shape[1] - 1]
te_label = test_data[:, test_data.shape[1] - 1:test_data.shape[1]]

normalize(tr_data)
normalize(te_data)

demo = liner_regression(lr=0.001, epoch=250000, data=tr_data, label=tr_label).train()
result = demo.prediction(te_data)
result = np.array(result)
result = result.T[0]


# 对预测结果进行评估
te_label = te_label.T[0]
result = result.T[0]

print("MAE: ", MAE(te_label, result))
print("MAE: ", MSE(te_label, result))
print("RMSE: ", RMSE(te_label, result))
print("R^2: ", R2(te_label, result))

# 绘图部分
# e_MAE = []
# e_MSE = []
# e_RMSE = []
# e_R2 = []
# e_count = []
# for i in range(1, 25):
#     demo = liner_regression(lr=0.001, epoch=i*10000, data=tr_data, label=tr_label).train()
#     result = demo.prediction(te_data)
#     e_MAE.append(MAE(te_label, result))
#     e_MSE.append(MSE(te_label, result))
#     e_RMSE.append(RMSE(te_label, result))
#     e_R2.append(R2(te_label, result))
#     e_count.append(i*10000)
#
#
# plt.figure(figsize=(20, 10), dpi=100)
# plt.plot(e_count, e_MAE, c='red', label="MAE")
# plt.plot(e_count, e_MSE, c='green', linestyle='--', label="MSE")
# plt.plot(e_count, e_RMSE, c='blue', linestyle='-.', label="RMSE")
# plt.plot(e_count, e_R2, c='black', linestyle=':', label="R2")
# plt.scatter(e_count, e_MAE, c='red')
# plt.scatter(e_count, e_MSE, c='green')
# plt.scatter(e_count, e_RMSE, c='blue')
# plt.scatter(e_count, e_R2, c='black')
# plt.legend(loc='best')
# # plt.yticks(range(0, 50, 5))
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("epoch", fontdict={'size': 12})
# plt.ylabel("score", fontdict={'size': 12})
# plt.title("Liner_regression_score", fontdict={'size': 20})
# plt.show()