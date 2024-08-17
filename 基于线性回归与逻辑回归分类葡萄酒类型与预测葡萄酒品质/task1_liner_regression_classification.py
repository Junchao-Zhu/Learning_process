import pandas as pd
import numpy as np
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


# 线性规划模型
class liner_regression:
    def __init__(self, data, lr, epoch, label):
        self.lr = lr
        self.epoch = epoch
        self.data = np.hstack((np.ones((data.shape[0], 1)), data))
        self.weight = np.zeros((self.data.shape[1], 1))
        self.loss = []
        self.label = np.reshape(label, (label.shape[0], 1))

    def c_loss(self):
        return np.dot((np.dot(self.data, self.weight)-self.label).T, (np.dot(self.data, self.weight)-self.label))/(2*self.label.shape[0])

    def train(self):
        for i in range(self.epoch):
            temp = np.dot(self.data, self.weight)
            cost = (1/self.data.shape[0])*np.dot(self.data.T, (temp-self.label))
            self.weight = self.weight-self.lr*cost
            self.loss.append(self.c_loss())
        return self

    def predict_classification(self, test):
        return np.where(np.dot(test, self.weight[1:]) + self.weight[0] >= 0.0, 1, 0)


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
demo = liner_regression(lr=0.01, epoch=2600, data=tr_data, label=tr_label).train()
result = demo.predict_classification(te_data)

count = 0
for i in range(result.shape[0]):
    if result[i] == te_label[i]:
        count += 1

acc = count/i
print(acc)

# # 绘图部分
# acc = []
# e_count = []
# for i in range(25):
#     demo = liner_regression(lr=0.01, epoch=i*100, data=tr_data, label=tr_label).train()
#     result = demo.predict_classification(te_data)
#
#     count = 0
#     for j in range(result.shape[0]):
#         if result[j] == te_label[j]:
#             count += 1
#
#     c_acc = count/j
#     acc.append(c_acc)
#     e_count.append(i*100)

# loss = demo.loss
# loss_c = []
# e_count = []
# for i in range(25):
#     loss_c.append(loss[i*100].T[0].T[0])
#     e_count.append(i*100)
#
# plt.figure(figsize=(20, 10), dpi=100)
# plt.plot(e_count, loss_c, c='blue')
# plt.scatter(e_count, loss_c, c='blue')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("epoch", fontdict={'size': 16})
# plt.ylabel("loss", fontdict={'size': 16})
# plt.title("liner_regression_loss", fontdict={'size': 20})
# plt.show()


