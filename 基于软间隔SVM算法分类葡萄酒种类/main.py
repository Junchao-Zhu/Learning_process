import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


# 导入数据集并拼接为一个data_frame, 颜色标签：red=1，white=-1, 并随即生成训练集与测试集
def load_data(df1, df2):
    df_red = pd.read_csv(df1, sep=";")
    df_red.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                      "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    df_red["type"] = 1

    df_white = pd.read_csv(df2, sep=";")
    df_white.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
                        "quality"]
    df_white["type"] = -1

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


# 定义SVM类
class SVM(object):
    """
    参数解释：
    data: 训练的数据集
    label: 标签
    C：惩罚系数
    b: bias
    kernel：选择的核函数与核函数初始化参数
    alpha: 拉格朗日乘子
    error: 误差值
    kernel: 核函数的值
    """

    def __init__(self, data, label, epoch=500, C=0.5, kernel_opt=("linear", 0.5)):
        self.data = data
        self.label = label
        self.C = C
        self.b = 0
        self.epoch = epoch
        self.kernel_opt = kernel_opt
        num_data = self.data.shape[0]
        self.alpha = np.zeros((num_data, 1))
        self.error = np.zeros((num_data, 2))
        self.kernel = None

    """
    初始化核函数: 可用核函数包括线性核函数，高斯核函数和多项式核函数
    init_kernel_value：返回各个样本之间的核函数值(value_shape = num_data*1)
    init_kernel：返回样本总体的核函数值(value_shape = num_data*num_data)
    """

    def init_kernel_value(self, temp_data):
        num_data = self.data.shape[0]
        kernel_init_value = self.kernel_opt[1]
        kernel_i = np.zeros((num_data, 1))
        if self.kernel_opt[0] == "linear":
            kernel_i = np.dot(self.data, temp_data.T)
        elif self.kernel_opt[0] == "rbf":
            for i in range(num_data):
                temp_diff = self.data[i, :] - temp_data
                kernel_i[i] = np.exp(np.dot(temp_diff, temp_diff.T) / (-2.0 * kernel_init_value ** 2))
            kernel_i = np.squeeze(kernel_i)
        elif self.kernel_opt[0] == "polynomial":
            for i in range(num_data):
                kernel_i[i] = (np.dot(self.data[i, :], temp_data.T) + 1) ** kernel_init_value
            kernel_i = np.squeeze(kernel_i)
        else:
            print("Available kernel choice: linear, rbf and polynomial")
        return kernel_i

    def init_kernel(self):
        num_data = self.data.shape[0]
        kernel = np.zeros((num_data, num_data))
        for i in range(num_data):
            kernel[:, i] = self.init_kernel_value(self.data[i, :])
        return kernel

    """
    cal_error：计算第i个alpha对应误差值
    choose_second_alpha：选择第二个变量并返回第二个变量的误差
    update_error：计算更新误差值
    choose_update_alpha：选择并更新两个alpha, b, error
    """

    def cal_error(self, i_index):
        temp_prediction = float(np.dot(np.multiply(self.alpha, self.label).T, self.kernel[:, i_index]) + self.b)
        temp_error = temp_prediction - float(self.label[i_index])
        return temp_error

    def choose_second_alpha(self, a_index, a_error):
        self.error[a_index] = [1, a_error]
        choice_list = np.nonzero(self.error[:, 0])[0]
        best_gap = 0
        b_error = 0
        b_index = -1
        # 若有多个备选的变量
        if len(choice_list) > 1:
            for temp_index in choice_list:
                if temp_index == a_index:
                    continue
                temp_error = self.cal_error(temp_index)
                if abs(temp_error - a_error) > best_gap:
                    best_gap = abs(temp_error - a_error)
                    b_index = temp_index
                    b_error = temp_error
        # 若只有一个或没有备选变量，则随机选择第二个变量
        else:
            b_index = a_index
            while b_index == a_index:
                b_index = np.random.randint(self.data.shape[0])
            b_error = self.cal_error(b_index)
        return b_index, b_error

    def update_error(self, i):
        temp_error = self.cal_error(i)
        self.error[i] = [i, temp_error]

    def choose_update_alpha(self, a_index):
        a_error = self.cal_error(a_index)
        if (self.alpha[a_index] < self.C) or (self.alpha[a_index] > 0):
            b_index, b_error = self.choose_second_alpha(a_index, a_error)
            last_a_alpha = self.alpha[a_index].copy()
            last_b_alpha = self.alpha[b_index].copy()
            # 计算上下界
            if self.label[a_index] != self.label[b_index]:
                l = max(0, self.alpha[b_index] - self.alpha[a_index])
                h = min(self.C, self.C + self.alpha[b_index] - self.alpha[a_index])
            else:
                l = max(0, self.alpha[b_index] + self.alpha[a_index] - self.C)
                h = min(self.C, self.alpha[b_index] + self.alpha[a_index])
            if l == h:
                return 0
            # 计算eta值
            eta = self.kernel[a_index, a_index] + self.kernel[b_index, b_index] - 2.0*self.kernel[a_index, b_index]
            if eta <= 0:
                return 0
            # 更新第二个alpha，根据范围约束最终的alpha(b)
            self.alpha[b_index] += self.label[b_index] * (a_error - b_error) / eta
            if self.alpha[b_index] > h:
                self.alpha[b_index] = h
            if self.alpha[b_index] < l:
                self.alpha[b_index] = l
            # 若第二个alpha值不在改变，则结束
            if abs(last_b_alpha - self.alpha[b_index]) < 1e-5:
                self.update_error(b_index)
                return 0
            # 更新第一个alpha值
            self.alpha[a_index] += self.label[a_index] * self.label[b_index] * (last_b_alpha - self.alpha[b_index])
            # 更新b
            b1 = self.b - a_error - self.label[a_index] * self.kernel[a_index, a_index] * (self.alpha[a_index] - last_a_alpha) - self.label[b_index] * self.kernel[a_index, b_index] * (self.alpha[b_index] - last_b_alpha)
            b2 = self.b - b_error - self.label[a_index] * self.kernel[a_index, b_index] * (self.alpha[a_index] - last_a_alpha) - self.label[b_index] * self.kernel[b_index, b_index] * (self.alpha[b_index] - last_b_alpha)
            if 0 < self.alpha[a_index] < self.C:
                self.b = b1
            elif 0 < self.alpha[b_index] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            # 更新error
            self.update_error(a_index)
            self.update_error(b_index)
            return 1
        else:
            return 0

    def train(self):
        all_dataset = True
        self.kernel = self.init_kernel()
        num_data = self.data.shape[0]
        pair_change = 0
        iteration = 0
        while (iteration < self.epoch) and (pair_change > 0 or all_dataset):
            print("\t epoch", iteration)
            pair_change = 0
            if all_dataset:
                for i in range(num_data):
                    print(i)
                    pair_change += self.choose_update_alpha(i)
                iteration += 1
            else:
                boundary_item = []
                for i in range(num_data):
                    if 0 < self.alpha[i, 0] < self.C:
                        boundary_item.append(i)
                for j in boundary_item:
                    pair_change += self.choose_update_alpha(j)
                iteration += 1
            if all_dataset:
                all_dataset = False
            elif pair_change == 0:
                all_dataset = True
        return self

    def predict(self, test_data_i):
        kernel_value = self.init_kernel_value(test_data_i)
        predict = np.dot(np.multiply(self.label, self.alpha).T, kernel_value) + self.b
        return predict

    def cal_acc(self, test_data, test_label):
        num_data = test_data.shape[0]
        count = 0.0
        for i in range(num_data):
            temp_pred = self.predict(test_data[i, :])
            if np.sign(temp_pred) == np.sign(test_label[i]):
                count += 1
        accuracy = 1.0 * count / num_data
        return accuracy


# 导入数据,进行分类问题处理
red_wine = r".\winequality-red.csv"
white_wine = r".\winequality-white.csv"

train_data, test_data = load_data(red_wine, white_wine)

tr_data = train_data[:, :train_data.shape[1]-2]
tr_label = train_data[:, train_data.shape[1]-1:]
te_data = test_data[:, :test_data.shape[1]-2]
te_label = test_data[:, test_data.shape[1]-1:]

normalize(tr_data)
normalize(te_data)

demo = SVM(tr_data, tr_label, kernel_opt=("rbf", 0.6))
demo_train = demo.train()
acc = demo.cal_acc(tr_data, tr_label)
print(acc)

# acc_kernel_choice = []
# t = []
# for a in [("rbf", 0.6), ("linear", 0.6), ("polynomial", 0.6)]:
#     time_start = time.time()
#     demo = SVM(tr_data, tr_label, kernel_opt=a)
#     demo_train = demo.train()
#     acc = demo.cal_acc(te_data, te_label)
#     time_end = time.time()
#     time_sum = time_end - time_start
#     acc_kernel_choice.append(acc)
#     t.append(time_sum)
#
# acc_c_choice = []
# for b in [0.05, 0.1, 0.3, 0.5, 1, 2, 5]:
#     d = SVM(tr_data, tr_label, C=b, kernel_opt=("rbf", 0.6))
#     d_train = d.train()
#     acc = d.cal_acc(te_data, te_label)
#     acc_c_choice.append(acc)
#
# acc_epoch = []
# for c in range(1, 10):
#     d1 = SVM(tr_data, tr_label, epoch=c, kernel_opt=("rbf", 0.6))
#     d1_train = d1.train()
#     acc = d1.cal_acc(te_data, te_label)
#     acc_epoch.append(acc)

