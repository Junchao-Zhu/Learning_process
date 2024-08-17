import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def make_data(file_path):
    df = pd.read_csv(file_path, sep='	', header=None)
    df.columns = ['area', 'perimeter', 'compactness', 'kernel_len', 'kernel_width', 'asy_cof', 'k_groove_len', 'label']
    df = np.array(df)
    data = df[:, :-1]
    label = df[:, -1] - 1
    return data, label


# 特征归一化
mu = []
std = []


def normalize(data):
    for i in range(0, data.shape[1] - 2):
        data[:, i] = ((data[:, i] - np.mean(data[:, i])) / np.std(data[:, i]))
        mu.append(np.mean(data[:, i]))
        std.append(np.std(data[:, i]))
    return data


class kMeans:
    """
    本模型中使用k-means有两个用途：（1）比较与gmm模型之间的聚类准确性，（2）为gmm初始化参数
    """

    def __init__(self, data, epoch, k):
        self.data = data
        self.k = k
        self.epoch = epoch
        self.center = None

    def init_center(self):
        data_num, feature_num = self.data.shape
        self.center = np.zeros((self.k, feature_num))
        # 从数据集中随机确定K个点
        rand_id = np.random.randint(0, data_num, self.k)
        for i in range(self.k):
            self.center[i, :] = self.data[rand_id[i], :]
        return self.center

    def cluster_step(self, center):
        data_num, feature_num = self.data.shape
        temp_result = np.zeros(data_num)
        for i in range(data_num):
            min_dis = 10000
            for j in range(self.k):
                temp_dis = np.sum((self.data[i, :] - center[j, :]) ** 2)
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    temp_result[i] = j
        return temp_result

    def renew_center(self, temp):
        data_num, feature_num = self.data.shape
        center = np.zeros((self.k, feature_num))
        for i in range(self.k):
            temp_class = np.where(temp == i)
            center[i, :] = (np.sum(self.data[temp_class, :], axis=1) / len(temp_class[0])).ravel()
        return center

    def train(self):
        init_center = self.init_center()
        center = init_center
        final_result = []
        for i in range(self.epoch):
            final_result = self.cluster_step(center)
            center = self.renew_center(final_result)
        return final_result


class GMM:
    """
    参数解释：k为高斯分布的个数（聚类数量)
    mean:高斯分布的均值向量
    cov：高斯分布的协方差矩阵
    data_dim:数据的特征数量
    data_num：数据的数量
    weight：高斯分布的初始权重
    k_init:是否需要用k-means初始化。
    高斯混合模型参数初始化中，可以选择用k-means初始化高斯分布，
    也可以随机生成高斯分布用于后续训练，默认随机初始化。
    """

    def __init__(self, data, Epoch, K, k_init=False, Delta=0.000001):
        self.Data = data
        self.Epoch = Epoch
        self.Data_num = data.shape[0]
        self.Feature_num = data.shape[1]
        self.K = K
        self.Delta = Delta
        self.k_init = k_init

        if not self.k_init:
            self.weight = np.random.rand(self.K)
            self.weight /= np.sum(self.weight)

            self.mean = []
            for i in range(self.K):
                temp_mean = np.random.rand(self.Feature_num)
                self.mean.append(temp_mean)

            self.cov = []
            for i in range(self.K):
                temp_cov = np.random.rand(self.Feature_num, self.Feature_num)
                self.cov.append(temp_cov)
        else:
            init_result = kMeans(self.Data, 100, 3).train()
            store = defaultdict(list)
            for num_id, label in enumerate(init_result):
                store[label].append(num_id)
            self.weight = []
            self.mean = []
            self.cov = []
            for num_id in store.values():
                temp_data = self.Data[num_id]
                self.weight.append(len(num_id) / self.Data_num)
                self.mean.append(temp_data.mean(axis=0))
                self.cov.append(np.cov(temp_data.T))
            self.mean = np.array(self.mean)
            self.weight = np.array(self.weight)
            self.cov = np.array(self.cov)

    def get_guass(self, x, mean, cov):
        """
        这是自定义的高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """
        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        # 概率密度
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def train(self):
        temp_p = 0
        last_p = 1
        # guass_p表示第i个样本属于第k类高斯分布的概率
        guass_p = [np.zeros(self.K) for i in range(self.Data_num)]

        # 利用EM算法优化模型参数，终止条件（1）对数似然概率不再改变；（2）达到最大训练次数
        for i in range(self.Epoch):
            if abs(temp_p - last_p) < self.Delta:
                break
            else:
                last_p = temp_p
                # E-step:估计混合高斯模型中的参数
                for j in range(self.Data_num):
                    post_weight = [self.weight[k] * self.get_guass(self.Data[j], self.mean[k], self.cov[k])
                                   for k in range(self.K)]
                    post_weight = np.array(post_weight)
                    guass_p[j] = post_weight / np.sum(post_weight)

                # M-step:最大化估计参数
                for k in range(self.K):
                    # 计算N个样本中属于第k类的数量，并更新高斯分布的概率、均值与协方差矩阵
                    num_k = np.sum([guass_p[q][k] for q in range(self.Data_num)])
                    self.weight[k] = (1.0 * num_k) / self.Data_num
                    self.mean[k] = (1.0 / num_k) * np.sum([guass_p[q][k] * self.Data[q] for q in range(self.Data_num)],
                                                          axis=0)
                    temp_diff = self.Data - self.mean[k]
                    self.cov[k] = (1.0 / num_k) * np.sum(
                        [guass_p[q][k] * temp_diff[q].reshape((self.Feature_num, 1)).dot(
                            temp_diff[q].reshape((1, self.Feature_num))) for q in range(self.Data_num)], axis=0)

                # 计算此时的对数似然概率，若小于预设值Delta，则符合条件1，结束
                temp_p = []
                for n in range(self.Data_num):
                    cur_p = [np.sum(self.weight[k] * self.get_guass(self.Data[n], self.mean[k], self.cov[k])) for k in
                             range(self.K)]
                    if cur_p != 0:
                        cur_p = np.log(np.array(cur_p))
                        temp_p.append(cur_p)
                temp_p = np.sum(temp_p)

        # 输出聚类结果
        for i in range(self.Data_num):
            guass_p[i] = guass_p[i] / np.sum(guass_p[i])
        result = [np.argmax(guass_p[i]) for i in range(self.Data_num)]
        return result


def acc(pred):
    pred1 = pred[:70]
    pred2 = pred[70:140]
    pred3 = pred[140:]
    t_1 = [np.sum(pred1 == 0), np.sum(pred1 == 1), np.sum(pred1 == 2)]
    t_2 = [np.sum(pred2 == 0), np.sum(pred2 == 1), np.sum(pred2 == 2)]
    t_3 = [np.sum(pred3 == 0), np.sum(pred3 == 1), np.sum(pred3 == 2)]
    t = np.array([t_1, t_2, t_3])
    temp = []
    print(t)
    for i in range(3):
        max_index = list(np.unravel_index(t.argmax(), t.shape))
        temp.append(max_index)
        t[max_index[0], :] = -1
        t[:, max_index[1]] = -1
    s = 0
    for i in temp:
        print(i)
        s += np.sum(pred[i[1]*70:(i[1]*70+70)] == i[0])
    s /= 210
    return s


train_data, train_label = make_data(
    r'.\seeds_dataset.txt')
train_data = normalize(train_data)
# result = GMM(train_data, 1000, 3).train()
# print(result)

e_count = []
acc_ = []
for i in range(6):
    e_count.append(i*50)
    result = GMM(train_data, i*50, 3, k_init=True).train()
    acc_.append(acc(result))

plt.figure(figsize=(20, 10), dpi=100)
plt.plot(e_count, acc_, c='blue')
plt.scatter(e_count, acc_, c='blue')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("epoch", fontdict={'size': 16})
plt.ylabel("acc", fontdict={'size': 16})
plt.title("gmm_no_k_init", fontdict={'size': 20})
plt.show()

# result = kMeans(train_data, 10, 3).train()
# r = acc(result)
# print(result)
