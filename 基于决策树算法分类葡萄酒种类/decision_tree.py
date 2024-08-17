import copy
import pandas as pd
import numpy as np
import math
import re
from draw_tree import createPlot
import matplotlib.pyplot as plt


# 导入数据集并拼接为一个data_frame, 颜色标签：red=1，white=0, 并随即生成训练集与测试集
def load_data(df1, df2, get_feature):
    df_red = pd.read_csv(df1, sep=";")
    df_red.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                      "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    df_red["type"] = 1
    df_red = df_red[get_feature]

    df_white = pd.read_csv(df2, sep=";")
    df_white.columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
                        "quality"]
    df_white["type"] = 0
    df_white = df_white[get_feature]

    df_red = np.array(df_red)
    df_white = np.array(df_white)

    df_red_train = df_red[:int(df_red.shape[0] * 0.8), :]
    df_red_test = df_red[int(df_red.shape[0] * 0.8) + 1:-1, :]
    df_white_train = df_white[:int(df_white.shape[0] * 0.8), :]
    df_white_test = df_white[int(df_white.shape[0] * 0.8) + 1:-1, :]

    train_set = np.concatenate((df_red_train, df_white_train))
    test_set = np.concatenate((df_red_test, df_white_test))
    train_set = train_set.tolist()
    test_set = test_set.tolist()
    value_list = get_feature[:-1]

    return train_set, test_set, value_list


# 定义信息熵函数
def entropy(data):
    label_list = {}
    for value in data:
        if value[-1] in label_list:
            label_list[value[-1]] += 1
        else:
            label_list[value[-1]] = 1
    info_en = 0
    for label in label_list:
        info_en -= label_list[label] / len(data) * math.log2((label_list[label] / len(data)))
    return info_en


class decision_tree:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tree = {}

    # 选定特征划分数据集。参数：axis:划分的特征；value：特征值；data_relationship:value与两侧的关系，1">=" or 0 "<"
    def split_data(self, data, axis, value, data_relationship):
        new_data = []
        for item in data:
            if data_relationship == 1:
                if item[axis] >= value:
                    temp = item[:axis]
                    temp.extend(item[axis + 1:])
                    new_data.append(temp)
            if data_relationship == 0:
                if item[axis] < value:
                    temp = item[:axis]
                    temp.extend(item[axis + 1:])
                    new_data.append(temp)
        return new_data

    def find_best_feature(self, data, label):
        base_entropy = entropy(data)
        base_gain_rate = 0.0
        best_feature_id = -1
        # split_dict用于存放连续性变量最佳划分点的具体值
        split_dict = {}
        for i in range(len(data[0]) - 1):
            feature_value = [item[i] for item in data]
            best_split = -1
            sorted_feature_value = sorted(feature_value)
            split_list = []
            for j in range(len(feature_value) - 1):
                split_list.append((sorted_feature_value[j] + sorted_feature_value[j + 1]) / 2.0)
            # 便利划分点，计算信息熵
            # 计算信息增益率 = 信息增益 / 该划分方式的分裂信息
            for j in range(len(split_list)):
                temp_entropy = 0.0
                value = split_list[j]
                l_data = self.split_data(data, i, value, 0)
                s_data = self.split_data(data, i, value, 1)
                p_l = len(l_data) / float(len(data))
                p_s = len(s_data) / float(len(data))
                if p_l != 0 and p_s != 0:
                    temp_entropy += p_l * entropy(l_data) + p_s * entropy(s_data)
                    # 计算该划分方式的分裂信息
                    infor_split = -p_l * math.log(p_l, 2) - p_s * math.log(p_s, 2)
                    gain_rate = float(base_entropy - temp_entropy) / infor_split
                    if gain_rate > base_gain_rate:
                        base_gain_rate = gain_rate
                        best_split = j
                        best_feature_id = i
            split_dict[label[i]] = split_list[best_split]
        best_feature_value = split_dict[label[best_feature_id]]
        return best_feature_id, best_feature_value

    # 计算类别的出现次数，并返回出现最多的类别标签
    def label_max(self, label):
        label_item = dict([(label.count(i), i) for i in label])
        return label_item[max(label_item.keys())]

    def build_tree(self, temp_data, temp_label):
        cate_list = [item[-1] for item in temp_data]
        # 如果分类类别完全相同，则结束
        if cate_list.count(cate_list[0]) == len(cate_list):
            return cate_list[0]
        # 遍历完所有特征，返回出现最多的类别标签
        if len(temp_data[0]) == 1:
            return self.label_max(cate_list)
        best_id, best_value = self.find_best_feature(temp_data, temp_label)
        if best_id == -1:
            return self.label_max(cate_list)
        # 对于连续性特征，不删除特征并构建大于/小于两条子树
        best_feature_label = temp_label[best_id] + "<" + str(best_value)
        tree = {best_feature_label: {}}
        sub_label = temp_label[:]
        value_sub_left = "True"
        tree[best_feature_label][value_sub_left] = self.build_tree(self.split_data(temp_data, best_id, best_value, 0),
                                                                   sub_label)
        value_sub_right = "False"
        tree[best_feature_label][value_sub_right] = self.build_tree(self.split_data(temp_data, best_id, best_value, 1),
                                                                    sub_label)
        return tree

    def train(self):
        label = copy.deepcopy(self.label)
        self.tree = self.build_tree(self.data, label)
        return self.tree


# 测试模型，返回分类结果
def classify(result_tree, label, test):
    root = list(result_tree.keys())[0]
    symbol_id = str(root).find("<")
    root_label = str(root)[:symbol_id]
    next_dict = result_tree[root]
    feature_id = label.index(root_label)
    class_label = None
    for key in next_dict.keys():
        temp_value = float(str(root)[symbol_id + 1:])
        if test[feature_id] < temp_value:
            if type(next_dict["True"]).__name__ == 'dict':
                class_label = classify(next_dict["True"], label, test)
            else:
                class_label = next_dict["True"]
        else:
            if type(next_dict["False"]).__name__ == 'dict':
                class_label = classify(next_dict["False"], label, test)
            else:
                class_label = next_dict["False"]
    return class_label


def st_data(data, axis, value, data_relationship):
    new_data = []
    for item in data:
        if data_relationship == 1:
            if item[axis] >= value:
                temp = item[:axis]
                temp.extend(item[axis + 1:])
                new_data.append(temp)
        if data_relationship == 0:
            if item[axis] < value:
                temp = item[:axis]
                temp.extend(item[axis + 1:])
                new_data.append(temp)
    return new_data


# 测试决策树正确率
def testing(tree, label, test):
    error = 0.0
    for k in range(len(test)):
        temp = classify(tree, label, test[k])
        if int(temp) != test[k][-1]:
            error += 1
    return float(error)


# 测试节点正确率
def testing_major(major, test):
    error = 0.0
    for i in range(len(test)):
        if int(major) != test[i][-1]:
            error += 1
    return float(error)


# 为提升模型的泛化能力，进行后剪枝操作
def post_cut_leaf(tree, train, test, label):
    root = list(tree.keys())[0]
    next_dict = tree[root]
    class_list = [item[-1] for item in train]
    feature_key = re.compile("(.+<)").search(root).group()[:-1]
    feature_value = float(re.compile("(<.+)").search(root).group()[1:])
    label_id = label.index(feature_key)
    temp_label = copy.deepcopy(label)
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            if key == 'True':
                sub_data = st_data(train, label_id, feature_value, '0')
                sub_test = st_data(test, label_id, feature_value, '0')
            else:
                sub_data = st_data(train, label_id, feature_value, '1')
                sub_test = st_data(test, label_id, feature_value, '1')
        if len(sub_test) > 0:
            tree[root][key] = post_cut_leaf(next_dict[key], sub_data, sub_test, copy.deepcopy(label))
    if testing(tree, temp_label, test) <= testing_major(max(class_list, key=class_list.count), test):
        print(tree)
        return tree
    return max(class_list, key=class_list.count)


g_feature = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "type"]
train_set, test_set, label_set = load_data(
    r'.\winequality-red.csv',
    r'.\winequality-white.csv', g_feature)
demo = decision_tree(train_set, label_set).train()
createPlot(demo)
print(demo)
count = 0
for i in range(len(test_set)):
    result = classify(demo, label_set, test_set[i])
    if int(result) == test_set[i][-1]:
        count += 1
print(count/float(len(test_set)))
after = post_cut_leaf(demo, train_set, test_set, label_set)
print(after)

# score_list = [0.756, 0.890, 0.878, 0.891, 0.907, 0.947]
# num_list = [3, 4, 5, 6, 7, 8]
# plt.plot(num_list, score_list, 'r-.p')
# plt.xlabel('Number of parameters')
# plt.ylabel('Accuracy')
# plt.title('The relationship between the number of parameters and the accuracy')
# plt.show()