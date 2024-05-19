import pandas as pd
import math
import torch

# 加载数据集
attack_csv = './data/attack.csv'
data = pd.read_csv(attack_csv, header=None)

x_train_attack = []
i = 0
while i < len(data):
    shixu = []
    for j in range(16):
        shujudian = []
        tmp = i + j * 6
        for k in range(4):
            shujudian += list(filter(lambda x: not math.isnan(x), data.iloc[k + tmp].tolist()))
        shujudian += list(filter(lambda x: not math.isnan(x), data.iloc[tmp + 5].tolist()))
        shixu.append(shujudian)
    x_train_attack.append(shixu)
    i += 97

y_train_attack = [[1, 0] for _ in x_train_attack]  # 使用 one-hot 编码

normal_csv = './data/normal.csv'
data = pd.read_csv(normal_csv, header=None)

x_train_normal = []
i = 0
while i < len(data):
    shixu = []
    for j in range(16):
        shujudian = []
        tmp = i + j * 6
        for k in range(4):
            shujudian += list(filter(lambda x: not math.isnan(x), data.iloc[k + tmp].tolist()))
        shujudian += list(filter(lambda x: not math.isnan(x), data.iloc[tmp + 5].tolist()))
        shixu.append(shujudian)
    x_train_normal.append(shixu)
    i += 97

y_train_normal = [[0, 1] for _ in x_train_normal]  # 使用 one-hot 编码

x_train = torch.tensor(x_train_attack + x_train_normal, dtype=torch.float32)
y_train = torch.tensor(y_train_attack + y_train_normal, dtype=torch.float32)