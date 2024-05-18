import pandas as pd

# 加载数据集
import torch

csv_file = './data/attack.csv'
data=pd.read_csv(csv_file)

print(len(data))

print(data.iloc[0].tolist())

exit(0)

x_train=list()
shujudian=list()
shixu=list()
for i in range(len(data)):
    if i != 0 and i % 97==0:
        shixu=list()
    if i != 0 and i % 6==0:
        shujudian=list()
    shujudian=shujudian+data.iloc[i].tolist()

