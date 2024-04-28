import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# 设定随机种子以便结果可重复
torch.manual_seed(0)
np.random.seed(0)

# 数据生成函数
def generate_data(batch_size, seq_len):
    # 随机生成序列数据和标签
    data = torch.randn(batch_size, seq_len, 10)  # 假设每个序列的特征维度为10
    labels = torch.randint(0, 2, (batch_size,))  # 二分类标签
    return data, labels

# LSTM + MLP 模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_hidden_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 2)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.mlp(x)
        return x

def main():

    # 初始化模型
    print("初始化模型")
    model = LSTMClassifier(input_dim=10, hidden_dim=50, mlp_hidden_dim=100, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 生成数据
    print("加载数据集")
    train_data, train_labels = generate_data(100, 30)  # 100个样本，每个样本长度为15
    test_data, test_labels = generate_data(20, 30)     # 测试数据

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=10, shuffle=True)

    # 训练模型
    for epoch in range(100):
        print(f"开始第{epoch}次训练")
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        time.sleep(60)
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


    # 测试模型
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == test_labels).float().mean()
        print(f'Test Accuracy: {accuracy.item()}')
