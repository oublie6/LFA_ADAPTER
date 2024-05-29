import math
from test import sprint
import test

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score


# 编码器定义
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, src):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.n_layers, src.size(0), self.hid_dim).to(src.device)
        c0 = torch.zeros(self.n_layers, src.size(0), self.hid_dim).to(src.device)

        # 前向传播
        outputs, (hidden, cell) = self.lstm(src, (h0, c0))
        return outputs, hidden, cell


# 注意力机制定义
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs: (batch_size, seq_len, hid_dim)
        # hidden: (n_layers, batch_size, hid_dim)

        # 计算注意力权重
        attn_scores = torch.bmm(encoder_outputs, hidden[-1].unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


# 解码器定义
class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(hid_dim * 2, hid_dim, n_layers, batch_first=True)

    def forward(self, context, hidden, cell, seq_len):
        # 解码器输入
        input = context.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hid_dim)
        context = context.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hid_dim)
        input = torch.cat((input, context), dim=2)  # (batch_size, seq_len, hid_dim * 2)

        # 前向传播
        outputs, (hidden, cell) = self.lstm(input, (hidden, cell))
        return outputs


# MLP定义
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数
        )

    def forward(self, x):
        return self.mlp(x)


# Seq2Seq定义
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, seq_len, output_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hid_dim, n_layers)
        self.attention = Attention(hid_dim)
        self.decoder = Decoder(hid_dim, n_layers)
        self.seq_len = seq_len
        self.mlp = MLP(hid_dim * seq_len, output_dim)

    def forward(self, src):
        encoder_outputs, hidden, cell = self.encoder(src)
        context, attn_weights = self.attention(encoder_outputs, hidden)
        decoded = self.decoder(context, hidden, cell, self.seq_len)
        decoded = decoded.reshape(decoded.size(0), -1)  # 将 LSTM 输出展开成 (batch_size, hid_dim * seq_len)
        output = self.mlp(decoded)
        return output


# 参数设置
INPUT_DIM = 26
HID_DIM = 128
N_LAYERS = 1
SEQ_LEN = 16
OUTPUT_DIM = 2


def tranAndEvalueate(attack_src,normal_src):
    # 加载数据集
    attack_csv = './data/attack1.0.csv'
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

    normal_csv = './data/normal1.0.csv'
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

    # 划分训练集和测试集
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(INPUT_DIM, HID_DIM, N_LAYERS, SEQ_LEN, OUTPUT_DIM).to(device)
    criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sprint(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'seq2seq_model.pth')
    print("Model saved as seq2seq_model.pth")

    # 预测
    model.eval()
    all_targets = []
    all_predicted = []
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # 将输出转换为二进制标签
            total += targets.size(0)
            correct += (predicted == targets).sum().item() / 2  # 每个样本有两个标签，所以除以2
            all_targets.extend(targets.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

    # 计算混淆矩阵
    all_targets = torch.tensor(all_targets)
    all_predicted = torch.tensor(all_predicted)
    cm = confusion_matrix(all_targets.argmax(axis=1), all_predicted.argmax(axis=1))
    precision = precision_score(all_targets.argmax(axis=1), all_predicted.argmax(axis=1), average='binary')
    recall = recall_score(all_targets.argmax(axis=1), all_predicted.argmax(axis=1), average='binary')

    # 计算漏报率（FNR）和误报率（FPR）
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])

    print('Confusion Matrix:')
    print(cm)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'False Negative Rate (FNR): {fnr:.4f}')
    print(f'False Positive Rate (FPR): {fpr:.4f}')

tranAndEvalueate("attack0.2.csv","normal0.2.csv")