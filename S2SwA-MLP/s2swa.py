import torch
import torch.nn as nn
import torch.optim as optim


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
            nn.Softmax(dim=1)
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

# 生成一些随机数据作为示例
x_train = torch.randn(1000, SEQ_LEN, INPUT_DIM)  # 1000 个样本，每个样本有 SEQ_LEN 个时间步
y_train = torch.randint(0, 2, (1000,)).long()  # 二分类标签，值为0或1

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(INPUT_DIM, HID_DIM, N_LAYERS, SEQ_LEN, OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
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

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'seq2seq_model.pth')
print("Model saved as seq2seq_model.pth")

# 预测
model.eval()
with torch.no_grad():
    # 生成一些随机数据作为测试
    x_test = torch.randn(10, SEQ_LEN, INPUT_DIM).to(device)
    predictions = model(x_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    print("Predicted labels:", predicted_labels.cpu().numpy())
