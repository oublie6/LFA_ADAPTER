from s2swa import Seq2Seq, Encoder, Decoder, Attention, MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 生成模拟数据
dataset_size = 1000  # 数据集中的样本总数
src_data = torch.randn(dataset_size, 32)  # 源数据
trg_data = torch.randint(0, 2, (dataset_size,))  # 目标数据

# 创建TensorDataset和DataLoader
dataset = TensorDataset(src_data, trg_data)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和训练设施
device = torch.device('cpu')  # 或者 'cuda' 如果可用
INPUT_DIM = 32
HID_DIM = 16
N_LAYERS = 2
OUTPUT_DIM = 2  # 对于二分类任务

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS)
attention = Attention(HID_DIM, HID_DIM)
dec = Decoder(HID_DIM, OUTPUT_DIM, N_LAYERS, attention)
mlp = MLP(HID_DIM)
model = Seq2Seq(enc, dec, mlp, device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src_batch, trg_batch in data_loader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)
        optimizer.zero_grad()
        trg_input = torch.ones_like(trg_batch, dtype=torch.float).unsqueeze(1)
        output = model(src_batch, trg_input, src_batch.shape[1], trg_batch.shape[0])
        loss = criterion(output, trg_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Training loss: {total_loss/len(data_loader)}")