import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, batch_first=True)
        
    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, encoder_hid_dim, decoder_hid_dim):
        super().__init__()
        self.attention = nn.Linear(encoder_hid_dim + decoder_hid_dim, decoder_hid_dim)
        self.v = nn.Linear(decoder_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(0, 1, 2)
        
        # Calculate energy
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2))) 
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, hid_dim, output_dim, n_layers, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.LSTM((hid_dim * 2) + output_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    # def forward(self, input, hidden, cell, encoder_outputs):
    #     a = self.attention(hidden[-1], encoder_outputs)
    #     a = a.unsqueeze(1)
    #     encoder_outputs = encoder_outputs.permute(0, 1, 2)
    #     weighted = torch.bmm(a, encoder_outputs)
    #     weighted = weighted.permute(0, 2, 1)
    #     # 假设 input 是上一时间步的输出，其维度为 [batch_size, hid_dim]
    #     input = input.unsqueeze(1)  # 将 input 的维度改为 [batch_size, 1, hid_dim]
    #
    #     # 保持 weighted 的维度不变，即 [batch_size, 1, hid_dim]
    #     weighted = torch.bmm(a, encoder_outputs)  # bmm 结果为 [batch_size, 1, hid_dim]
    #
    #     # 现在 input 和 weighted 的维度一致，可以安全合并
    #     rnn_input = torch.cat((input, weighted), dim=2)  # 维度变为 [batch_size, 1, 2*hid_dim]
    #     output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
    #     prediction = self.fc_out(output.squeeze(1))
    #
    #     return prediction, hidden, cell
    def forward(self, input, hidden, cell, encoder_outputs):
        # 假设 input 已经是 [batch_size, feature_size] 形状，我们需要添加一个时间维度
        input = input.unsqueeze(1)  # 现在 input 的形状是 [batch_size, 1, feature_size]

        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(0, 1, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(0, 2, 1)

        # 确保 input 和 weighted 的最后一个维度一致
        rnn_input = torch.cat((input, weighted), dim=2)  # 现在两者都是 [batch_size, 1, feature_size]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 2)  # Binary classification
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, mlp, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp = mlp
        self.device = device

    def forward(self, src, trg, src_len, trg_len):
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Decoder
        output, hidden, cell = self.decoder(trg, hidden, cell, encoder_outputs)
        # Pass the decoder's output through MLP
        output = self.mlp(output)
        
        return output

# # Initialize components
# INPUT_DIM = 32
# HID_DIM = 16
# N_LAYERS = 16
# OUTPUT_DIM = 2  # For binary classification
#
# enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS)
# attention = Attention(HID_DIM, HID_DIM)
# dec = Decoder(HID_DIM, OUTPUT_DIM, N_LAYERS, attention)
# mlp = MLP(HID_DIM)  # The input dimension here should be aligned with the decoder's output dimension.
# model = Seq2Seq(enc, dec, mlp, device="cpu")
#
# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# # (续上文)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 假设我们有一个数据加载器，用于批量获取数据
# # for epoch in range(num_epochs):
# #     for src, trg in data_loader:
#
# # 模拟一个批次的数据
# src = torch.randn(10, 32)  # 假设每个批次有10个样本，每个样本的特征维度为32
# trg = torch.ones(10, dtype=torch.long)  # 假设目标是二分类问题，这里使用1表示所有样本的标签
#
# # 将数据转移到模型指定的设备上，在这个例子中是CPU
# src = src.to('cpu')
# trg = trg.to('cpu')
#
# model.train()  # 将模型设置为训练模式
#
# optimizer.zero_grad()  # 清零梯度
#
# # 假设的源数据长度和目标数据长度，根据实际情况调整
# src_len = src.shape[1]
# trg_len = trg.shape[0]
#
# # 由于我们的模型是针对序列到序列的任务，我们需要为目标数据创建一个与之对应的序列
# # 这里我们简化处理，直接使用一个全1的向量来模拟
# trg_input = torch.ones_like(trg, dtype=torch.float).unsqueeze(1)  # 增加一个维度，以匹配解码器输入的期望维度
#
# src_batch = src.unsqueeze(0)
#
# # 模型前向传播
# output = model(src_batch, trg_input, src_len, trg_len)
#
# # 计算损失，注意输出层的维度和损失函数的要求
# loss = criterion(output, trg)
#
# loss.backward()  # 反向传播
# optimizer.step()  # 参数更新
#
# print(f"Training loss: {loss.item()}")

