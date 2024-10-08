import torch.utils
import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MFCCDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)
	
# 对数据进行填充，使得每个序列长度一致
def pad_sequences(sequences, maxlen=None):
    lengths = [len(seq) for seq in sequences]
    max_len = maxlen or max(lengths)
    
    padded_sequences = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences, lengths

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # 打包序列以处理不同长度
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        # 仅使用最后一个时间步的隐藏状态
        out = self.fc(hn[-1])
        return out