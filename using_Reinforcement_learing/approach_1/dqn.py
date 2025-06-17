import logging

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

# create logger
logger = logging.getLogger('root')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = None


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()       
        num_classes = 26
        num_layers = 2  # 增加 LSTM 层数
        input_size = 27
        hidden_size = 128  # 增加隐藏层大小
        seq_length = 27
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # 使用双向 LSTM
        self.lstm = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=num_layers, 
                          batch_first=True,
                          bidirectional=True)  # 使用双向 LSTM
        
        # 增加网络深度和宽度
        self.fc1 = nn.Linear(hidden_size * 2 + 26, 256)  # *2 是因为双向 LSTM
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.2)  # 添加 dropout
        self.relu = nn.ReLU()
    
    def forward(self, word, actions):
        # LSTM 层
        output, (hn, cn) = self.lstm(word.float())
        
        # 获取最后一层的隐藏状态
        hn = hn.view(self.num_layers, 2, -1, self.hidden_size)  # 2 是因为双向
        hn = hn[-1].view(-1, self.hidden_size * 2)  # 使用最后一层的输出
        
        # 连接 LSTM 输出和动作历史
        combined = torch.cat((hn, actions), 1)
        
        # 全连接层
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out
    