import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# 设置设备
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no, device):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # 将数据移动到GPU
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()                
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            loss_estimate.append(loss)
            batch_no.append(current)
            epoch_no.append(epoch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn, device):
    size = len(data_loader.dataset)
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in data_loader:
            # 将数据移动到GPU
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        return features, label

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.Sequential(
            nn.Embedding(64, 32, max_norm=1, norm_type=2),
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True),
            extract_tensor(),
            nn.Linear(128, 26)
        )
    
    def forward(self, x):
        logits = self.LSTM_stack(x)
        return logits

def create_dataloader(input_tensor, target_tensor, batch_size=128):
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return all_features_dataloader

def save_model(model, save_path="bi-lstm-embedding-model-state.pt"):
    torch.save(model.state_dict(), save_path)    

def train_model(input_tensor, target_tensor):
    # 获取设备
    device = get_device()
    
    # 创建数据加载器
    all_features_dataloader = create_dataloader(input_tensor, target_tensor)
    
    # 创建模型并移动到GPU
    model = NeuralNetwork().to(device)
    
    # 创建损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练参数
    loss_estimate = []
    batch_no = []
    epoch_no = []
    epochs = 8
    
    # 训练循环
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(all_features_dataloader, model, loss_fn, optimizer, loss_estimate, batch_no, t, epoch_no, device)
        test_loop(all_features_dataloader, model, loss_fn, device)
    
    print("Done!")
    
    # 保存模型
    save_model(model)
    
    # 保存训练历史
    history = {
        'loss_estimate': loss_estimate,
        'batch_no': batch_no,
        'epoch_no': epoch_no
    }
    torch.save(history, 'training_history.pt')
    
    return model, history
