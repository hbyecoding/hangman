# train_hangman_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

alphabet = 'abcdefghijklmnopqrstuvwxyz'
char2idx = {ch: idx for idx, ch in enumerate(alphabet)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
mask_idx = len(alphabet)  # 26
PAD_IDX = 27
MAX_LEN = 20


def mask_word(word, n_mask=1):
    # 随机mask n_mask个字母
    word = list(word)
    idxs = [i for i in range(len(word))]
    random.shuffle(idxs)
    mask_pos = idxs[:n_mask]
    masked = [ch if i not in mask_pos else '_' for i, ch in enumerate(word)]
    return ''.join(masked), [word[i] for i in mask_pos], mask_pos

# class HangmanDataset(Dataset):
#     def __init__(self, wordlist, n_mask=1):
#         self.samples = []
#         for word in wordlist:
#             if len(word) < 4: continue
#             masked, targets, mask_pos = mask_word(word, n_mask)
#             if '_' not in masked: continue
#             self.samples.append((masked, targets[0], mask_pos[0]))
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         masked, target, mask_pos = self.samples[idx]
#         # encode masked word
#         x = [mask_idx if ch == '_' else char2idx[ch] for ch in masked]
#         y = char2idx[target]
#         return torch.tensor(x), torch.tensor(y)

class HangmanDataset(Dataset):
    def __init__(self, wordlist, n_mask=1):
        self.samples = []
        for word in wordlist:
            if len(word) < 4 or len(word) > MAX_LEN: continue
            masked, targets, mask_pos = mask_word(word, n_mask)
            if '_' not in masked: continue
            self.samples.append((masked, targets[0], mask_pos[0]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        masked, target, mask_pos = self.samples[idx]
        x = [mask_idx if ch == '_' else char2idx[ch] for ch in masked]
        x = x + [PAD_IDX] * (MAX_LEN - len(x))
        y = char2idx[target]
        return torch.tensor(x), torch.tensor(y)


# class HangmanLSTM(nn.Module):
#     def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_dim)
#         self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size-1)  # 不预测mask
#     def forward(self, x):
#         emb = self.embedding(x)
#         out, _ = self.lstm(emb)
#         out = out[:, -1, :]
#         return self.fc(out)
class HangmanLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size-2)  # 不预测mask和pad
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out[:, -1, :]
        return self.fc(out)

def load_words(path):
    with open(path) as f:
        return [w.strip() for w in f if w.strip().isalpha()]

def train():
    train_words = load_words('hangman/train.txt')
    val_words = load_words('hangman/val.txt')
    train_ds = HangmanDataset(train_words)
    val_ds = HangmanDataset(val_words)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = HangmanLSTM(vocab_size=27)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"Epoch {epoch+1}, Val acc: {correct/total:.3f}")
    torch.save(model.state_dict(), 'hangman/hangman_lstm.pth')
    print("Model saved to hangman/hangman_lstm.pth")

if __name__ == "__main__":
    train()