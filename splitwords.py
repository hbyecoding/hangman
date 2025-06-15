import random

with open('words_250000_train.txt', 'r') as f:
    words = f.read().splitlines()

random.shuffle(words)
split_idx = int(0.8 * len(words))
train_words = words[:split_idx]
val_words = words[split_idx:]

with open('train.txt', 'w') as f:
    f.write('\n'.join(train_words))
with open('val.txt', 'w') as f:
    f.write('\n'.join(val_words))