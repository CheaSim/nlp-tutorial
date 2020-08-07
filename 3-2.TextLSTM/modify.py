'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

dtype = torch.FloatTensor

# 这次是字符级别的LSTM，对于单词的最后一个进行预测
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
word_dict = {n: i for i, n in enumerate(char_arr)}
number_dict = {i: w for i, w in enumerate(char_arr)}
n_class = len(word_dict) # number of class(=number of vocab)

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

# TextLSTM Parameters
# 所有的词都是4个字母组成，所以time 为3
n_step = 3
n_hidden = 128

# batch的形式都是 batch[i] 是第i个batch
class MyDataSet(Dataset):
    def __init__(self, seq_data):
        self.X, self.Y = self.make_batch(seq_data)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def make_batch(self, seq_data):
        input_batch, target_batch = [], []

        for seq in seq_data:
            input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
            target = word_dict[seq[-1]] # 'e' is target
            input_batch.append(np.eye(n_class)[input])
            target_batch.append(target)

        return torch.Tensor(input_batch), torch.LongTensor(target_batch)



class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        # 第一层LSTMc层
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        # 第二层FCN全连接层进行预测
        self.l2 = nn.Linear(n_hidden, n_class)

    def forward(self, X):
        # X 就是一个batch
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1, len(X), n_hidden)   # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.l2(outputs)
        return model

train_dataset = MyDataSet(seq_data = seq_data)
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True)
model = TextLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training
# 这样就可以调整batch_size 和 是否打乱sample的顺序了
for epoch in range(1000):
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

inputs = [sen[:3] for sen in seq_data]
