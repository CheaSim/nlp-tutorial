'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
from pytorch_lightning import Trainer

#用Float作为数据类型
dtype = torch.float32

# 输入的句子 ， 小demo就搞三句话吧
sentences = [ "i like dog", "i love coffee", "i hate milk"]

# 常规操作 获取词表，词表id，id对应词，词的个数
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# TextRNN Parameter
batch_size = len(sentences)
n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell

# to Torch.Tensor
#input_batch, target_batch = make_batch(sentences)
# list 变成 Tensor
#input_batch = Variable(torch.Tensor(input_batch))
#target_batch = Variable(torch.LongTensor(target_batch))


class MyDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.X ,self.Y= self.make_batch(sentences)

    def __getitem__(self, index):
        return self.X, self.Y
    
    def __len__(self):
        return len(self.X)

    def make_batch(self, sentences):
        input_batch = []
        target_batch = []

        for sen in sentences:
            #获取每一个句子的wordlist
            word = sen.split()
            # 除了最后一个词语 都当做input
            # input 是 id
            input = [word_dict[w] for w in word[:-1]]
            # 最后一个词当做target
            target = word_dict[word[-1]]
            # input_batch 是每一个词的ont-hot encoding
            input_batch.append(np.eye(n_class)[input])
            print(input_batch)
            target_batch.append(target)

        return torch.tensor(input_batch, dtype=dtype), torch.tensor(target_batch, dtype=dtype)


# 主要的模型
class TextRNN(LightningModule):
    def __init__(self, hidden, sentences):
        super(TextRNN, self).__init__()

        # 直接继承了 pytorch自带的rnn网络
        # one-hot encoding embedding长度就是词表 = n_class
        # n_hidden 如何确定？
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.l2 = nn.Linear(n_hidden,n_class)
        self.hidden = hidden
        # 做成了batch的应该也可以作为x, y吧
        self.dataset = MyDataset(sentences)

    def forward(self, x):
        #????
        x = x.transpose(0, 1)
        x, _ = self.rnn(x)
        x = x[-1]
        x = self.l2(x)
        return x
        """
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        # 一层linear层
        model = torch.mm(outputs, self.W) + self.b # model : [batch_size, n_class]
        return model
        """

    def configure_optimizers(self):
        # 优化器用Adam
        return optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        # batch 是从dataloader 取出来的东西
        x, y = batch
        # 类似forward(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss' : loss}
        return {'loss' : loss, 'log' : tensorboard_logs}

    
    def train_dataloader(self):
        train_loader = DataLoader(self.dataset, batch_size = 2, num_workers= 2, shuffle= False)
        return train_loader

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):
        test_loader = DataLoader(self.dataset, batch_size = 2, num_workers= 2, shuffle= False)
        return test_loader


def main():
    model = TextRNN(hidden=n_hidden, sentences = sentences)
    trainer = Trainer(max_epochs=5000)
    trainer.fit(model)

    trainer.test()



if __name__ == '__main__':
    main()
# 用交叉熵作为损失函数

# Training
"""
for epoch in range(5000):
    optimizer.zero_grad()

    # hidden : [num_layers * num_directions, batch, hidden_size]
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    # input_batch : [batch_size, n_step, n_class]
    output = model(hidden, input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()
input = [sen.split()[:2] for sen in sentences]

# Predict
hidden = Variable(torch.zeros(1, batch_size, n_hidden))
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
"""