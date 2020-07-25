# code by Tae Hwan Jung @graykode
# modified by Chea Sim @CheaSim
import numpy as np #导入numpy包

import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # torch的优化器 有SGD Adam
from torch.utils.data import Dataset, DataLoader
import tensorboard


dtype = torch.FloatTensor
sentences = [ "i like dog", "i love coffee", "i hate milk"]


word_list = " ".join(sentences).split() # 为了提取每个句子中的词语，我们先将所有句子用" "分开之后
#在用split()将所有单词分开获得一个word_list
word_list = list(set(word_list)) #去重

#{} 是词典的意思 这两个一对一产生类似于map的功能
# word_ditc[word] = idx
# number_dict[idx] = word
word_dict = {w: i for (i, w) in enumerate(word_list)}
number_dict = {i: w for (i, w) in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper

class MyDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.x, self.y = self.make_batch(sentences)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def make_batch(self, sentences):
        input_batch = []
        target_batch = []

        for sen in sentences:
            # split the words into a list
            word = sen.split()
            # insert all wordds into input except the last one as the last one is the target
            input = [word_dict[n] for n in word[:-1]]
            # target is the id of the last word
            target = word_dict[word[-1]]

            # append a list to a list. conxxx to 
            # input_batch = [input, input, input]
            input_batch.append(input)
            target_batch.append(target)

        return torch.tensor(input_batch, dtype=int), torch.tensor(target_batch, dtype=int)


# Model
class NNLM(pl.LightningModule):
    def __init__(self, sentences):
        super().__init__()
        # Embedding 是一个函数 将一个word转化为高维向量的形式。
        # m = 2 表示输入是2维的input
        self.C = nn.Embedding(n_class, m) 
        self.l1 = nn.Linear(m*n_step, n_hidden)
        self.l3 = nn.Linear(m*n_step, n_class,bias = False)
        # self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        # self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        # self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.l2 = nn.Linear(n_hidden, n_class)
        # self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        # self.b = nn.Parameter(torch.randn(n_class).type(dtype))
        self.sentences = sentences

    def forward(self, X):
        # 第一步先编码
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]
        # tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        tanh = torch.tanh(self.l1(X))
        # output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        output = self.l3(X) + self.l2(tanh)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        train_dataset = MyDataset(self.sentences)
        train_loader = DataLoader(train_dataset,batch_size=3, shuffle=True)
        return train_loader


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def SHOW_ME(self):
        input_batch = []
        target_batch = []

        for sen in sentences:
            # split the words into a list
            word = sen.split()
            # insert all wordds into input except the last one as the last one is the target
            input = [word_dict[n] for n in word[:-1]]
            # target is the id of the last word
            target = word_dict[word[-1]]

            # append a list to a list. conxxx to 
            # input_batch = [input, input, input]
            input_batch.append(input)
            target_batch.append(target)

        input_batch = torch.tensor(input_batch, dtype=int)
        predict = self(input_batch).data.max(1, keepdim=True)[1]
        print(predict.squeeze())

        print(predict)
        # Test
        print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

        



# Training

def main():
    model = NNLM(sentences = sentences)
    trainer = pl.Trainer(max_epochs=5000, gpus=1)
    trainer.fit(model)
    model.SHOW_ME()

   

# model(input_batch) 获得概率函数， 之后取其中最大的。
if __name__ == '__main__':
    main()
