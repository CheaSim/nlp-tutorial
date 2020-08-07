'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  modified by Chea Sim @CheaSim
  这些项目就不搞什么args设定参数了，都是demo
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

dtype = torch.FloatTensor

# Text-CNN Parameter
embedding_size = 2 # n-gram 因为CNN 也用了个2*2 的kernal
sequence_length = 3 # 句子的长度，或者是最大长度
num_classes = 2  # 0 or 1  二分类问题， 正面或者是负面
filter_sizes = [2, 2, 2] # n-gram window 滑动窗口， 宽度为两个词的窗口
num_filters = 3 # filiter的个数

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)
# 数据处理函数 输出batch[0] batch[1]
def processer(sentences):
    inputs = []
    targets = []
    for sen in sentences:
        inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

    for out in labels:
        targets.append(out) # To using Torch Softmax Loss function

    return inputs, targets
class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms)
        self.prepare_data()
        self.setup()
    def prepare_data(self):
        self.inputs, self.outputs = processer(sentences)

    def setup(self):
        inputs = torch.tensor(self.inputs, dtype=torch.long)
        targets = torch.tensor(self.outputs, dtype=torch.long)

        self.train_dataset = TensorDataset(inputs, targets)
        self.test_dataset = TensorDataset(inputs, targets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=6)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=6)

class TextCNN(pl.LightningModule):
    def __init__(self, dropout = 0.2, lr = 0.001):
        super().__init__()

        self.num_filters_total = num_filters * len(filter_sizes)
        self.convs = nn.ModuleList([
                                        nn.Conv2d(in_channels = 1, 
                                                out_channels = num_filters, 
                                                kernel_size = (fs, embedding_size))
                                        for fs in filter_sizes
        ])
        self.embedding = nn.Embedding(vocab_size ,embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.num_filters_total, num_classes)
        self.lr = lr
    def forward(self, X):
        # embedded_chars.shape = [batch_size, 1, sent_len, embedding_size]
        embedded_chars = self.embedding(X).unsqueeze(1)
        # 更加优美的写法
        conved = [F.relu(conv(embedded_chars)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # for filter_size in filter_sizes:
        #     # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
        #     conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(embedded_chars)
        #     h = F.relu(conv)
        #     # mp : ((filter_height, filter_width))
        #     mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1))
        #     # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
        #     pooled = mp(h).permute(0, 3, 2, 1)
        #     pooled_outputs.append(pooled)
        cat = self.dropout(torch.cat(pooled, dim=1))
        model = self.fc(cat)
        return model

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)

        return result





# Test

def main():

    dm = MyDataModule()

    model = TextCNN()
     
    trainer = pl.Trainer(gpus=1, max_epochs=5000)
    trainer.fit(model, dm)


    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]

    test_batch = torch.tensor(tests, dtype=torch.long)
    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text,"is Bad Mean...")
    else:
        print(test_text,"is Good Mean!!")

if __name__ == '__main__':
    main()