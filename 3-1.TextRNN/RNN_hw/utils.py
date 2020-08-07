import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from torch import nn

def load_data(path, is_train=False, is_test=False):
    if not (train or test):
        print('wrong, please enter train or test')
        return None
    if is_train:
        if 'training_label' in path:
            with open(path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip('\n').split(' ') for line in lines]
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
            return x, y
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                x = [line.strip('\n').split(' ') for line in lines]
            return x

    if is_test:
        with open(path, 'r') as f:
            lines = f.readlines()
            # 0,my dog ate our dinner . no , seriously ... he ate it .
            x = [''.join(line.strip('\n').split(',')[1:]).strip() for line in lines]
            x = [sen.split(' ') for sen in x] # x 是一个二维数组
        return x

def evaluation(outputs, labels):
    outputs[outpus >= 0.5] = 1
    outputs[outpus < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

class Preprocessor():
    def __init__(self, sentences, sen_len):
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = []
    
class TwitterDataset(pl.LightningDataModule):
    def __init__(self, path, args):
        super().__init__()
        self.path = path
        
    def prepare_data(self):
        #TODO 还有分词tokenize 需要处理
        self.train_x, self.train_y = load_data(os.path.join(path, 'training_label.txt'),is_train=True)
        self.test_x = load_data(os.path.join(path, 'testing_data.txt'), is_test=True)

    def setup(self):

        self.train_dataset = TensorDataset(torch.tensor(self.train_x, torch.long), torch.tensor(self.train_y, torch.long))
        self.test_dataset = TensorDataset(torch.tensor(self.test_x, torch.long))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size)
