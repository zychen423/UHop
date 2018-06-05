import pickle
from collections import defaultdict
import itertools
import operator
from torch.utils.data import DataLoader, Dataset
import torch
import json
from functools import reduce
from itertools import accumulate
import random
import numpy as np

PATH = {}
PATH['wq'] = '../data/WQ/main_exp'
PATH['wq_train1test2'] = '../data/WQ/train1test2_exp'
PATH['sq'] = '../data/SQ'

from itertools import accumulate

def quick_collate(batch):
    return batch[0]

def random_split(dataset, train_p, valid_p):
    random.shuffle(dataset.data_objs)
    return Subset(dataset.data_objs[:int(len(dataset)*train_p)]), Subset(dataset.data_objs[int(len(dataset)*train_p):]) 

class Subset(Dataset):
    def __init__(self, data_objs):
        self.data_objs = data_objs
    def __getitem__(self, idx):
        return self.data_objs[idx]
    def __len__(self):
        return len(self.data_objs)

class PerQuestionDataset(Dataset):
    def __init__(self, args, mode, word2id, rela2id):
        super(PerQuestionDataset, self).__init__()
        self.data_objs = self._get_data(args, mode, word2id, rela2id)
    def _get_data(self, args, mode, word2id, rela2id):
        data_objs = []
        with open(f'{PATH[args.dataset]}/{mode}_data.txt', 'r') as f:
            for i, line in enumerate(f):
                print(f'\rreading line {i}', end='')
                data = json.loads(line)
                data = self._numericalize(data, word2id, rela2id)
                data_objs.append(data)
        return data_objs
    def _numericalize(self, data, word2id, rela2id):
        index, ques, step_list = data[0], data[1], data[2]
        ques = self._numericalize_str(ques, word2id, ' ')
        new_step_list = []
        for step in step_list:
            new_step = []
            for t in step:
                new_step.append((self._numericalize_str(t[0], rela2id, ['.']), 
                        self._numericalize_str(t[0], word2id, ['.', '_']), 
                        t[2]))
            new_step_list.append(new_step)
        return index, ques, new_step_list
    def _numericalize_str(self, string, map2id, dilemeter):
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        return tokens
    def __len__(self):
        return len(self.data_objs)
    def __getitem__(self, index):
        return self.data_objs[index]


if __name__ == '__main__':
    with open('../data/WQ/main_exp/rela2id.json', 'r') as f:
        rela2id =json.load(f)
    word2id_path = '../data/glove.300d.word2id.json' 
    with open(word2id_path, 'r') as f:
        word2id = json.load(f)
    class ARGS():
        def __init__(self):
            self.dataset = 'wq'
            self.mode = 'train'
        pass
    args = ARGS()
    dataset = PerQuestionDataset(args, 'train', word2id, rela2id)
    print()
    print(0, dataset[100][0])
    print(1, dataset[100][1])
    print(2, dataset[100][2])
    print(len(dataset[100]))
