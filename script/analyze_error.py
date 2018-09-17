import sys
import json
import numpy as np
import pandas as pd
from pandas import DataFrame

KEY1, KEY2 = sys.argv[1], sys.argv[2]

#U_PATH = [('pq2hr', '127'), ('pq3hr', '82'), ('pq1hr', '79'), 
#            ('pql2hr', '144'), ('pql3hr', '95'), ('pql1hr', '159'),
#            ('pq2ab', '181'), ('pq3ab', '162'), ('pq1ab', '89'),
#            ('pql2ab', '92'), ('pql3ab', '204'), ('pql1ab', '160'),
#            ('wqhr', '289'), ('wqab', '119'), ('exp4hr', '438'), ('exp4ab', '346')]

U_PATH=[('wqab2', '262'), ('wqab', '999'), ('pq2hr', '127'), ('pq2ab', '127'), ('pq1hr', '79'),
            ('pq1ab', '89'), ('pq3hr', '82'), ('pq3ab', '162'), ('pql2hr', '144'),
            ('pql2ab', '92'), ('pql1hr', '159'), ('pql1ab', '160'), ('pql3hr', '95'),
            ('pql3ab', '204'), ('exp4hr', '438'), ('exp4ab', '346')]

U_PATH=[('wqab', '350')]

DATA_PATH = '../data/PQ/WPQL/1'

LABEL_DICT1 = {'<WR>':'1RC', '<CR> <C>':'1TD', '<CR> <T>':'OK'}
LABEL_DICT2 = {'<WR>':'1RC', '<CR> <T>':'1TD', '<CR> <C> <WR>':'2RC', '<CR> <C> <CR> <C>':'2TD', '<CR> <C> <CR> <T>':'OK', '<CR> <C>':'OK'}
LABEL_DICT3 = {'<WR>':'1RC', '<CR> <T>':'1TD', '<CR> <C> <WR>':'2RC', '<CR> <C> <CR> <T>':'2TD', '<CR> <C> <CR> <C> <WR>':'3RC', '<CR> <C> <CR> <C> <CR> <C>':'3TD', '<CR> <C> <CR> <C> <CR> <T>':'OK'}
LABEL_LIST = ['1RC', '1TD', '2RC', '2TD', '3RC', '3TD', 'OK']

def read_lines(filepath):
    with open(filepath, 'r') as f:
        return f.read().splitlines()

def label_UHop(line, hop):
    label_dict = LABEL_DICT1 if hop==1 else (LABEL_DICT2 if hop==2 else LABEL_DICT3)
    return label_dict[line.replace('\t', ' ')]
    labels = []
    for label in line.split('\t'):
        if label == '':
            labels.append('<CR>')
        elif label == '<C>':
            labels.append('<C>')
        elif label == '<T>':
            labels.append('<T>')
            break
        else:
            labels.append('<WR>')
            break
    if len(labels)>1 and (labels[-2]=='<Terminate>' or labels[-2]=='<T>'):
        labels = labels[:-1]
    if ' '.join(labels) not in label_dict:
        print(line)
    return label_dict[' '.join(labels)]

def analysis_uhop(info):
    path, dataset = info
    uhop = read_lines(f'{path}/prediction.txt')
    #print(dataset)
    if 'pq2' in dataset or 'pql2' in dataset or 'exp4' in dataset:
        labels = [[label_UHop(line, 2) for line in uhop]]
    elif 'pq3' in dataset or 'pql3' in dataset:
        labels = [[label_UHop(line, 3) for line in uhop]]
    elif 'pq1' in dataset:
        label2 = [label_UHop(line, 2) for line in uhop[:191]]
        label3 = [label_UHop(line, 3) for line in uhop[191:]]
        labels = [label2, label3]
    elif 'pql1' in dataset:
        label2 = [label_UHop(line, 2) for line in uhop[:160]]
        label3 = [label_UHop(line, 3) for line in uhop[160:]]
        labels = [label2, label3]
    else:
        label1, label2 = [], []
        with open('/home/lance5/UHop/data/WQ/main_exp/test_data.txt', 'r') as f:
            gold=f.read().splitlines()
            print(len(gold), len(uhop))
            for g, u in zip(gold, uhop):
                hop = len(json.loads(g)[2])
                if hop==2:
                    label1.append(label_UHop(u, 1))
                else:
                    label2.append(label_UHop(u, 2))
            labels = [label1, label2]

    return labels

def full(path):
    model = 'HR_BiLSTM' if 'hr' in path[0] else 'ABWIM'
    return f'/home/lance5/UHop/weighted_wq/{model}_{path[1]}', path[0]

def acc(label):
    return label.count('OK')/len(label)

def main():
    labels = {}
    for path in U_PATH:
        if path[1] != '':
            labels[path[0]] = analysis_uhop(full(path))
    label_list = list(labels.items())
#    label_list = sorted(list(labels.items()), key=lambda x:x[0])

    model_index, model_counts = [], []
    for model, label in label_list:
        model_index.append(model)
        model_counts.append([100*label[0].count(l)/len(label[0]) for l in LABEL_LIST])
        if len(label) > 1:
            model_index[-1]+='2'
            model_index.append(model+'3')
            model_counts.append([100*label[1].count(l)/len(label[1]) for l in LABEL_LIST])

    df = DataFrame(data=model_counts, index=model_index, columns=LABEL_LIST).round(2)
    df.to_csv('main.csv')

    with open('/home/lance5/UHop/data/PQ/PQL2/test_data.txt') as f:
        gold = f.read().splitlines()
        for l, g in zip(labels['wqab'][0], gold):
            if l == '1RC':
                print(g.split('"')[1])

    print(f'{df}\n\ncomparison between {KEY1} and {KEY2}:\n')
    compare(labels[KEY1][0], labels[KEY2][0])

#    print((labels['wqhr'][0]+labels['wqhr'][1]).count('OK')/len(labels['wqhr'][0]+labels['wqhr'][1]))
    print((labels['wqab'][0]+labels['wqab'][1]).count('OK')/len(labels['wqab'][0]+labels['wqab'][1]))

    return

def compare(labels1, labels2):
    label2id = {l:i for i,l in enumerate(LABEL_LIST)}
    counts = np.zeros((7, 7))

    tuples = [[], [], []]

    for l1, l2 in zip(labels1, labels2):
        counts[label2id[l1]][label2id[l2]] += 1
    df = DataFrame(data=counts, index=LABEL_LIST, columns=LABEL_LIST, dtype=int)
    print(df)

    return df

if __name__=='__main__':
    main()
