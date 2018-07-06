import json
import sys
import numpy as np
from collections import defaultdict
import operator

def make_word2id_from_w2v(dim, word2vec_path, output_word2id_path, output_emb_path):
    word_id_map = {'PADDING':0, '<unk>':1, 'TOPIC_ENTITY':2}
    embedding = [np.zeros(dim), np.ones(dim), np.random.uniform(size=dim)]
    with open(word2vec_path, 'r') as f, open(output_word2id_path, 'w') as f_out:
        for i, line in enumerate(f):
            print(f'\rmaking word2id from w2v {i}', end='')
            word, vector = ' '.join(line.strip().split(' ')[:-dim]), line.strip().split(' ')[-dim:]
            word_id_map[word] = len(embedding)
            vector = np.array([float(v) for v in vector])
            embedding.append(vector)
        print()
        embedding = np.array(embedding)
        json.dump(word_id_map, f_out)
        np.save(output_emb_path, embedding)

def make_rela2id(data_path_list, output_path):
    rela2id = {'PADDING':0, '<unk>':1}
    rela_nums = defaultdict(int)
    for data_path in data_path_list:
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                print(f'\rmaking rela2id {i}', end='')
                #print(line)
                data = json.loads(line.strip())
                for step_list in data[2]:
                    for relas, _, _ in step_list:
                        for rela in relas.split('.'):
                            rela_nums[rela] += 1
    sorted_rela = sorted(rela_nums.items(), key=operator.itemgetter(1))
    print(sorted_rela)
    sorted_rela = [x[0] for x in sorted_rela][int(len(sorted_rela)*0.2):]
    for rela in sorted_rela:
        rela2id[rela] = len(rela2id)
    print(rela2id)
    with open(output_path, 'w') as f_out:
        json.dump(rela2id, f_out)

def make_baseline_rela2id(data_path, output_path):
    rela2id = {}
    rela2id['PADDING'] = len(rela2id)
    with open(data_path, 'r') as f, open(output_path, 'w') as f_out:
        for i, line in enumerate(f):
            if line.startswith('/'):
                line = line.replace('/', '', 1)
                line = line.replace('/', '.')
            rela2id[line.strip().replace('..', '.')] = i+1
        rela2id['<unk>'] = len(rela2id)
        json.dump(rela2id, f_out)

if __name__ == '__main__':
    #make_word2id_from_w2v(300, '/corpus/glove/pretrained_vector/english/glove.6B.300d.txt', 
    #        '../data/glove.300d.word2id.json', '../data/glove.300d.word_emb.npy')
    #make_word2id_from_w2v(50, '/corpus/glove/pretrained_vector/english/glove.6B.50d.txt', 
    #        '../data/glove.50d.word2id.json', '../data/glove.50d.word_emb.npy')
    make_rela2id(['../data/SQ/train_data.txt', '../data/SQ/test_data.txt']
            , '../data/SQ/rela2id.json')
    make_rela2id(['../data/WQ/main_exp/train_data.txt', '../data/WQ/main_exp/test_data.txt']
            , '../data/WQ/main_exp/rela2id.json')
    make_rela2id(['../data/WQ/train1test2_exp/train_data.txt', '../data/WQ/train1test2_exp/test_data.txt']
            , '../data/WQ/train1test2_exp/rela2id.json')
    #make_baseline_rela2id('/home/zychen/project/UHop/data/baseline/KBQA_RE_data/webqsp_relations/relations.txt', 
    #        '../data/baseline/KBQA_RE_data/webqsp_relations/rela2id.json')
    #make_baseline_rela2id('/home/zychen/project/UHop/data/baseline/KBQA_RE_data/sq_relations/relation.2M.list', 
    #        '../data/baseline/KBQA_RE_data/sq_relations/rela2id.json')

