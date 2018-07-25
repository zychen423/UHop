import json
import os, errno
import random

class PQ_Parser():
    def __init__(self, file_name, kb_name, hop_num):
        # read and split file 
        with open(file_name, 'r') as f:
            raw_data = [line.split('\t') for line in f.read().splitlines()]
        with open(kb_name, 'r') as f:
            kb = [line.split('\t') for line in f.read().splitlines()]
        self.kb = kb
        # extract candidates
        self.data = []
        self.rela = []
        for idx, data in enumerate(raw_data):
            path = data[2].split('#<end>')[0].split('#')
            steplist = []
            for i in range(0,hop_num*2,2):
                relas = list(set([k[1] for k in kb if k[0] == path[i]]))
                steplist.append([[self._format(rela),[],int(len(path)>i+1 and rela==path[i+1])] for rela in relas])
#                for k in kb:
#                    if k[0] == path[i]:
#                        step = [self._format(k[1]), [], int(len(path)>i+1 and k[1]==path[i+1])]
#                        if not(len(path)>i+1 and k[2]!=path[i+2] and k[1]==path[i+1]):
#                            steplist[i//2].append(step)
            self.rela.append(sum([self._format(path[i]).split('.') for i in range(1,len(path),2)],[]))
            self.data.append([idx, data[0].replace(path[0],'TOPIC_ENTITY'), steplist])
        self.hop = hop_num

    def merge(self, parser2):
        self.data += parser2.data
        random.shuffle(self.data, random.random)
        print(len(self.data))

    def dump(self, dir_name):
        self._check_dir(dir_name)
        # split data into 8:1:1 and dump
        a, b = int(len(self.data)*0.8), int(len(self.data)*0.9)
        print(a, b-a, len(self.data)-b)
        self._write(self.data[:a], dir_name+'/train_data.txt') 
        self._write(self.data[a:b], dir_name+'/valid_data.txt') 
        self._write(self.data[b:], dir_name+'/test_data.txt')
        self._rela2id(self.rela[:a], dir_name+'/rela2id.json')

    def dump2(self, dir_name, mode):
        self._check_dir(dir_name)
        if mode=='test':
            print(len(self.data))
            self._write(self.data, dir_name+'/test_data.txt')
        else:
            a = int(len(self.data)*0.9)
            print(a, len(self.data)-a)
            self._write(self.data[:a], dir_name+'/train_data.txt')
            self._write(self.data[a:], dir_name+'/valid_data.txt')
            self._rela2id(self.rela[:a], dir_name+'/rela2id.json')
   
    def _format(self, rela):
        if rela[:2]=='__':
            rela = rela[2:]
        return rela.replace('__','.')

    def _write(self, data, output_name):
        with open(output_name, 'w') as f:
            for line in data:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

    def _rela2id(self, relations, rela_name):
        rela = ["PADDING", "<unk>"] + list(set([i for j in relations for i in j]))
        rela = dict([(r,idx) for idx, r in enumerate(rela)])
        with open(rela_name, 'w') as f:
            json.dump(rela, f, ensure_ascii=False)
                
    def _check_dir(self, dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def all_rela(self):
        return list(set([i for j in self.rela for i in j]))

    def count_rela(self):
        return list(set([k[1] for k in self.kb]))

    def count_candidate(self):
        counts = [[len(d[2][i]) for d in self.data] for i in range(self.hop)]
        return [[step.count(i) for i in range(max(step))] for step in counts]

def main():
    print('dumping PQ2...')
    pq_2h = PQ_Parser('PQ-2H.txt', '2H-kb.txt', 3)
    print(pq_2h.count_candidate())
    #pq_2h.dump('PQ2')
    #pq_2h.dump2('PQ', 'train')
    print('dumping PQ3...')
    pq_3h = PQ_Parser('PQ-3H.txt', '3H-kb.txt', 4)
    print(pq_3h.count_candidate())
    #pq_3h.dump('PQ3')
    #pq_3h.dump2('PQ', 'test')
    print(len(list(set(pq_2h.all_rela()+pq_3h.all_rela()))))
    print(len(list(set(pq_2h.count_rela()+pq_3h.count_rela()))))
    #pq_2h.merge(pq_3h)
    #pq_2h.dump('PQ1')
    print('dumping PQL2...')
    pql_2h = PQ_Parser('PQL-2H.txt', 'PQL2-KB.txt', 3)
    print(pql_2h.count_candidate())
    #pql_2h.dump('PQL2')
    #pql_2h.dump2('PQL', 'train')
    print('dumping PQL3...')
    pql_3hm = PQ_Parser('PQL-3H_more.txt', 'PQL3-KB.txt', 4)
    print(pql_3hm.count_candidate())
    #pql_3hm.dump('PQL3m')
    #pql_3hm.dump2('PQL', 'test')
    print(len(list(set(pql_2h.all_rela()+pql_3hm.all_rela()))))
    print(len(list(set(pql_2h.count_rela()+pql_3hm.count_rela()))))
    #pql_2h.merge(pql_3hm)
    #pql_2h.dump('PQL1')

if __name__ == "__main__":
    main()
