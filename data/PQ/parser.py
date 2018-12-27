import json
import os, errno
import itertools

from pq_reader import PQ_Reader
from wq_reader import WQ_Reader
from utility import *

WQ_DATA = ['/home/zychen/project/UHop/script/preprocess/data/WQ/train_data.txt', 
                '/home/zychen/project/UHop/script/preprocess/data/WQ/test_data.txt']
WQ_BASELINE = ['../WQ2/train_data1.txt','../WQ2/test_data1.txt']
WQ_RELA2ID = '../WQ2/rela2id.json'
WQ_CONCAT_RELA2ID = '../WQ2/concat_rela2id.json'

def main():
    # exp2 dataset - PQ
 #   pq_2h = PQ_Reader('PQ-2H.txt', '2H-kb.txt', 3)
 #   pq_3h = PQ_Reader('PQ-3H.txt', '3H-kb.txt', 4)
#    pq_2h.std_dump('PQ2')
#    pq_3h.std_dump('PQ3')
#    pq_2h.std_dump('baseline/PQ2', baseline=True)
#    pq_3h.std_dump('baseline/PQ3', baseline=True)
#    pq_2h.merge_dump(pq_3h, 'PQ1')
#    pq_2h.merge_dump(pq_3h, 'baseline/PQ1', baseline=True)

    # exp2 dataset - PQL
    pql_2h = PQ_Reader('PQL-2H.txt', 'PQL2-KB.txt', 3)
    pql_3h = PQ_Reader('PQL-3H_more.txt', 'PQL3-KB.txt', 4)
    #pql_2h._rela2id(pql_2h.concat_rela, 'pql2gg.json')
    #pql_3h._rela2id(pql_3h.concat_rela, 'pql3gg.json')
    #return
#    pql_2h.std_dump('PQL2')
#    pql_3h.std_dump('PQL3')
#    pql_2h.std_dump('baseline/PQL2', baseline=True)
#    pql_3h.std_dump('baseline/PQL3', baseline=True)
#    pql_2h.merge_dump(pql_3h, 'PQL1')
#    pql_2h.merge_dump(pql_3h, 'baseline/PQL1', baseline=True)

    # exp3 dataset
    wq = WQ_Reader(WQ_DATA, WQ_BASELINE, WQ_RELA2ID, WQ_CONCAT_RELA2ID)
    wq.WPQ_maker(pql_2h, pql_3h, 'WPQLt2/')
#    return
    # exp4 dataset
#    exp4_rela, exp4_concat_rela = pql_2h.rela+pql_3h.rela, pql_2h.concat_rela+pql_3h.concat_rela
#    pql_2h._rela2id(exp4_rela, 'exp4/rela2id.json')
#    exp4_concat_rela2id = pql_2h._rela2id(exp4_concat_rela, 'exp4/concat_rela2id.json')
#    pql_2h.dump_exp4('exp4', exp4_concat_rela2id, train_mode=False)
#    pql_3h.dump_exp4('exp4', exp4_concat_rela2id, train_mode=True)
#    pql_2h.dump_exp4('exp4/baseline', exp4_concat_rela2id, train_mode=False, baseline=True)
#    pql_3h.dump_exp4('exp4/baseline', exp4_concat_rela2id, train_mode=True, baseline=True)

if __name__ == "__main__":
    main()
