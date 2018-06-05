'''
1. find all relations in KB
2. for each relation, obtain 500 tail entity
3. if 95% entities with same type, record relation as that type
4. else, record relation as no_type

'''
from collections import defaultdict
import pickle
import timeit
import requests

def check_if_type(types_list):
    type_count = defaultdict(int)
    for types in types_list:
        for type in types:
            type_count[type] += 1
    for type, count in type_count.items():
        if count >= 500*0.95:
            return type
    return 'no_type'

def query_relas(mid):
    r = requests.get(f'http://140.109.19.67:7705/kb_query?mid={mid}')
    text = r.text.replace('<pre>', '').replace('</pre>', '')
    #print(text)
    relas = list(set([x.split('\t')[0] for x in text.split('\n') ]))
    #print(relas)
    return relas

def get_next_topic_mid_list(mid, rela):
    if not mid.startswith('m.') and not mid.startswith('g.'):
        print(f'mid {mid} is not a valid MID, return []')
        return []
    r = requests.get(f'http://140.109.19.67:7705/kb_query?mid={mid}')
    print(mid, rela)
    text = r.text.replace('<pre>', '').replace('</pre>', '')
    #print(text.split('\n'))
    mids = list(set([x.split('\t')[1] for x in text.split('\n') if x.split('\t')[0] == rela ]))
    print(mids)
    return mids

stored_rela_dict = defaultdict(list)
'''
s_time = timeit.default_timer()
for i in range(0, 805):
    n_time = timeit.default_timer()
    print(f'\rloading pickle {i}/805, time {int(n_time-s_time)}s', end='')
    pkl_path = f'/home/zychen/project/KB_engine/data/{i}.pkl'
    with open(pkl_path, 'rb') as f:
        mid_dict = pickle.load(f)
    for mid in mid_dict:
        for rela, tail_mid in mid_dict[mid]:
            if len(stored_rela_dict[rela]) < 500:
                stored_rela_dict[rela].append(tail_mid)
with open('stored_rela_dict.pkl', 'wb') as f:
    pickle.dump(stored_rela_dict, f)
exit()
'''
with open('stored_rela_dict.pkl', 'rb') as f:
    stored_rela_dict = pickle.load(f)
print(f'len of stored_rela_dict is {len(stored_rela_dict)}')
rela_to_type_dict = {}
s_time = timeit.default_timer()
for c, (rela, tail_mids) in enumerate(stored_rela_dict.items()):
    n_time = timeit.default_timer()
    print(f'\rloading pickle {c}/{len(stored_rela_dict)}, time {int(n_time-s_time)}s', end='')
    types_list = [[]] * 500
    for i, tail_mid in enumerate(tail_mids):
        print(f' processing {i}/{len(tail_mids)}')
        print(f'get type of mid {tail_mid}')
        types = get_next_topic_mid_list(tail_mid, 'type.object.type')
        types_list[i] = types
    tail_entity_type = check_if_type(types_list)
    print(f'rela {rela} is type {tail_entity_type}')
    rela_to_type_dict[str(rela)] = tail_entity_type

with open('./tail_entity_types.txt', 'w') as f:
    for i, (rela, tail_type) in enumerate(rela_to_type_dict.items()):
        print(f'\routput {i}/{len(rela_to_type_dict)}', end='')
        f.write(f'{rela}\t{tail_type}\n')



