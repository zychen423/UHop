import argparse
from datetime import datetime
import os
import json 
import subprocess
import pickle
import sys
import requests
import MySQLdb 
from functools import lru_cache
import random

MAX_MID_NUM = 4096

db = MySQLdb.connect("localhost","zychen","myphone1459","KBQA", charset='utf8mb4')
cursor = db.cursor()

def load_relation_list(mode):
    have_rela_list = []
    if mode == 'sq':
        with open('sq_relations.txt', 'r') as f:
            for line in f:
                rela = line.strip().replace('/', '', 1).replace('/', '.')
                have_rela_list.append(rela)
    if mode == 'wq':
        with open('wq_relations.txt', 'r') as f:
            for line in f:
                have_relas = line.strip().split('..')
                have_rela_list += have_relas
    return have_rela_list

def filter_relation(rela):
    if rela in have_rela_list:
        return True
    else:
        return False

@lru_cache(maxsize=128)
def query_kb(mid):
    cursor.execute(f"SELECT * FROM freebase where head = '{mid}'")
    relas = []
    mids = []
    for head, rela, tail in cursor:
        relas.append(rela)
        mids.append(tail)
    #print(relas)
    return relas, mids

def get_next_topic_mid_list(mid, gold_rela):
    relas, mids = query_kb(mid)
    tails = set()
    for rela, mid in zip(relas, mids):
        if rela == gold_rela:
            tails.add(mid)
    return list(tails)

def preprocess_SQ(data_path, data_type=''):
    total_line = 0
    with open(data_path, 'r') as f:
        for o_id, line in enumerate(f):
            total_line += 1
    with open(data_path, 'r') as f:
        for o_id, line in enumerate(f):
            print(f'reading data {o_id}/{total_line}')
            data_list_by_question = []
            # remain a version question is not substitute by <e>
            raw_topic_mid, raw_gold_rela, raw_ans_mid, question = line.strip().split('\t')
            #print(question)
            topic_entity_mid = '.'.join(raw_topic_mid.split('/')[-2:])
            
            topic_entity_name = None
            topic_entity_mention = None
            rela_list = [raw_gold_rela.replace('www.freebase.com/', '').replace('/', '.')]
            #print('rela list is', rela_list)
            if topic_entity_name != None:
                question = question.replace(topic_entity_mention, 'TOPIC_ENTITY')
            q_obj = (o_id, question, [])
            if 'biggie' in question:
                input('theres biggie')
            #print(q_obj); exit()
            seen_tuple = []

            # the process likes recursion, and for the 2 and later
            # recursion, now_mid may be multiple, since the extracted
            # gold relation may lead to multiple entity node.
            now_mid_list = [topic_entity_mid]
            if rela_list == None or rela_list == 'null':
                input('there is no gold rela at', o_id)
                continue
            for i, gold_rela_at_len in enumerate(rela_list):
                print('at relation len {}'.format(i))
                # gold_rela_at_len is the len^th gold relation
                candid_list = []
                for j, mid in enumerate(now_mid_list):
                    print(f'\r{j}/{len(now_mid_list)}', end='')
                    candid_relas = query_relas(mid)
                    #print('find {} candidates'.format(len(candid_relas)))
                    seen_rela = []
                    label = 0
                    gold_count = 0
                    for candid_rela in candid_relas:
                        if candid_rela in seen_rela:
                            continue
                        seen_rela.append(candid_rela)
                        if candid_rela == gold_rela_at_len:
                            label = 1
                            gold_count += 1
                        else:
                            label = 0
                        t = (candid_rela, rela_list[:i], label)
                        if str(t) in seen_tuple:
                            continue
                        elif filter_relation(candid_rela) == True:
                            candid_list.append(t)
                            seen_tuple.append(str(t))
                        else:
                            continue
                    if gold_count == 0:
                        candid_list.append((gold_rela_at_len, rela_list[:i], 1))

                q_obj[2].append(candid_list)
                next_mid_list = []
                for mid in now_mid_list:
                    print(mid, gold_rela_at_len)
                    next_mid_list += get_next_topic_mid_list(mid, gold_rela_at_len)
                now_mid_list = next_mid_list
                if len(now_mid_list) > max_mid_num:
                    random.shuffle(now_mid_list)
                    now_mid_list = now_mid_list[:max_mid_num]
            # this is the extracted for the added time.
            # that is, if the len(gold_rela) == 2, this is the len == 3
            # relation extraction. this is going to be compared with len ==
            # 2 relation extraction score.
            candid_list = []
            #print('at final step')
            for j, mid in enumerate(now_mid_list):
                print(f'\r{j}/{len(now_mid_list)}', end='')
                candid_relas = query_relas(mid)
                seen_rela = []
                label = 0
                for candid_rela in candid_relas:
                    if candid_rela in seen_rela:
                        continue
                    seen_rela.append(candid_rela)
                    t = (candid_rela, rela_list, label)
                    if filter_relation(candid_rela) == True:
                        candid_list.append(t)
                        seen_tuple.append(t)
                    elif t in seen_tuple:
                        continue
                    else:
                        continue

            q_obj[2].append(candid_list)
            data_list_by_question.append(q_obj)
            with open('./tmp/{}.json', 'w') as f:
                json.dump(data_list_by_question, f)
                data_list_by_question = []
        #print(ad_hoc_list)

def preprocess_WQ(data_path):
    with open(data_path, 'r') as f:
        objs = json.load(f)['Questions']
        obj_len = len(objs)
        data_list_by_question = []
        for o_id, obj in enumerate(objs):
            write_flag = 1
            #if o_id > 2:
            #    break
            #print(f'reading data {o_id}/{obj_len}')
            question = obj['ProcessedQuestion']
            #print(question)
            correct_relas = [x['InferentialChain'] for x in obj['Parses']]
            if len(obj['Parses']) > 1:
                print('multiple parse')
            for p_i, parse in enumerate(obj['Parses']):
                topic_entity_mid = parse['TopicEntityMid']
                topic_entity_name = parse['TopicEntityName']
                topic_entity_mention = parse['PotentialTopicEntityMention']
                rela_list = parse['InferentialChain']
                #print('rela list is', rela_list)
                if topic_entity_name != None:
                    question = question.replace(topic_entity_mention, 'TOPIC_ENTITY')
                q_obj = (o_id, question, [])
                seen_tuple = set()

                # the process likes recursion, and for the 2 and later
                # recursion, now_mid may be multiple, since the extracted
                # gold relation may lead to multiple entity node.
                now_mid_list = [topic_entity_mid]
                if rela_list == None:
                    #input('there is no gold rela at', o_id)
                    print('there is no gold rela')
                    write_flag = 0
                    break
                for i, gold_rela_at_len in enumerate(rela_list):
                    #print('at relation len {}'.format(i))
                    # gold_rela_at_len is the len^th gold relation
                    candid_list = []
                    seen_rela = set()
                    gold_count = 0
                    for j, mid in enumerate(now_mid_list):
                        print(f'\r{datetime.now()} reading data {o_id}/{obj_len} parse:{p_i} {i}/{len(rela_list)} {j}/{len(now_mid_list)}             ', end='')
                        candid_relas, tails = query_kb(mid)
                        #print('find {} candidates'.format(len(candid_relas)))
                        label = 0
                        for candid_rela in candid_relas:
                            if candid_rela in seen_rela:
                                continue
                            seen_rela.add(candid_rela)
                            if candid_rela == gold_rela_at_len:
                                label = 1
                                gold_count += 1
                            else:
                                label = 0
                                if candid_rela in correct_relas:
                                    continue
                            t = (candid_rela, rela_list[:i], label)
                            if str(t) in seen_tuple:
                                continue
                            elif filter_relation(candid_rela) == True:
                                candid_list.append(t)
                                seen_tuple.add(str(t))
                            else:
                                continue
                    if gold_count == 0:
                        print('add gold ad-hoc')
                        candid_list.append((gold_rela_at_len, rela_list[:i], 1))

                    q_obj[2].append(candid_list)
                    next_mid_list = []
                    for mid in now_mid_list:
                        #print(mid, gold_rela_at_len)
                        next_mid_list += get_next_topic_mid_list(mid, gold_rela_at_len)
                    now_mid_list = next_mid_list
                # this is the extracted for the added time.
                # that is, if the len(gold_rela) == 2, this is the len == 3
                # relation extraction. this is going to be compared with len ==
                # 2 relation extraction score.
                candid_list = []
                for j, mid in enumerate(now_mid_list):
                    print(f'\r{datetime.now()} reading data {o_id}/{obj_len} parse:{p_i} last/{len(rela_list)} {j}/{len(now_mid_list)}                ', end='')
                    candid_relas, tails = query_kb(mid)
                    seen_rela = set()
                    label = 0
                    for candid_rela in candid_relas:
                        if candid_rela in seen_rela:
                            continue
                        seen_rela.add(candid_rela)
                        t = (candid_rela, rela_list, label)
                        if str(t) in seen_tuple:
                            continue
                        elif filter_relation(candid_rela) == True:
                            candid_list.append(t)
                            seen_tuple.add(str(t))
                        else:
                            continue

                q_obj[2].append(candid_list)
                data_list_by_question.append(q_obj)
            if write_flag == 1:
                with open(f'./tmp/{o_id}.json', 'w') as f:
                    json.dump(data_list_by_question, f)
                    data_list_by_question = []
            print()

def combine_data(path, mode, data_type):
    txts = os.listdir(path)
    datas = []
    if mode == 'wq':
        output_path = f'./data/WQ/{data_type}_data.txt'
    if mode == 'sq':
        output_path = f'./data/SQ/{data_type}_data.txt'
    with open(output_path, 'w') as f_out:
        for i, txt_file in enumerate(txts):
            print(f'combining {i}/{len(txts)}')
            with open(os.path.join(path, txt_file), 'r') as f:
                f_out.write(f.readline()+'\n')

def remove_tmp(path):
    files = os.listdir(path)
    for i, f in enumerate(files):
        print(f'removing {i}/{len(files)}')
        os.remove(os.path.join(path, f))
    os.rmdir(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', type=str, default='wq')   # wq | wq_train1test2 | sq
    parser.add_argument('--type', action='store', type=str, default='train')   # train | valid | test
    args = parser.parse_args()
    if args.mode == 'wq' and args.type == 'valid':
        raise ValueError('No validation set in WQ.')

    if not os.path.exists('tmp'):
        os.mkdir('./tmp')
    else:
        raise ValueError('./tmp dir exist, please remove it first to prevent error.')

    if args.mode == 'wq' or args.mode == 'wq_train1test2':
        have_rela_list = load_relation_list('wq')
        preprocess_WQ(f'/home/zychen/dataset/KBQA/WebQuestion/WebQSP/data/WebQSP.{args.type}.json')
    elif args.mode == 'sq':
        have_rela_list = load_relation_list('sq')
    else:
        raise ValueError("Wrong mode argument, shoule be one of wq, wq_train1test2, or sq.")
    combine_data('./tmp', args.mode, args.type)
    remove_tmp('./tmp')

