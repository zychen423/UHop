import argparse
import os
from UHop import UHop
from Baseline import Baseline
import utility
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', type=str, default=None) # HR-BiLSTM, ABWIM, MVM
parser.add_argument('--framework', action='store', type=str, default='UHop') # UHop, baseline

train_parser = parser.add_mutually_exclusive_group(required=True)   # train + test | test only
train_parser.add_argument('--train', action='store_true')
train_parser.add_argument('--test', action='store_true')

parser.add_argument('--emb_size', action='store', type=int)
parser.add_argument('--path', action='store', type=str, default=None) # for test mode, specify model path
parser.add_argument('--epoch_num', action='store', type=int, default=1000)
parser.add_argument('--hidden_size', action='store', type=int, default=256)
parser.add_argument('--num_filters', action='store', type=int, default=150)
parser.add_argument('--neg_sample', action='store', type=int, default=2048)
parser.add_argument('--dropout_rate', action='store', type=float, default=0.0)
parser.add_argument('--learning_rate', action='store', type=float, default=0.0001)
parser.add_argument('--optimizer', action='store', type=str, default='adam')
parser.add_argument('--l2_norm', action='store', type=float, default=0.0)
parser.add_argument('--earlystop_tolerance', action='store', type=int, default=10)
parser.add_argument('--margin', action='store', type=float, default=0.5)
parser.add_argument('--train_step_1_only', action='store', type=bool, default=False)
parser.add_argument('--train_rela_choose_only', action='store', type=bool, default=False)
parser.add_argument('--show_result', action='store', type=bool, default=False)
parser.add_argument('--train_embedding', action='store', type=bool, default=False)
parser.add_argument('--log_result', action='store', type=bool, default=False)
parser.add_argument('--dataset', action='store', type=str) #sq, wq, wq_train1_test2
parser.add_argument('--saved_dir', action='store', type=str, default='saved_model')
parser.add_argument('--hop_weight', action='store', type=float, default=1)
parser.add_argument('--task_weight', action='store', type=float, default=1)
parser.add_argument('--acc_weight', action='store', type=float, default=1)
parser.add_argument('--stop_when_err', action='store_true')
parser.add_argument('--step_every_step', action='store_true')
parser.add_argument('--change_ques', action='store_true')

args = parser.parse_args()
print(f'args: {args}')

import_model_str = 'from model.{} import Model as Model'.format(args.model)
exec(import_model_str)
if args.train == True:
    args.path = utility.find_save_dir(args.saved_dir, args.model) if args.path == None else args.path
    with open(os.path.join(args.path, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

#baseline_path, UHop_path = path_finder.path_finder()
#wpq_path = path_finder.WPQ_PATH()

args.Model = Model
if args.framework == 'baseline':
    if args.dataset.lower() == 'wq':
        with open('../data/baseline/KBQA_RE_data/webqsp_relations/rela2id.json', 'r') as f:
            rela2id = json.load(f)
        with open('../data/WQ/main_exp/rela2id.json', 'r') as f:
            rela_token2id =json.load(f)
    elif args.dataset.lower() == 'sq':
        with open('../data/baseline/KBQA_RE_data/sq_relations/rela2id.json', 'r') as f:
            rela2id = json.load(f)
        with open('../data/SQ/rela2id.json', 'r') as f:
            rela_token2id =json.load(f)
    elif args.dataset.lower() == 'exp4':
        with open('../data/PQ/exp4/rela2id.json', 'r') as f:
            rela_token2id = json.load(f)
        with open('../data/PQ/exp4/concat_rela2id.json', 'r') as f:
            rela2id = json.load(f)
#    elif 'wpq' in args.dataset.lower():
#        with open(wpq_path.concat_rela2id, 'r') as f:
#            rela2id = json.load(f)
#        with open('../data/PQ/exp3/baseline/rela2id.json', 'r') as f:
        #with open(wpq_path.rela2id, 'r') as f:
#            rela_token2id = json.load(f)
#    elif 'pq' in args.dataset.lower():
#        with open(baseline_path.concat_rela2id(args.dataset.lower()), 'r') as f:
#            rela2id = json.load(f)
#        with open(baseline_path.rela2id(args.dataset.lower()), 'r') as f:
#            rela_token2id =json.load(f)
    else:
        raise ValueError('Unknown dataset')
elif args.framework == 'UHop':
    if args.dataset == 'wq' or args.dataset == 'WQ':
        with open('../data/WQ/main_exp/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset == 'sq' or args.dataset == 'SQ':
        with open('../data/SQ/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'wq_train1test2':
        with open('../data/WQ/train1test2_exp/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'exp4':
        with open('../data/PQ/exp4/rela2id.json', 'r') as f:
            rela2id = json.load(f)
    elif 'wpq' in args.dataset.lower():
        with open('../data/PQ/exp3/UHop/rela2id.json', 'r') as f:
            rela2id = json.load(f)
    elif args.dataset.lower() == 'pq':
        with open('../data/PQ/PQ/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pq1':
        with open('../data/PQ/PQ1/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pq2':
        with open('../data/PQ/PQ2/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pq3':
        with open('../data/PQ/PQ3/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql':
        with open('../data/PQ/PQL/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql1':
        with open('../data/PQ/PQL1/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql2':
        with open('../data/PQ/PQL2/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    elif args.dataset.lower() == 'pql3':
        with open('../data/PQ/PQL3/rela2id.json', 'r') as f:
            rela2id =json.load(f)
    else:
        raise ValueError('Unknown dataset.')

#print(rela2id)
#print(rela2id['scientist'])
#exit()
word2id_path = '../data/glove.300d.word2id.json' if args.emb_size == 300 else '../data/glove.50d.word2id.json' 
word_emb_path = '../data/glove.300d.word_emb.npy' if args.emb_size == 300 else '../data/glove.50d.word_emb.npy'
with open(word2id_path, 'r') as f:
    word2id = json.load(f)
word_emb = np.load(word_emb_path)
args.word_embedding = word_emb
if args.framework == 'UHop': 
    args.rela_vocab_size = len(rela2id)
if args.framework == 'baseline':
    args.rela_vocab_size = len(rela_token2id)

# Should introduce UHop here!
if args.framework == 'UHop':
    uhop = UHop(args, word2id, rela2id, args.dataset.lower())
    model = Model(args).cuda()
    if args.train == True:
        uhop.train(model)
        model, loss, acc, rc, td, output, scores = uhop.eval(model=None, mode='test', dataset=None, output_result=True)
        #utility.save_model_with_result(model, loss, acc, rc, td, args.path)
        with open(f'{args.path}/prediction.txt', 'w') as f:
            f.write(output)
        with open(f'{args.path}/scores_{100*acc:.2f}.json', 'w') as f:
            json.dump(scores, f)
    if args.test == True:
        _, _, acc, _, _, outupt, scores = uhop.eval(model=None, mode='test', dataset=None, output_result=True)
        with open(f'{args.path}/prediction.txt', 'w') as f:
            f.write(outupt)
        with open(f'{args.path}/scores_{100*acc:.2f}.json', 'w') as f:
            json.dump(scores, f)
elif args.framework == 'baseline':
    baseline = Baseline(args, word2id, rela2id, rela_token2id)
    model = Model(args).cuda()
    if args.train == True:
        baseline.train(model)
        model, loss, acc = baseline.eval(model=None, mode='test', dataset=None)
        #utility.save_model_with_result(model, loss, acc, 0, 0, 0, args.path)
    if args.test == True:
        baseline.eval(model=None, mode='test', dataset=None, path=args.path)
    

