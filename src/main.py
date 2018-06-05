import argparse
from UHop import UHop
import utility
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', type=str, default=None) # HR-BiLSTM, ABWIM, MVM

train_parser = parser.add_mutually_exclusive_group(required=True)   # train + test | test only
train_parser.add_argument('--train', action='store_true')
train_parser.add_argument('--test', action='store_true')

parser.add_argument('--emb_size', action='store', type=int)
parser.add_argument('--path', action='store', type=str, default=None) # for test mode, specify model path
parser.add_argument('--epoch_num', action='store', type=int, default=1000)
parser.add_argument('--hidden_size', action='store', type=int, default=256)
parser.add_argument('--neg_sample', action='store', type=int, default=2048)
parser.add_argument('--dropout_rate', action='store', type=float, default=0.0)
parser.add_argument('--learning_rate', action='store', type=float, default=0.0001)
parser.add_argument('--optimizer', action='store', type=str, default='adam')
parser.add_argument('--l2_norm', action='store', type=float, default=0.0)
parser.add_argument('--earlystop_tolerance', action='store', type=int, default=20)
parser.add_argument('--margin', action='store', type=float, default=0.5)
parser.add_argument('--stop_when_err', action='store', type=bool, default=False)
parser.add_argument('--train_step_1_only', action='store', type=bool, default=False)
parser.add_argument('--train_rela_choose_only', action='store', type=bool, default=False)
parser.add_argument('--show_result', action='store', type=bool, default=False)
parser.add_argument('--train_embedding', action='store', type=bool, default=False)
parser.add_argument('--dataset', action='store', type=str) #sq, wq, wq_train1_test2

args = parser.parse_args()
print(f'args: {args}')

import_model_str = 'from model.{} import Model as Model'.format(args.model)
exec(import_model_str)
args.Model = Model
if args.train == True:
    args.path = utility.find_save_dir(args.model) if args.path == None else args.path
if args.dataset == 'wq' or args.dataset == 'WQ':
    with open('../data/WQ/main_exp/rela2id.json', 'r') as f:
        rela2id =json.load(f)
elif args.dataset == 'sq' or args.dataset == 'SQ':
    with open('../data/SQ/rela2id.json', 'r') as f:
        rela2id =json.load(f)
elif args.dataset.lower() == 'wq_train1test2':
    with open('../data/WQ/train1test2_exp/rela2id.json', 'r') as f:
        rela2id =json.load(f)
else:
    raise ValueError('Unknown dataset.')

word2id_path = '../data/glove.300d.word2id.json' if args.emb_size == 300 else '../data/glove.50d.word2id.json' 
word_emb_path = '../data/glove.300d.word_emb.npy' if args.emb_size == 300 else '../data/glove.50d.word_emb.npy'
with open(word2id_path, 'r') as f:
    word2id = json.load(f)
word_emb = np.load(word_emb_path)
args.word_embedding = word_emb
args.rela_vocab_size = len(rela2id)

# Should introduce UHop here!
uhop = UHop(args, word2id, rela2id)
model = Model(args).cuda()
if args.train == True:
    uhop.train(model)
    model, loss, acc = uhop.eval(model=None, mode='test', dataset=None)
    utility.save_model_with_result(model, loss, acc, args.path)
if args.test == True:
    uhop.eval(model=None, mode='test', dataset=None)

