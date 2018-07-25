import torch
import torch.nn as nn
import numpy
import math
from sklearn.utils import shuffle
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.optim import lr_scheduler
from utility import save_model, load_model, log_result 
import random
from datetime import datetime
from baseline_data_utility import PerQuestionDataset, random_split, quick_collate

class Baseline():
    def __init__(self, args, word2id, rela2id, rela_token2id):
        self.loss_function = nn.MarginRankingLoss(margin=args.margin)
        self.args = args
        self.word2id = word2id
        self.rela2id = rela2id
        self.rela_token2id = rela_token2id
    def get_optimizer(self, model):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm, amsgrad=True)
        if self.args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(params = filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm)
        if self.args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(params = filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=self.args.learning_rate, weight_decay=self.args.l2_norm)
        return optimizer
    def _padding(self, lists, maxlen, type, padding):
        new_lists = []
        for list in lists:
            if type == 'prepend':
                new_list = [padding] * (maxlen - len(list)) + list
            elif type == 'append':
                new_list = list + [padding] * (maxlen - len(list))
            new_lists.append(new_list)
        return new_lists
    def train(self, model):
        '''
        train 1 batch / question
        '''
        dataset = PerQuestionDataset(self.args, 'train', self.word2id, self.rela2id, self.rela_token2id)
        if self.args.dataset.lower() == 'wq' or self.args.dataset.lower() == 'wq_train1test2':
            train_dataset, valid_dataset = random_split(dataset, 0.9, 0.1)
        else:
            train_dataset = dataset
            valid_dataset = PerQuestionDataset(self.args, 'valid', self.word2id, self.rela2id, self.rela_token2id)
        datas = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, 
                pin_memory=False, collate_fn=quick_collate)
        optimizer = self.get_optimizer(model)
        earlystop_counter, min_valid_metric = 0, 100
        for epoch in range(0, self.args.epoch_num):
            model = model.train().cuda()
            total_loss, total_acc = 0.0, 0.0
            loss_count, acc_count = 0, 0
            for trained_num, (index, ques, ans, candidates) in enumerate(datas):
                if len(candidates) == 0:
                    total_acc += 1; acc_count += 1; loss_count += 1
                    continue
                relas = [ans[0]] + [x[0] for x in candidates]
                maxlen = max([len(x) for x in relas])
                relas = self._padding(relas, maxlen, 'prepend', self.rela_token2id['PADDING'])
                rela_texts = [ans[1]] + [x[1] for x in candidates]
                maxlen = max([len(x) for x in rela_texts])
                rela_texts = self._padding(rela_texts, maxlen, 'prepend', self.word2id['PADDING'])
                # j'ai fait padding pour assurer la convolution marche
                if len(ques)<5:
                    ques = self._padding([ques], 5, 'prepend', self.word2id['PADDING'])[0]
                ques = torch.LongTensor([ques]*len(relas)).cuda()
                relas = torch.LongTensor(relas).cuda()
                rela_texts = torch.LongTensor(rela_texts).cuda()
                optimizer.zero_grad();model.zero_grad(); 
                scores = model(ques, rela_texts, relas)
                pos_scores = scores[0].repeat(len(scores)-1)
                neg_scores = scores[1:]
                ones = torch.ones(len(neg_scores)).cuda()
                loss = self.loss_function(pos_scores, neg_scores, ones)
                loss.backward(); optimizer.step()
                acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0
                total_loss += loss.data; loss_count += 1
                total_acc += acc; acc_count += 1
                print(f'\r{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch {epoch} {trained_num}/{len(datas)} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f}', end='')
            _, valid_loss, valid_acc = self.eval(model, 'valid', valid_dataset)
            if valid_loss < min_valid_metric:
                min_valid_metric = valid_loss
                earlystop_counter = 0
                save_model(model, self.args.path)
            else:
                earlystop_counter += 1
            if earlystop_counter > self.args.earlystop_tolerance:
                break       
        return model, total_loss / loss_count, total_acc / acc_count
            

    def eval(self, model, mode, dataset):
        '''
        For test and validation
        mode: valid | test
        '''
        if model == None:
            model = self.args.Model(self.args).cuda()
            model = load_model(model, self.args.path)
        model = model.eval().cuda()
        if dataset == None:
            dataset = PerQuestionDataset(self.args, mode, self.word2id, self.rela2id, self.rela_token2id)
        datas = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, 
                pin_memory=False, collate_fn=quick_collate)
        total_loss, total_acc = 0.0, 0.0
        loss_count, acc_count = 0, 0
        for num, (index, ques, ans, candidates) in enumerate(datas):
            if len(candidates) == 0:
                total_acc += 1; acc_count += 1; loss_count += 1
                continue
            relas = [ans[0]] + [x[0] for x in candidates]
            maxlen = max([len(x) for x in relas])
            relas = self._padding(relas, maxlen, 'prepend', self.rela2id['PADDING'])
            rela_texts = [ans[1]] + [x[1] for x in candidates]
            maxlen = max([len(x) for x in rela_texts])
            rela_texts = self._padding(rela_texts, maxlen, 'prepend', self.word2id['PADDING'])
            # padding in order to achieve size of filter
            if len(ques)<5:
                ques = self._padding([ques], 5, 'prepend', self.word2id['PADDING'])[0]
            ques = torch.LongTensor([ques]*len(relas)).cuda()
            relas = torch.LongTensor(relas).cuda()
            rela_texts = torch.LongTensor(rela_texts).cuda()
            scores = model(ques, rela_texts, relas)
            pos_scores = scores[0].repeat(len(scores)-1)
            neg_scores = scores[1:]
            ones = torch.ones(len(neg_scores)).cuda()
            loss = self.loss_function(pos_scores, neg_scores, ones)
            acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0
            if self.args.log_result == True and self.args.test == True:
                print(f'\rlogging {num}/{len(datas)}', end='')
                log_result(num, ques, relas, rela_texts, scores, acc, self.args.path, self.word2id, self.rela_token2id)
            total_loss += loss.data; loss_count += 1
            total_acc += acc; acc_count += 1
        print(f' Eval {num} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f}', end='')
        print('')
        return model, total_loss / loss_count, total_acc / acc_count
