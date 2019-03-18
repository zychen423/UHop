import torch
import torch.nn as nn
import numpy
import math
from data_utility import PerQuestionDataset, random_split, quick_collate
from torch.utils.data import DataLoader 
import torch.nn as nn
from torch.optim import lr_scheduler
from utility import save_model, load_model 
import random
from datetime import datetime
import json

total_rank = 0
rank_count = 0
total_all = 0

class UHop():
    def __init__(self, args, word2id, rela2id, score):
        self.loss_function = nn.MarginRankingLoss(margin=args.margin)
        self.args = args
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2rela = {v:k for k,v in rela2id.items()}
        self.score = score
    def train_batch(self, model):
        '''
        Making batching, so cannot get acc/question
        '''
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

    def _padding(self, lists, position, maxlen, type, padding):
        new_lists, new_position = [], []
        #for list, pos in zip(lists, position):
        for list in lists:
            if type == 'prepend':
                new_list = [padding] * (maxlen - len(list)) + list
        #        new_pos = [0] * (maxlen - len(list)) + pos
            elif type == 'append':
                new_list = list + [padding] * (maxlen - len(list))
        #        new_pos = pos + [0] * (maxlen - len(list))
            new_lists.append(new_list)
        #    new_position.append(new_pos)
        return new_lists, new_position

    def _padding_for_bert(self, seqs, maxlen, pad_type):
        padded_seqs, atten_mask = [], []
        for seq in seqs:
            if pad_type == 'prepend':
                padded_seq = [padding] * (maxlen - len(seq)) + seq
                mask = [0]*(maxlen-len(seq)) + [1]*len(seq)
            elif pad_type == 'append':
                padded_seq = seq + [padding] * (maxlen - len(seq))
                mask = [1]*len(seq) + [0]*(maxlen-len(seq))
            padded_seqs.append(padded_seq)
            atten_mask.append(mask)
        return padded_seqs, atten_mask

    def _pad_rela(self, pos_rela, neg_rela, pos_text, neg_text, pos_pos, neg_pos):
        maxlen = max([len(x) for x in pos_rela+neg_rela])
        pos_rela, pos_pos = self._padding(pos_rela, pos_pos, maxlen, 
                                                'append', self.rela2id['PADDING'])
        neg_rela, neg_pos = self._padding(neg_rela, neg_pos, maxlen, 
                                                'append', self.rela2id['PADDING'])
        maxlen = max([len(x) for x in pos_text+neg_text])
        pos_text, pos_pos = self._padding(pos_text, pos_pos, maxlen,
                                                'prepend', self.word2id['PADDING'])
        neg_text, neg_pos = self._padding(neg_text, neg_pos, maxlen,
                                                'prepend', self.word2id['PADDING'])
        relas = torch.LongTensor(pos_rela+neg_rela).cuda()
        rela_text = torch.LongTensor(pos_text+neg_text).cuda()
        rela_pos = torch.LongTensor(pos_pos+neg_pos).cuda()
        return relas, rela_text, rela_pos

    def _loss_weight(self, current_len, total_len, acc, task):
        hop_weight = self.args.hop_weight**(current_len)
        task_weight = self.args.task_weight if task=='TD' else 1
        acc_weight = self.args.acc_weight if acc==1 else 1
        return acc_weight / (hop_weight * task_weight)
        #hop_weight = self.args.hop_weight**(total_len-current_len)
        #task_weight = self.args.task_weight if task=='RC' else 1
        #acc_weight = self.args.acc_weight if acc==1 else 1
        #return hop_weight * task_weight * acc_weight

    def _single_step_rela_choose(self, model, ques, tuples, ques_pos):
        pos_tuples = [t for t in tuples if t[-1] == 1]
        neg_tuples = [t for t in tuples if t[-1] == 0]
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            return 0, 1, 'noNegativeInRC'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
            neg_tuples = neg_tuples[:self.args.neg_sample]

        if not self.args.change_ques:
            pos_rela, pos_rela_text, pos_rela_pos, _ = zip(*pos_tuples)
            neg_rela, neg_rela_text, neg_rela_pos, _ = zip(*neg_tuples)
        else:
            pos_rela, pos_rela_text, pos_rela_pos, pos_prev, pos_prev_text, _ = zip(*pos_tuples)
            neg_rela, neg_rela_text, neg_rela_pos, neg_prev, neg_prev_text, _ = zip(*neg_tuples)
        rev_rela = [[self.id2rela[r] for r in rela] for rela in pos_rela+neg_rela]

        ques = torch.LongTensor([ques]*(len(pos_rela)+len(neg_rela))).cuda()
        ques_pos = torch.LongTensor([ques_pos]*(len(pos_rela)+len(neg_rela))).cuda()
        relas, rela_texts, rela_pos = self._pad_rela(pos_rela, neg_rela, 
                            pos_rela_text, neg_rela_text, pos_rela_pos, neg_rela_pos)

        # deal with previous relations separately
        if self.args.change_ques:
            prevs, prev_texts, prev_pos = self._pad_rela(pos_prev, neg_prev, 
                            pos_prev_text, neg_prev_text, [],[])#pos_prev_pos, neg_prev_pos)
		
        if not self.args.change_ques:
            scores = model(ques, rela_texts, relas, ques_pos, rela_pos)
        else:
            scores = model(ques, rela_texts, relas, prev_texts, prevs)#maxlen)
        pos_scores = scores[0].repeat(len(scores)-1)
        neg_scores = scores[1:]
        ones = torch.ones(len(neg_scores)).cuda()
        loss = self.loss_function(pos_scores, neg_scores, ones)
        acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0

        rela_score=list(zip(scores.detach().cpu().numpy().tolist()[:], rev_rela[:]))
        '''
        scores = scores.data.cpu().numpy()
        pos_score = scores[0]
        print('pos', pos_score)
        scores = sorted(scores, reverse=True)
        print(scores)
        rank = scores.index(pos_score) + 1
        global total_rank
        global rank_count
        global total_all
        total_rank += rank
        total_all += len(scores)
        rank_count += 1
        print('average_rank', total_rank / rank_count)
        print('average_all', total_all / rank_count)
        input()
        '''
        return loss, acc, rela_score

    def _termination_decision(self, model, ques, tuples, next_tuples, ques_pos, movement):
        if movement == 'continue':
            pos_tuples = [t for t in next_tuples if t[-1] == 1]
            neg_tuples = [t for t in tuples if t[-1] == 1]
        elif movement == 'terminate':
            pos_tuples = [t for t in tuples if t[-1] == 1]
            neg_tuples = [t for t in next_tuples if t[-1] == 0]
        else:
            raise ValueError(f'Unknown movement:{movement} in UHop._termination_decision')

        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            return 0, 1, 'noNegativeInTD'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
            neg_tuples = neg_tuples[:self.args.neg_sample]
	
        if not self.args.change_ques:
            pos_rela, pos_rela_text, pos_rela_pos, _ = zip(*pos_tuples)
            neg_rela, neg_rela_text, neg_rela_pos, _ = zip(*neg_tuples)
        else:
            pos_rela, pos_rela_text, pos_rela_pos, pos_prev, pos_prev_text, _ = zip(*pos_tuples)
            neg_rela, neg_rela_text, neg_rela_pos, neg_prev, neg_prev_text, _ = zip(*neg_tuples)

        rev_rela = [[self.id2rela[r] for r in rela] for rela in pos_rela+neg_rela]

        ques = torch.LongTensor([ques]*(len(pos_rela)+len(neg_rela))).cuda()
        ques_pos = torch.LongTensor([ques_pos]*(len(pos_rela)+len(neg_rela))).cuda()
        relas, rela_texts, rela_pos = self._pad_rela(pos_rela, neg_rela, 
                            pos_rela_text, neg_rela_text, pos_rela_pos, neg_rela_pos)

        if self.args.change_ques:
            prevs, prev_texts, prev_pos = self._pad_rela(pos_prev, neg_prev, 
                            pos_prev_text, neg_prev_text, [], [])#pos_prev_pos, neg_prev_pos)

        if not self.args.change_ques:
            scores = model(ques, rela_texts, relas, ques_pos, rela_pos)
        else:
            scores = model(ques, rela_texts, relas, prev_texts, prevs)
        rela_score=list(zip(scores.detach().cpu().numpy().tolist()[:], rev_rela[:]))
        '''
        for q, r, t, s in zip(ques, relas, rela_texts, scores):
            print()
            print('q', q.data.cpu().numpy())
            print('r', r.data.cpu().numpy())
            print('t', t.data.cpu().numpy()) 
            print('s', s.data.cpu().numpy())
        input()
        '''
        pos_scores = scores[0].repeat(len(scores)-1)
        neg_scores = scores[1:]
        ones = torch.ones(len(neg_scores)).cuda()
        loss = self.loss_function(pos_scores, neg_scores, ones)
        acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0
        return loss, acc, rela_score

    def train(self, model):
        '''
        train 1 batch / question
        '''
        dataset = PerQuestionDataset(self.args, 'train', self.word2id, self.rela2id)
        if self.args.dataset.lower() == 'wq' or self.args.dataset.lower() == 'wq_train1test2':
            train_dataset, valid_dataset = random_split(dataset, 0.9, 0.1)
        else:
            train_dataset = dataset
            valid_dataset = PerQuestionDataset(self.args, 'valid', self.word2id, self.rela2id)
        datas = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=18, 
                pin_memory=False, collate_fn=quick_collate)
        optimizer = self.get_optimizer(model)
        earlystop_counter, min_valid_metric = 0, 100
        for epoch in range(0, self.args.epoch_num):
            model = model.train().cuda()
            total_loss, total_acc, total_rc_acc, total_td_acc = 0.0, 0.0, 0.0, 0.0
            loss_count, acc_count, rc_count, td_count = 0, 0, 0, 0
            for trained_num, (_, ques, step_list, ques_pos) in enumerate(datas):
                loss = torch.tensor(0, dtype=torch.float, requires_grad=True).cuda()
                acc_list = []
                optimizer.zero_grad();model.zero_grad()
                step_count = 0
                for i in range(len(step_list)-1):
                    if self.args.step_every_step:
                        optimizer.zero_grad();model.zero_grad();
                    step_loss, acc, _ = self._single_step_rela_choose(model, ques, step_list[i], ques_pos)
                    if not self.args.stop_when_err :
                        step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'RC')
                    if step_loss != 0:
                        if self.args.step_every_step:
                            step_loss.backward(); optimizer.step()
                        else:
                            step_count += 1
                            loss += step_loss
                        total_loss += step_loss.data; loss_count += 1
                    else:
                        loss_count += 1
                    acc_list.append(acc)
                    total_rc_acc += acc; rc_count += 1
                    if self.args.stop_when_err and acc != 1:
                        break
                    if i + 2 < len(step_list):
                        # do continue
                        if self.args.step_every_step:
                            optimizer.zero_grad();model.zero_grad(); 
                        step_loss, acc, _ = self._termination_decision(model, ques, step_list[i], step_list[i+1], ques_pos, 'continue')
                        if not self.args.stop_when_err :
                            step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                        if step_loss != 0:
                            if self.args.step_every_step:
                                step_loss.backward(); optimizer.step()
                            else:
                                step_count += 1
                                loss += step_loss
                            total_loss += step_loss.data; loss_count += 1
                        else:
                            loss_count += 1
                        acc_list.append(acc)
                        total_td_acc += acc; td_count += 1
                step_loss, acc, _ = self._termination_decision(model, ques, step_list[i], step_list[i+1], ques_pos, 'terminate')
                if step_loss != 0:
                    if self.args.step_every_step:
                        step_loss.backward(); optimizer.step()
                    else:
                        step_count += 1
                        loss += step_loss
                    total_loss += step_loss.data; loss_count += 1
                else:
                    loss_count += 1
                if not self.args.step_every_step:
                    loss /= (step_count if step_count>0 else 1)
                    loss.backward(); optimizer.step()
                acc_list.append(acc)
                total_td_acc += acc; td_count += 1
                acc = 1 if all([x == 1 for x in acc_list]) else 0
                total_acc += acc; acc_count += 1
                print(f'\r{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch {epoch} {trained_num}/{len(datas)} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f}', end='')
            _, valid_loss, valid_acc, _, _, _, score_debug = self.eval(model, 'valid', valid_dataset)
            if valid_loss < min_valid_metric:
                min_valid_metric = valid_loss
                earlystop_counter = 0
                save_model(model, self.args.path)
            else:
                earlystop_counter += 1
            if earlystop_counter > self.args.earlystop_tolerance:
                break       
        return model, total_loss / loss_count, total_acc / acc_count
            

    def eval(self, model, mode, dataset, output_result=False):
        '''
        For test and validation
        mode: valid | test
        '''
        output_labels, output_scores = [], []
        if model == None:
            import_model_str = 'from model.{} import Model as Model'.format(self.args.model)
            model = self.args.Model(self.args).cuda()
            model = load_model(model, self.args.path)
        model = model.eval().cuda()
        if dataset == None:
            dataset = PerQuestionDataset(self.args, mode, self.word2id, self.rela2id)
        datas = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=12, 
                pin_memory=False, collate_fn=quick_collate)
        total_loss, total_acc, total_rc_acc, total_td_acc = 0.0, 0.0, 0.0, 0.0
        loss_count, acc_count, rc_count, td_count = 0, 0, 0, 0
        for num, (_, ques, step_list, ques_pos) in enumerate(datas):
            labels, scores = [], []
            acc_list, rc_list = [], []
            for i in range(len(step_list)-1):
                loss, acc, rc_s = self._single_step_rela_choose(model, ques, step_list[i], ques_pos)
                labels.append('<CR>' if acc else '<WR>')
                scores.append(rc_s)
                acc_list.append(acc)
                rc_list.append(acc)
                if loss != 0:
                    total_loss += loss.data; loss_count += 1
                else:
                    loss_count += 1
                total_rc_acc += acc; rc_count += 1
                if self.args.stop_when_err and acc != 1:
                    break
                if i + 2 < len(step_list):
                    # do continue
                    loss, acc, td_s = self._termination_decision(model, ques, step_list[i], step_list[i+1], ques_pos, 'continue')
                    acc_list.append(acc)
                    if output_result and all(acc_list[:-1]):
                        labels.append(('<C>' if acc else '<T>'))
                        scores.append(td_s)
                    if loss != 0:
                        total_loss += loss.data; loss_count += 1
                    else:
                        loss_count += 1
                        total_td_acc += acc; td_count += 1
            loss, acc, td_s = self._termination_decision(model, ques, step_list[i], step_list[i+1], ques_pos, 'terminate')
            acc_list.append(acc)
            if output_result and all(acc_list[:-1]):
                labels.append('<T>' if acc else '<C>')
                scores.append(td_s)
            if loss != 0:
                total_loss += loss.data; loss_count += 1
            else:
                loss_count += 1
                total_td_acc += acc; td_count += 1
            total_td_acc += acc; td_count += 1
            acc = 1 if all(acc_list) else 0
            total_acc += acc; acc_count += 1
            output_labels.append('\t'.join(labels))
            output_scores.append(scores)
        print(f' Eval {num} Loss:{total_loss/loss_count:.5f} Acc:{total_acc/acc_count:.4f} RC_Acc:{total_rc_acc/rc_count:.2f} TD_Acc:{total_td_acc/td_count:.2f}', end='')
        print('')
        return model, total_loss / loss_count, total_acc / acc_count, total_rc_acc/rc_count, total_td_acc/td_count, '\n'.join(output_labels), output_scores
