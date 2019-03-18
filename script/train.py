from sklearn.model_selection import ParameterGrid
import subprocess
import os
import random

param_dict = {
        # Traing Setting:
        '--model': ['ABWIM_plus'],# 'HR_BiLSTM_plus'], 
        '--dataset': ['pql3'],# 'pqm8'], 
        '--framework': ['UHop'],# 'baseline'], 
        '--train': [' '],  
        #'--train1test2': [True],
        '--earlystop_tolerance': [20],

        '--saved_dir':['bert'],
#        '--stop_when_err': [''],    # comment to use weighted loss 
#        '--step_every_step' : [''], # comment to step only once in each question
        '--dynamic' : ['flatten'],     # none/flatten/recurrent
        '--reduce_method' : ['dense'], # bilstm or dense, for abwim_plus
#        '--only_one_hop' : [''], # uncomment for one-hop-only training/testing
        
        # HyperParameter:
        '--epoch_num': [1000], 
        '--emb_size': [300], 
        '--hidden_size': [150],
        '--dropout_rate': [0.2],
        '--learning_rate': [0.0001],# 0.01, 0.001],
        '--optimizer': ['rmsprop'], 
        '--neg_sample': [1024], 
        '--l2_norm': [0.0],# 0.00000001, 0.000000001], 
        '--margin': [0.1],
        #'--train_embedding': [True], 
        '--hop_weight': [1],#0.8, 1, 1.25],
        '--task_weight': [1],
        '--acc_weight': [1],# 0.25, 0.5, 1]
        '--pretrained_bert': ['bert-base-uncased'],#'bert-base-uncased' or 'bert-base-multilingual-cased'
        '--q_representation': ['lstm']
        }

os.chdir('../src/')
process_str = 'python3.6 main.py'

################################ NEED TO CONFIGURE ABOVE ####################################################

possible_param_list = list(ParameterGrid(param_dict))
random.shuffle(possible_param_list)
#print(possible_param_list)
print(f'There will be {len(possible_param_list)} runs')
for i, param in enumerate(possible_param_list[:]):
    print(f'{i}/{len(possible_param_list)} run')
    # param is a dict
    run_str = process_str
    for arg, arg_val in param.items():
        run_str += f' {arg} {arg_val}'
    print(run_str)
    process = subprocess.run(run_str.split(), encoding='UTF-8')
    #print(process.stdout.split('\n')[-3])


