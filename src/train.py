from sklearn.model_selection import ParameterGrid
import subprocess

param_dict = {
        '--model': ['HR_BiLSTM'], 
        '--epoch_num': [1000], 
        '--emb_size': [300], 
        '--hidden_size': [128, 256, 512], 
        '--dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
        '--learning_rate': [0.001, 0.0005, 0.0001], 
        '--optimizer': ['rmsprop'], 
        '--neg_sample': [1024], 
        '--l2_norm': [0.0, 0.00000001, 0.000000001], 
        '--margin': [0.1], 
        '--stop_when_err': [True],  
        '--earlystop_tolerance': [20], 
        '--dataset': ['wq'], 
        #'--train_embedding': [True], 
        #'--train1test2': [True],
        }

process_str = 'python3.6 main.py --train'

################################ NEED TO CONFIGURE ABOVE ####################################################

possible_param_list = list(ParameterGrid(param_dict))
#print(possible_param_list)
print(f'There will be {len(possible_param_list)} runs')
for i, param in enumerate(possible_param_list):
    print(f'{i}/{len(possible_param_list)} run')
    # param is a dict
    run_str = process_str
    for arg, arg_val in param.items():
        run_str += f' {arg} {arg_val}'
    print(run_str)
    process = subprocess.run(run_str.split(), encoding='UTF-8')
    #print(process.stdout.split('\n')[-3])


