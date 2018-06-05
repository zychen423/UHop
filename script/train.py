from sklearn.model_selection import ParameterGrid
import subprocess

param_dict = {
        #'--model': ['improve_neural_relation_detection_model'], 
        '--model': ['yp_ABWIM_model'], 
        '--train': [True],  
        '--epoch_num': [1000], 
        '--emb_size': [300], 
        '--hidden_size': [128], 
        '--num_layers': [1], 
        '--bidirectional': [True], 
        '--dropout_rate': [0.3, 0.35], 
        '--learning_rate': [0.00001, 0.000005, 0.000001], 
        '--optimizer': ['rmsprop'], 
        '--sample_num': [1024], 
        '--weight_decay': [0], 
        '--margin': [0.1], 
        '--stop_when_err': [True],  
        '--earlystop_tolerance': [20], 
        #'--train_embedding': [True], 
        #'--train1test2': [True],
        }

process_str = 'python3.6 TBRE.py'

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


