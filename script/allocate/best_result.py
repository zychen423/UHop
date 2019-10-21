import os
import sys
import json

result_dir = '../../saved_model/'
model_name = sys.argv[1]
requirements = sys.argv[2:]

index = 0
result_list = []
keys = ['exp2t', requirements[1], model_name, requirements[3]]
save_model_files = os.listdir(result_dir)
for model_dir_name in [x for x in save_model_files if x.startswith(model_name)]:
    save_model_dir = os.path.join(result_dir, model_dir_name)
    if os.path.exists(save_model_dir):
        with open(os.path.join(save_model_dir, 'args.txt')) as f_args:
            args = json.load(f_args)
            model_name_list =  [x for x in os.listdir(save_model_dir) if x.startswith('scores_')]
            try:
                test_acc  = model_name_list[0].replace('.json', '').replace('scores_', '')
                #test_loss, test_acc, rc, td = model_name_list[0].replace('.pth', '').split('_')[1:5]
            except IndexError:
#                print(model_name_list, 'error', 'continue')
                continue
            test_acc = float(test_acc.replace('.pth', ''))
            #test_loss, test_acc, rc, td = float(test_loss), float(test_acc), float(rc), float(td)
            for i in range(0, len(requirements), 2):
                key = requirements[i]
                value = requirements[i+1]
                if key not in args or str(args[key]) != str(value):
                    break
            else:
                result_list.append((save_model_dir, test_acc, [args['hidden_size'], args['dropout_rate'], args['l2_norm'], args['margin'], args['learning_rate']], args['dataset']))
result_list = sorted(result_list, key=lambda x: x[1])#(x[3][3], x[3][2], x[1]))
temp=[]
for result in result_list[-1:]:
    test_acc = result[1]#, test_loss, rc, td = result[1], result[2], result[3], result[4]
    hidden_size, margin, learning_rate = result[2][0], result[2][3], result[2][4]
    dropout = result[2][1]
    print(f'acc={test_acc:.4f} (hidden={hidden_size}, margin={margin}, learning_rate={learning_rate:5}, dropout_rate={dropout}) {result[0].replace(result_dir,"")}')
    temp.append(result[0].replace(result_dir,""))
