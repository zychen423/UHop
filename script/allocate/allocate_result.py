import os
import sys
import json

result_dir = f'../../{sys.argv[1]}/'
model_name = sys.argv[2]
requirements = sys.argv[3:]

index = 0
result_list = []
save_model_files = os.listdir(result_dir)
for model_dir_name in [x for x in save_model_files if x.startswith(model_name)]:
    save_model_dir = os.path.join(result_dir, model_dir_name)
#    print(save_model_dir)
    if os.path.exists(save_model_dir):
        with open(os.path.join(save_model_dir, 'args.txt')) as f_args:
            args = json.load(f_args)
     #       if args['dataset']=='pql2':
            model_name_list =  [x for x in os.listdir(save_model_dir) if x.startswith('scores_')]
            try:
                test_acc  = model_name_list[0].replace('.json', '').replace('scores_', '')
            except IndexError:
#                print(model_name_list, 'error', 'continue')
                continue
            test_acc = float(test_acc)
            #test_loss, test_acc, rc, td = float(test_loss), float(test_acc), float(rc), float(td)
            for i in range(0, len(requirements), 2):
                key = requirements[i]
                value = requirements[i+1]
                if str(args[key]) != str(value):
                    break
            else:
                result_list.append((save_model_dir, test_acc, args))
				#[args['hidden_size'], args['dropout_rate'], args['l2_norm'], args['margin'], args['learning_rate']], args['dataset']))
                #result_list.append((save_model_dir, test_acc, test_loss, str(args)))
result_list = sorted(result_list, key=lambda x: x[1], reverse=True)#(x[3][3], x[3][2], x[1]))
temp=[]
for result in result_list:
    test_acc, args = result[1], result[2]
    hidden_size, margin, learning_rate, dropout = args['hidden_size'], args['margin'], args['learning_rate'], args['dropout_rate']
    hop_w, task_w, acc_w = args['hop_weight'], args['task_weight'], args['acc_weight']#, args['reduce_method']
    #hidden_size, margin, learning_rate = result[5][0], result[5][3], result[5][4]
    #dropout = result[5][1]
    print(f'acc={test_acc:.4f} (hidden={hidden_size}, margin={margin}, learning_rate={learning_rate}, dropout_rate={dropout}, weight={(hop_w, task_w, acc_w)}) \
    {result[0].replace(result_dir,"")}')#, {method}')
    #print(f'acc={test_acc:.4f}, loss={test_loss:.4f}, rc={rc:.2f}, td={td:.2f} (hidden={hidden_size}, margin={margin}, learning_rate={learning_rate:5}, dropout_rate={dropout}) {result[0].replace(result_dir,"")}')
    temp.append(result[0].replace(result_dir,""))
#print(' '.join(temp))
