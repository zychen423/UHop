import os
import sys
import json

result_dir = '../save_model/'
model_name = sys.argv[1]
requirements = sys.argv[2:]

index = 0
result_list = []

save_model_files = os.listdir(result_dir)
for model_dir_name in [x for x in save_model_files if x.startswith(model_name)]:
    save_model_dir = os.path.join(result_dir, model_dir_name)
    print(save_model_dir)
    if os.path.exists(save_model_dir):
        with open(os.path.join(save_model_dir, 'args.txt')) as f_args:
            args = json.load(f_args)
            model_name_list =  [x for x in os.listdir(save_model_dir) if x.startswith('model_')]
            try:
                test_loss, test_acc = model_name_list[0].split('_')[1], model_name_list[0].split('_')[2]
            except IndexError:
                print(model_name_list, 'error', 'continue')
                continue
            test_acc = test_acc.replace('.pth', '')
            test_loss, test_acc = float(test_loss), float(test_acc)
            for i in range(0, len(requirements), 2):
                key = requirements[i]
                value = requirements[i+1]
                if str(args[key]) != str(value):
                    break
            else:
                result_list.append((save_model_dir, test_acc, test_loss, str(args)))
result_list = sorted(result_list, key=lambda x: x[1])
print('sorted result...')
for result in result_list:
    print(result)
    print()
        
