import os

result_dir = '../save_model/'
model_name = 'improve_neural_relation_detection_model'
#model_name = 'single_side_word_by_word_attention_model'
#model_name = 'yp_ABWIM_model'

index = 0
result_list = []

save_model_dir = os.path.join(result_dir, model_name+f'_{index}')
while os.path.exists(save_model_dir):
    with open(os.path.join(save_model_dir, 'log.txt')) as f, open(os.path.join(save_model_dir, 'args.txt')) as f_args:
        args_txt = f_args.readline()
        lines = f.readlines()
        if len(lines) == 0:
            index += 1
            save_model_dir = os.path.join(result_dir, model_name+f'_{index}')
            continue
        print(index, lines[-8:])
        if len(lines) <= 8:
            index += 1
            #index += 1
            save_model_dir = os.path.join(result_dir, model_name+f'_{index}')
            continue
        #if 'train1test2=True' not in args_txt:
        #    index += 1
        #    save_model_dir = os.path.join(result_dir, model_name+f'_{index}')
        #    continue

        train_line_1 = lines[-8].strip()
        train_line_2 = lines[-7].strip()
        dev_line_1 = lines[-6].strip()
        dev_line_2 = lines[-5].strip()
        test_line_1 = lines[-3].strip()
        test_line_2 = lines[-2].strip()
        #print(lines[-3].split())
        #for i, n in enumerate(lines[-3].split()):
        #    print(i,n)
        print(test_line_1)
        tokens = test_line_1.split()
        loss = tokens[5]
        acc = tokens[7]
        rela_acc = tokens[10]
        deci_acc = tokens[13]
        print(f'index {index} {loss} acc {acc} rela_chose_acc {rela_acc} decision_acc {deci_acc}')
        result_list.append((str(index), acc, train_line_1, train_line_2, dev_line_1, dev_line_2, test_line_1, test_line_2, args_txt))
        #exit()
        index += 1
        save_model_dir = os.path.join(result_dir, model_name+f'_{index}')
result_list = sorted(result_list, key=lambda x: x[1])
print('sorted result...')
for result in result_list:
    print('\n'.join(result))
    print()
        
