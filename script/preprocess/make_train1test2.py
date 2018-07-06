import sys
import json

one_list = []
two_list = []

with open(sys.argv[1], 'r') as f_in:
    for line in f_in:
        obj = json.loads(line.strip())
        if len(obj[2]) == 2:
            one_list.append(obj)
        else:
            two_list.append(obj)
with open(sys.argv[2], 'r') as f_in:
    for line in f_in:
        obj = json.loads(line.strip())
        if len(obj[2]) == 2:
            one_list.append(obj)
        else:
            two_list.append(obj)
with open(sys.argv[3], 'w') as f_one:
    for one_obj in one_list:
        f_one.write(json.dumps(one_obj, ensure_ascii=False))
        f_one.write('\n')
with open(sys.argv[4], 'w') as f_two:
    for two_obj in two_list:
        f_two.write(json.dumps(two_obj, ensure_ascii=False))
        f_two.write('\n')
