import pickle
import json
import sys

pkl_path = sys.argv[1]
output_path = sys.argv[2]

with open(pkl_path, 'rb') as f, open(output_path, 'a') as f_out:
    l = pickle.load(f)
    for i in l:
        json.dump(i, f_out)
        f_out.write('\n')

