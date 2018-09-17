import sys
import subprocess
import numpy as np
import pandas as pd
from pandas import DataFrame

process_str = f"python3.6 best_result.py {' '.join(sys.argv[1:])} neg_sample 1024"

learning_rates = ['5e-05', '0.0001', '0.001']
hidden_sizes = ['100', '150']
margins = ['0.1', '0.3', '0.5', '0.7', '1.0']

results = np.zeros((3,2,5))

for i, learning_rate in enumerate(learning_rates):
    for j, hidden_size in enumerate(hidden_sizes):
        for k, margin in enumerate(margins):
            run_str = process_str+f' learning_rate {learning_rate} hidden_size {hidden_size} margin {margin}'
            process = subprocess.run(run_str.split(), encoding='UTF-8', stdout=subprocess.PIPE)
            acc = process.stdout.split(' ')[0].replace('acc=', '')[:-1]
            acc = -1 if acc=='' else 100*float(acc)
            results[i][j][k] = acc

results = results.reshape(3,-1)
argmax = [learning_rates[j] if results[j][i]>0 else 0 for i,j in enumerate(results.argmax(axis=0))]
pd.set_option('display.width', 100)
df = DataFrame(data=results.tolist()+[argmax], index=learning_rates+['max_lr'], columns=[f'{h}/{m}' for h in hidden_sizes for m in margins])
print(f'\n{df}\n')
