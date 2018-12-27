import subprocess
import pandas as pd
from pandas import DataFrame

FRAMEWORK = [ 'UHop']
MODEL = ['HR_BiLSTM', 'ABWIM']
DATASET = ['pq2', 'pq3', 'pq1', 'pql2', 'pql3', 'pql1', 'wq', 'exp4']

process_str = 'python3.6 best_result.py'

results = []
index = []
for framework in FRAMEWORK:
    for model in MODEL:
        result = []
        for dataset in DATASET:
            run_str = process_str+f' {model} framework {framework} dataset {dataset} neg_sample 1024 optimizer rmsprop' #acc_weight 0.4/1.0
            process = subprocess.run(run_str.split(), encoding='UTF-8', stdout=subprocess.PIPE)
            acc = process.stdout.split(' ')[0].replace('acc=', '')[:-1]
            acc = '    X' if acc=='' else float(acc)
            #acc = process.stdout.split(' ')[-1].split('_')[-1][:-1]
            result.append(acc)
        results.append(result)
        index.append(framework+'/'+model)
pd.set_option('display.width', 150)
print(DataFrame(data=results, index=index, columns=DATASET))
