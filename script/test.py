from sklearn.model_selection import ParameterGrid
import subprocess
import os
import random
import json
import sys

dir_name = sys.argv[1:]

for dir in dir_name:

	path=f'../{dir}'

	with open(f'{path}/args.txt', 'r') as f:
		model_info = json.load(f)

	model_name = model_info['model']
	dataset = model_info['dataset']
	framework = model_info['framework']
	hidden = model_info['hidden_size']
	margin = model_info['margin']

	print(f'{framework}_{model_name}_{dataset}  (from {path}, hidden_size={hidden})')

	param_dict = {
			# Traing Setting:
			'--model': [model_name], 
			'--dataset': [dataset],
			'--framework': [framework],
			'--test': [' '],  

			'--stop_when_err': [''],  
			'--earlystop_tolerance': [20],
			'--path' : [path],

			# HyperParameter:
			'--epoch_num': [1000], 
			'--emb_size': [300], 
			'--hidden_size': [hidden],
			'--margin': [margin]
			}

	os.chdir('../src/')
	process_str = 'python3.6 main.py'
################################ NEED TO CONFIGURE ABOVE ####################################################

	possible_param_list = list(ParameterGrid(param_dict))
	for param in possible_param_list:
		# param is a dict
		run_str = process_str
		for arg, arg_val in param.items():
			run_str += f' {arg} {arg_val}'
		print(run_str)
		process = subprocess.run(run_str.split(), encoding='UTF-8')
