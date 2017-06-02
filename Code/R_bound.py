import sys,ast
import simulator
from joblib import Parallel, delayed
import random
import tqdm
import json

if __name__ == '__main__':
	N_list = ast.literal_eval(sys.argv[1])
	N_list = [str(N) for N in N_list]
	unchange_limit = ast.literal_eval(sys.argv[2])
	n_jobs = ast.literal_eval(sys.argv[3])
	save_file = sys.argv[4]
	with open("..\Simulation_Data\R_bounds.json", 'r') as infile:
		limits_dict = json.load(infile)
	for N in N_list:
		if N not in limits_dict.keys():
			limits_dict[N] = 0.2
	results = Parallel(n_jobs=n_jobs)(delayed(simulator.lower_bound_R)(N,limits_dict[N],unchange_limit) for N in N_list)
	for rslt in results:
		limits_dict[rslt[0]] = rslt[1]
	with open("..\Simulation_Data\R_bounds.json", 'w') as outfile:
		json.dump(limits_dict,outfile)
