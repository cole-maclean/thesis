import sys,ast
import simulator
from joblib import Parallel, delayed
import random
import tqdm
import json

if __name__ == '__main__':
	N = ast.literal_eval(sys.argv[1])
	lamd = ast.literal_eval(sys.argv[2])
	R_list = ast.literal_eval(sys.argv[3])
	R_list = [str(R) for R in R_list]
	unchange_limit = ast.literal_eval(sys.argv[4])
	n_jobs = ast.literal_eval(sys.argv[5])
	with open("..\Simulation_Data\/theta_bounds.json", 'r') as infile:
		limits_dict = json.load(infile)
	results = Parallel(n_jobs=n_jobs)(delayed(simulator.upper_bound_theta)(N,lamd,R,limits_dict[R],unchange_limit) for R in R_list)
	for rslt in results:
		limits_dict[rslt[0]] = rslt[1]
	with open("..\Simulation_Data\/theta_bounds.json", 'w') as outfile:
		json.dump(limits_dict,outfile)
