import sys,ast
import simulator
import multiprocessing as mp
import random
import tqdm
import json

if __name__ == '__main__':
	N_list = ast.literal_eval(sys.argv[1])
	N_list = [str(N) for N in N_list]
	R_range = ast.literal_eval(sys.argv[2])
	R_step = ast.literal_eval(sys.argv[3])
	iterations = ast.literal_eval(sys.argv[4])
	n_jobs = ast.literal_eval(sys.argv[5])
	save_file = sys.argv[6]
	sim_data = []
	with open("..\Simulation_Data\/theta_bounds.json", 'r') as infile:
		limits_dict = json.load(infile)
	sim_params = []
	for N in N_list:
		if N in limits_dict.keys():
			for R in range(R_range[0],R_range[1],R_step):
				if R in limits_dict[N].keys():
					sim_params.append([N,R,limits_dict[N][R],iterations,save_file])
				else:
					sim_params.append([N,R,0.001,iterations,save_file])
		else:
			for R in range(R_range[0],R_range[1],R_step):
				sim_params.append([N,R,0.001,iterations,save_file])
	p = mp.Pool(n_jobs)
	for rslt in tqdm.tqdm(p.imap_unordered(simulator.upper_bound_theta, sim_params,chunksize=1), total=len(sim_params),smoothing=0):
		if rslt[0] in limits_dict.keys():
			limits_dict[rslt[0]][rslt[1]] = rslt[2]
		else:
			limits_dict[rslt[0]] = {rslt[1]:rslt[2]}	
		sim_data = sim_data + rslt[3]
	simulator.add_sim_data(save_file,sim_data) 
	with open("..\Simulation_Data\/theta_bounds.json", 'w') as outfile:
		json.dump(limits_dict,outfile)