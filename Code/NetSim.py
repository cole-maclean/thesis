import sys,ast
import simulator
from joblib import Parallel, delayed
import multiprocessing as mp
import random
import tqdm

if __name__ == '__main__':
    N_list = ast.literal_eval(sys.argv[1])
    lamd_limits = ast.literal_eval(sys.argv[2])
    R_limits = ast.literal_eval(sys.argv[3])
    alpha_limits = ast.literal_eval(sys.argv[4])
    theta_limits =  ast.literal_eval(sys.argv[5])
    beta_limits = ast.literal_eval(sys.argv[6])
    iterations = ast.literal_eval(sys.argv[7])
    n_jobs = ast.literal_eval(sys.argv[8])
    verbose = ast.literal_eval(sys.argv[9])
    save_file = sys.argv[10]
    sim_data = []
    sim_parameters = [[random.sample(N_list,1)[0],
                      random.uniform(lamd_limits[0], lamd_limits[1]),
                      random.uniform(R_limits[0], R_limits[1]),
                      random.uniform(alpha_limits[0], alpha_limits[1]),
                      random.uniform(theta_limits[0], theta_limits[1]),
                      random.uniform(beta_limits[0], beta_limits[1])] for i in range(iterations)]
    p = mp.Pool(n_jobs)

    for rslt in tqdm.tqdm(p.imap_unordered(simulator.simulation, sim_parameters,chunksize=1), total=len(sim_parameters)):
        sim_data.append(rslt)  
    simulator.add_sim_data(save_file,sim_data)
