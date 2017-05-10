import sys,ast
import simulator
import multiprocessing as mp
import random
import tqdm

if __name__ == '__main__':
    N_list = ast.literal_eval(sys.argv[1])
    lamd_list = ast.literal_eval(sys.argv[2])
    R_limits = ast.literal_eval(sys.argv[3]) #Lower, upper, step size
    alpha_limits = ast.literal_eval(sys.argv[4])
    theta_limits =  ast.literal_eval(sys.argv[5])
    beta_limits = ast.literal_eval(sys.argv[6])
    load_g_file = sys.argv[7]
    iterations = ast.literal_eval(sys.argv[8])
    n_jobs = ast.literal_eval(sys.argv[9])
    verbose = ast.literal_eval(sys.argv[10])
    save_file = sys.argv[11]
    sim_data = []
    sim_parameters = [[N,
                      lamd,
                      R/1000,
                      random.uniform(alpha_limits[0], alpha_limits[1]),
                      theta/1000,
                      random.uniform(beta_limits[0], beta_limits[1]),
                      load_g_file]
                      for lamd in lamd_list
                      for R in range(R_limits[0],R_limits[1],R_limits[2])
                      for theta in range(theta_limits[0],theta_limits[1],theta_limits[2])
                      for N in N_list
                      for i in range(iterations)]
    p = mp.Pool(n_jobs)

    for rslt in tqdm.tqdm(p.imap_unordered(simulator.simulation, sim_parameters,chunksize=1), total=len(sim_parameters),smoothing=0):
        sim_data.append(rslt)  
    simulator.add_sim_data(save_file,sim_data)
