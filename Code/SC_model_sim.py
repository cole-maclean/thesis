import sys,ast
import simulator
import multiprocessing as mp
import random
import tqdm

if __name__ == '__main__':
    N = ast.literal_eval(sys.argv[1])
    R = ast.literal_eval(sys.argv[2])
    theta = ast.literal_eval(sys.argv[3]) #Lower, upper, step size
    model_list = ast.literal_eval(sys.argv[4])
    iterations = ast.literal_eval(sys.argv[5])
    n_jobs = ast.literal_eval(sys.argv[6])
    save_file = sys.argv[7]
    sim_data = []
    sim_parameters = [[N,R,theta,model] for model in model_list for i in range(iterations)]
    p = mp.Pool(n_jobs)

    for rslt in tqdm.tqdm(p.imap_unordered(simulator.sim_SC_net, sim_parameters,chunksize=1), total=len(sim_parameters),smoothing=0):
        sim_data.append(rslt)  
    simulator.add_sim_data(save_file,sim_data)
