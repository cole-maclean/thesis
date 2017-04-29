import sys,ast
import simulator
from joblib import Parallel, delayed
import random
import tqdm

if __name__ == '__main__':
    limit_dict = ast.literal_eval(sys.argv[1])
    iterations = ast.literal_eval(sys.argv[2])
    step_size = ast.literal_eval(sys.argv[3])
    n_jobs = ast.literal_eval(sys.argv[4])
    save_file = sys.argv[5]
    Parallel(n_jobs=n_jobs)(delayed(simulator.lower_bound_R)(N,limit_dict[N],iterations,max_N,step_size,save_file) for N in limit_dict.keys())
