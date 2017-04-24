import sys,ast
import random
import math
import igraph
import networkx as nx
import numpy as np
from scipy import spatial,stats
from joblib import Parallel, delayed

def generate_graph_nodes(N,lamd):
    g = igraph.Graph()
    dimensions = 2
    for node in range(N):
        pos = [random.random() for dim in range(dimensions)]
        g.add_vertex(pos=pos,weight=random.expovariate(lamd))
    return g

def generate_network(N,lamd,R,alpha,theta,beta):
    #theta is a function parameterized by the instantaneous # of edges in the graph, cannot be dynamic for GTGs
    #TODO: if R is too large, dont make KDTree
    base_g = generate_graph_nodes(N,lamd)
    pos = base_g.vs['pos']
    weight = base_g.vs['weight']
    if R == 0:
        g = nx.geographical_threshold_graph(N,theta,alpha,pos=pos,weight=weight)
    else:
        g = base_g.copy()
        point_tree = spatial.cKDTree(pos)
        potential_edges = point_tree.query_pairs(R, 2)#dimensions = 2
        for edge in potential_edges:
            dist = np.linalg.norm(np.array(g.vs['pos'][edge[0]])-np.array(g.vs['pos'][edge[1]]))
            link_prob = dist**-alpha
            link_strength = (g.vs['weight'][edge[0]]+g.vs['weight'][edge[1]])*link_prob
            if alpha == 0:
                threshold = theta/(1+beta*g.ecount())
            else:
                threshold = N*theta/(1+beta*g.ecount())
            if link_strength >= threshold:
                g.add_edge(edge[0],edge[1],weight=link_strength)                         
    return g


def simulation(N_list,lamd_limits,R_limits,alpha_limits,theta_limits,beta_limits,i):
    removal_percent = 0.05
    N = random.sample(N_list,1)[0]
    lamd = random.uniform(lamd_limits[0], lamd_limits[1])
    R = random.uniform(R_limits[0], R_limits[1])
    alpha = random.uniform(alpha_limits[0], alpha_limits[1])
    theta = random.uniform(theta_limits[0], theta_limits[1])
    beta = random.uniform(beta_limits[0], beta_limits[1])
    print ("Running Simulation i = %s N = %s lamda = %s R = %s alpha = %s theta = %s  beta = %s" % (i,N,lamd,R,alpha,theta,beta))
    g = generate_network(N,lamd,R,alpha,theta,beta)
    K = g.ecount()
    connectivity = 2*K/N
    weights = g.vs['weight']
    mu = np.mean(weights)
    comps = g.components()
    all_comps = [comp/N for comp in comps.sizes()]
    first_comp = comps.giant().vcount()/N
    diameter = g.diameter()
    total_weight = sum(weights)
    endurance = 0 
    for remove_count in range(100):#segment removal of removal_percent of nodes into 100 discrete instances ie. remove (removal_percent/100)*N nodes at a time
        remove_nodes = random.sample(range(g.vcount()),int(removal_percent/100*N))
        g.delete_vertices(remove_nodes)
        failure_g_big_weight = sum(g.components().giant().vs['weight'])
        endurance = endurance + (1-failure_g_big_weight/total_weight)*removal_percent/100
    return([N,lamd,R,alpha,theta,beta,K,mu,connectivity,first_comp,all_comps,diameter,endurance/removal_percent,removal_percent])

def add_sim_data(data_file,new_data):
    with open(data_file, 'r') as infile:
        loaded_sim_data = json.load(infile)
    for el in new_data:
        loaded_sim_data.append(el)
    with open(data_file, 'w') as outfile:
        json.dump(loaded_sim_data, outfile)
    return loaded_sim_data

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
    sim_data = Parallel(n_jobs=n_jobs,verbose=verbose)(delayed(simulation)(N_list,lamd_limits,R_limits,alpha_limits,theta_limits,beta_limits,i) 
                                                                            for i in range(iterations)
                                                                        )                                              
    add_sim_data(save_file,sim_data)
