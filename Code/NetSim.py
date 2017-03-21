import random
import math
import networkx as nx
import numpy as np
from scipy import spatial,stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def generate_graph_nodes(N,dimensions=2):
    G = nx.Graph()
    for node in range(N):
        pos = [random.random() for dim in range(dimensions)]
        G.add_node(node,pos=pos,weight=random.expovariate(1.0))
    return G

def dynamic_GTG(G,max_distance,alpha,beta,gamma):
    pos_points = list(nx.get_node_attributes(G, 'pos').values())
    point_tree = spatial.cKDTree(pos_points)
    potential_edges = point_tree.query_pairs(max_distance, len(pos_points[0]))
    for edge in potential_edges:
        dist = spatial.distance.euclidean(G.node[edge[0]]['pos'],G.node[edge[1]]['pos'])
        link_value = dist**-alpha
        link_strength = (G.node[edge[0]]['weight']+G.node[edge[1]]['weight'])*link_value
        if link_strength >= (beta*len(G))/(gamma*len(G.edges())+1):
            G.add_edge(edge[0],edge[1],weight=link_strength)                         
    return G

def integrate_threshold(N,beta,gamma):
    if gamma == 0:
        return N*beta
    else:
        return (beta*math.log(N*gamma+1))/gamma

def visualize_GTG(GTG):
    """
    Filename: nx_demo.py
    Authors: John Stachurski and Thomas J. Sargent
    """
    pos = nx.get_node_attributes(GTG, 'pos')    # Get positions of nodes
    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(GTG, pos, alpha=0.4)
    nx.draw_networkx_nodes(GTG, pos, nodelist=list(GTG.nodes()),
                           node_size=120, alpha=0.5,node_color='blue')
    plt.show()

def run_sim(N_list,iterations,max_distance_limits,beta_limits,gamma_limits,n_jobs):
    sim_data = Parallel(n_jobs=n_jobs)(delayed(simulation)(N,max_distance_limits,beta_limits,gamma_limits,i) 
                                                                            for i in range(iterations)
                                                                            for N in N_list)
                                                                        
    return sim_data

def simulation(N,max_distance_limits,beta_limits,gamma_limits,i):
    print('Running iteration ' + str(i) + ' for ' + str(N) + ' nodes')
    alpha=2
    G = generate_graph_nodes(N)
    max_distance = random.uniform(max_distance_limits[0], max_distance_limits[1])
    beta = random.uniform(beta_limits[0], beta_limits[1])
    gamma = random.uniform(gamma_limits[0], gamma_limits[1])
    AUC = integrate_threshold(N,beta,gamma)
    DGG = dynamic_GTG(G,max_distance,alpha,beta,gamma)
    connectivity = 2*len(DGG.edges())/len(DGG)
    mu = np.mean(list(nx.get_node_attributes(DGG, 'weight').values()))
    comps = sorted([len(G_comp) for G_comp in nx.connected_component_subgraphs(DGG)], reverse=True)
    if len(comps) > 1:
        first_comp = comps[0]/N
        second_comp = comps[1]/N
    else:
        first_comp = comps[0]/N
        second_comp = 0       
    return([N,mu,connectivity,max_distance,AUC,first_comp,second_comp])