import random
import math
import networkx as nx
import numpy as np
from scipy import spatial,stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def generate_graph_nodes(N,lamd):
    G = nx.Graph()
    dimensions = 2
    for node in range(N):
        pos = [random.random() for dim in range(dimensions)]
        G.add_node(node,pos=pos,weight=random.expovariate(lamd))
    return G

def generate_network(N,lamd,R,alpha,theta):
    #TODO: if R is too large, dont make KDTree
    base_G = generate_network(N,lamd)
    pos = nx.get_node_attributes(base_G, 'pos')
    weight = nx.get_node_attributes(base_G, 'weight')
    if R == 0:
        G = nx.geographical_threshold_graph(N,theta,alpha,pos=pos,weight=weight)
    else:
        G = base_G.copy()
        pos_points = list(pos.values())
        point_tree = spatial.cKDTree(pos_points)
        potential_edges = point_tree.query_pairs(R, len(pos_points[0]))#len(pos_points[0]) = dimensions of the plane (ie. 2)
        for edge in potential_edges:
            dist = np.linalg.norm(edge[0]['pos']-edge[1]['pos'])
            link_prob = dist**-alpha
            link_strength = (G.node[edge[0]]['weight']+G.node[edge[1]]['weight'])*link_prob
            if link_strength >= theta:
                G.add_edge(edge[0],edge[1],weight=link_strength)                         
    return G

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

def run_sim(N_list,iterations,max_distance_limits,theta_limits,n_jobs):
    sim_data = Parallel(n_jobs=n_jobs)(delayed(simulation)(N,max_distance_limits,theta_limits,i) 
                                                                            for i in range(iterations)
                                                                            for N in N_list)
                                                                        
    return sim_data

def simulation(N,max_distance_limits,theta_limits,i):
    print('Running iteration ' + str(i) + ' for ' + str(N) + ' nodes')
    alpha=0
    G = generate_graph_nodes(N)
    max_distance = random.uniform(max_distance_limits[0], max_distance_limits[1])
    theta = random.uniform(theta_limits[0], theta_limits[1])*N
    sim_GTG = build_GTG(G,max_distance,theta)
    K = sim_GTG.number_of_edges()
    connectivity = 2*K/N
    mu = np.mean(list(nx.get_node_attributes(sim_GTG, 'weight').values()))
    comps = sorted([len(G_comp) for G_comp in nx.connected_component_subgraphs(sim_GTG)], reverse=True)
    if len(comps) > 1:
        first_comp = comps[0]/N
        second_comp = comps[1]/N
    else:
        first_comp = comps[0]/N
        second_comp = 0       
    return([N,K,mu,connectivity,max_distance,theta,first_comp,second_comp])