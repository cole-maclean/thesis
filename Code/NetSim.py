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

def generate_network(N,lamd,R,alpha,theta,beta):
    #theta is a function parameterized by the instantaneous # of edges in the graph, cannot be dynamic for GTGs
    #TODO: if R is too large, dont make KDTree
    base_G = generate_graph_nodes(N,lamd)
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
            dist = np.linalg.norm(np.array(G.node[edge[0]]['pos'])-np.array(G.node[edge[1]]['pos']))
            link_prob = dist**-alpha
            link_strength = (G.node[edge[0]]['weight']+G.node[edge[1]]['weight'])*link_prob
            if alpha == 0:
                threshold = theta/(1+beta*G.number_of_edges())
            else:
                threshold = N*theta/(1+beta*G.number_of_edges())
            if link_strength >= threshold:
                G.add_edge(edge[0],edge[1],weight=link_strength)                         
    return G

def visualize_network(G):
    """
    Filename: nx_demo.py
    Authors: John Stachurski and Thomas J. Sargent
    """
    pos = nx.get_node_attributes(G, 'pos')    # Get positions of nodes
    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()),
                           node_size=120, alpha=0.5,node_color='blue')
    plt.show()

def run_sim(N_list,lamd_limits,R_limits,alpha_limits,theta_limits,beta_limits,iterations,n_jobs):
    sim_data = Parallel(n_jobs=n_jobs)(delayed(simulation)(N,lamd_limits,R_limits,alpha_limits,theta_limits,beta_limits,i) 
                                                                            for i in range(iterations)
                                                                            for N in N_list
                                                                        )                                              
    return sim_data

def simulation(N,lamd_limits,R_limits,alpha_limits,theta_limits,beta_limits,i):
    removal_percent = 0.1
    lamd = random.uniform(lamd_limits[0], lamd_limits[1])
    R = random.uniform(R_limits[0], R_limits[1])
    alpha = random.uniform(alpha_limits[0], alpha_limits[1])
    theta = random.uniform(theta_limits[0], theta_limits[1])
    beta = random.uniform(beta_limits[0], beta_limits[1])
    print ("Running Simulation i = %s N = %s lamda = %s R = %s alpha = %s theta = %s  beta = %s" % (i,N,lamd,R,alpha,theta,beta))
    G = generate_network(N,lamd,R,alpha,theta,beta)
    K = G.number_of_edges()
    connectivity = 2*K/N
    mu = np.mean(list(nx.get_node_attributes(G, 'weight').values()))
    comps = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    if len(comps) > 1:
        first_comp = len(comps[0])/N
        second_comp = len(comps[1])/N
    else:
        first_comp = len(comps[0])/N
        second_comp = 0
    diameter = nx.algorithms.diameter(comps[0])
    big_weight = sum(nx.get_node_attributes(comps[0], 'weight').values())
    resilence = []
    for remove_count in range(math.ceil(len(G)*removal_percent)):
        remove_node = random.sample(G.nodes(),1)[0]
        G.remove_node(remove_node)
        failure_G_comps = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
        failure_G_big_weight = sum(nx.get_node_attributes(failure_G_comps[0], 'weight').values())
        resilence.append(failure_G_big_weight/big_weight)
    return([N,lamd,R,alpha,theta,beta,K,mu,connectivity,first_comp,second_comp,diameter,resilence,removal_percent])