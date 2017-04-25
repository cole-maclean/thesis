import random
import math
import igraph
import networkx as nx
import numpy as np
from scipy import spatial,stats
import json

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

def simulation(sim_parameters):
    removal_percent = 0.05
    N,lamd,R,alpha,theta,beta = sim_parameters
    #print ("N = %s lamda = %s R = %s alpha = %s theta = %s  beta = %s" % (N,lamd,R,alpha,theta,beta))
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
    return([N,lamd,R,alpha,theta,beta,K,mu,connectivity,first_comp,all_comps[0:3],diameter,endurance/removal_percent,removal_percent])

def add_sim_data(data_file,new_data):
    with open(data_file, 'r') as infile:
        loaded_sim_data = json.load(infile)
    updated_dataset = loaded_sim_data + new_data
    with open(data_file, 'w') as outfile:
        json.dump(updated_dataset, outfile)
    return updated_dataset