import random
import math
import igraph
import networkx as nx
import numpy as np
from scipy import spatial,stats
import json
import tqdm

def generate_graph_nodes(N,lamd):
    g = igraph.Graph()
    dimensions = 2
    for node in range(N):
        pos = [random.random() for dim in range(dimensions)]
        if lamd == 0:
            g.add_vertex(pos=pos,weight=1)
        else:
            g.add_vertex(pos=pos,weight=np.random.power(lamd))
    return g

def threshold_edge(g,N,i,j,alpha,theta,beta):
    if alpha == 0:
        link_prob = 1
    else:
        dist = np.linalg.norm(np.array(i['pos'])-np.array(j['pos']))
        link_prob = dist**-alpha
    link_strength = (i['weight']+j['weight'])*link_prob
    if beta == 0:
        threshold = theta
    elif alpha == 0:
        threshold = theta/(1+beta*g.ecount())
    else:
        threshold = N*theta/(1+beta*g.ecount())
    if link_strength >= threshold:
        return True
    else:
        return False

def generate_network(N,lamd,R,alpha,theta,beta):
    g = generate_graph_nodes(N,lamd)
    pos = g.vs['pos']
    weight = g.vs['weight']
    edges = []
    if R == 0:
        g = nx.geographical_threshold_graph(N,theta,alpha,pos=pos,weight=weight)
    elif R >= math.sqrt(2): #R at math.sqrt(2) becomes strictly threshold graph in unit square
        #attempting some optimization. If node weight < theta/2, only loop through those with weight > theta/2
        upper_node_indxs = set([node.index for node in g.vs if node['weight'] >= theta/2])
        upper_nodes = igraph.VertexSeq(g, upper_node_indxs)
        for i in g.vs:
            if i.index in upper_node_indxs:
                for j in g.vs:
                    if j.index > i.index:
                        if threshold_edge(g,N,i,j,alpha,theta,beta):
                            edges.append([i.index,j.index])
            else:
                for j in upper_nodes:
                    if j.index > i.index:
                        if threshold_edge(g,N,i,j,alpha,theta,beta):
                            edges.append([i.index,j.index])
    else:      
        point_tree = spatial.cKDTree(pos)
        potential_edges = point_tree.query_pairs(R, 2)#dimensions = 2
        for edge in potential_edges:
            i = g.vs[edge[0]]
            j = g.vs[edge[1]]
            if threshold_edge(g,N,i,j,alpha,theta,beta):
                edges.append([i.index,j.index])
    g.add_edges(edges)                        
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
    calc_mu = lamd/(lamd+1)
    comps = g.components()
    all_comps = sorted([comp/N for comp in comps.sizes()],reverse=True)
    first_comp = all_comps[0]
    if len(all_comps) > 1:
        second_comp = all_comps[1]
    else:
        second_comp = 0
    # diameter = g.diameter()
    # total_weight = sum(weights)
    # endurance = 0 
    # for remove_count in range(100):#segment removal of removal_percent of nodes into 100 discrete instances ie. remove (removal_percent/100)*N nodes at a time
    #     remove_nodes = random.sample(range(g.vcount()),int(removal_percent/100*N))
    #     g.delete_vertices(remove_nodes)
    #     failure_g_big_weight = sum(g.components().giant().vs['weight'])
    #     endurance = endurance + (1-failure_g_big_weight/total_weight)*removal_percent/100
    return([N,lamd,R,alpha,theta,beta,K,mu,calc_mu,connectivity,first_comp,second_comp,all_comps[0:3]])

def lower_bound_R(N,R_limit,iterations,max_N,save_file):
    all_sim_data = []
    for i in range(iterations):
        if N == max_N: #only print progress for largest N (slowest loop)
            print('iteration %s' %i)
        R_GC = random.uniform(R_limit*0.999, R_limit)
        sim_parameters = [int(N),3,R_GC,0,0,0]
        sim_data = simulation(sim_parameters)
        all_sim_data.append(sim_data)
        if sim_data[9] == 1 and R_GC < R_limit:
            R_limit = R_GC
    return [N,R_limit,all_sim_data]  

def add_sim_data(data_file,new_data):
    with open(data_file, 'r') as infile:
        loaded_sim_data = json.load(infile)
    updated_dataset = loaded_sim_data + new_data
    with open(data_file, 'w') as outfile:
        json.dump(updated_dataset, outfile)
    return updated_dataset