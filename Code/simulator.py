import random
import math
import igraph
import networkx as nx
import numpy as np
from scipy import spatial,stats
import json
import tqdm
import powerlaw
import pickle

def generate_graph_nodes(N,lamd):
    g = igraph.Graph()
    dimensions = 2
    g.add_vertices(N)
    g.vs['pos'] = [[random.random() for dim in range(dimensions)] for node in range(N)]
    g.vs['weight'] = powerlaw.Power_Law(xmin=0.025,parameters=[lamd]).generate_random(N)
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

def sim_SC_network(g,g_NA,N,R,theta,distance_distrib,model):
    edges = []
    node_list = [v for v in g_NA.vs]
    random.shuffle(node_list)
    while g.vcount() < N and node_list:
        current_index = g.vcount()       
        rnd_node = node_list.pop()
        add_node = False
        for node in g.vs:
            dist = np.linalg.norm(np.array(node['pos'])-np.array(rnd_node['pos']))
            if model == "RGG":
                if dist <= R:
                    if add_node == False:
                        add_node = True
                        edges.append([node.index,current_index])
                    else:
                        edges.append([node.index,current_index])
            elif model == "TRGG":
                if dist <= R and node['weight'] + rnd_node['weight'] >= theta:
                    if add_node == False:
                        add_node = True
                        edges.append([node.index,current_index])
                    else:
                        edges.append([node.index,current_index])
            elif model == "SRGG":
                if dist <= R:
                    link_prob = distance_distrib.integrate_box_1d(dist-0.01,dist+0.01) #prob of linkage within +/-1% of dist
                    if link_prob >= random.random():
                        if add_node == False:
                            add_node = True
                            edges.append([node.index,current_index])
                        else:
                            edges.append([node.index,current_index])
            elif model== "GTG":
                link_prob = distance_distrib.integrate_box_1d(dist-0.005,dist+0.005) #prob of linkage within +/-1% of dist
                print(link_prob)
                if (node['weight'] + rnd_node['weight'])*link_prob >= theta*N:
                    if add_node == False:
                        add_node = True
                        edges.append([node.index,current_index])
                    else:
                        edges.append([node.index,current_index])
        if add_node == True:
            g.add_vertex(pos=rnd_node['pos'],weight=rnd_node['weight'])    
    if edges:
        g.add_edges(edges)
    return g

def generate_network(N,lamd,R,alpha,theta,beta,g=None):
    if g == None:
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
    N,lamd,R,alpha,theta,beta,load_g_file = sim_parameters
    #print ("N = %s lamda = %s R = %s alpha = %s theta = %s  beta = %s" % (N,lamd,R,alpha,theta,beta))
    if load_g_file:
        with open(load_g_file,'rb') as infile:
            loaded_g = pickle.load(infile)
        N = loaded_g.vcount()
        lamd = 0
        g = generate_network(N,lamd,R,alpha,theta,beta,loaded_g)
    else:
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