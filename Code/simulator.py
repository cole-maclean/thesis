import random
import math
import igraph
import networkx as nx
import numpy as np
from scipy import spatial,stats
import json
import tqdm
from scipy.stats import pareto, expon
import pickle

def generate_graph_nodes(N,dist_args):
    #TODO: allow for power and exp distributions
    g = igraph.Graph()
    dimensions = 2
    g.add_vertices(N)
    g.vs['pos'] = [[random.random() for dim in range(dimensions)] for node in range(N)]
    if dist_args[0] == None:
        weights = [0 for dummy in range(N)]
    elif dist_args[0] == 'pareto':
        weights = pareto.rvs(dist_args[1],size=N)
    elif dist_args[0] == 'exp':
        weights = expon.rvs(scale=(1/dist_args[1]),size=N)
    g.vs['weight'] = weights
    return g

def threshold_edge(g,N,i,j,alpha,theta,beta):
    if theta == 0:
        return True
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

def sim_SC_network(g,g_NA,N,R,theta,distance_distrib,model,nodes_only = False):
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
                    if nodes_only:
                        add_node = True
                        break
                    elif add_node == False:
                        add_node = True
                        edges.append([node.index,current_index])
                    else:
                        edges.append([node.index,current_index])
            elif model == "TRGG":
                if dist <= R and node['weight'] + rnd_node['weight'] >= theta:
                    if nodes_only:
                        add_node = True
                        break
                    elif add_node == False:
                        add_node = True
                        edges.append([node.index,current_index])
                    else:
                        edges.append([node.index,current_index])
            elif model == "SRGG":
                if dist <= R:
                    link_prob = distance_distrib.integrate_box_1d(dist-0.005,dist+0.005) #prob of linkage within +/-1% of dist
                    if link_prob >= random.random():
                        if nodes_only:
                            add_node = True
                            break
                        elif add_node == False:
                            add_node = True
                            edges.append([node.index,current_index])
                        else:
                            edges.append([node.index,current_index])
            elif model== "GTG":
                if (node['weight'] + rnd_node['weight']) >= theta:
                    link_prob = distance_distrib.integrate_box_1d(dist-0.005,dist+0.005) #prob of linkage within +/-1% of dist
                    if (node['weight'] + rnd_node['weight'])*link_prob >= theta:
                        if nodes_only:
                            add_node = True
                            break
                        elif add_node == False:
                            add_node = True
                            edges.append([node.index,current_index])
                        else:
                            edges.append([node.index,current_index])
        if add_node == True:
            g.add_vertex(pos=rnd_node['pos'],weight=rnd_node['weight'],gh=rnd_node['gh'])    
    if edges:
        g.add_edges(edges)
    return g

def generate_network(N,dist_args,R,alpha,theta,beta,g=None):
    if g == None:
        g = generate_graph_nodes(N,dist_args)
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
    N,dist_args,R,alpha,theta,beta,load_g_file = sim_parameters
    if load_g_file:
        with open(load_g_file,'rb') as infile:
            loaded_g = pickle.load(infile)
        N = loaded_g.vcount()
        g = generate_network(N,dist_args,R,alpha,theta,beta,loaded_g)
    else:
        g = generate_network(N,dist_args,R,alpha,theta,beta)
    K = g.ecount()
    connectivity = 2*K/N
    weights = g.vs['weight']
    mu = np.mean(weights)
    comps = g.components()
    all_comps = sorted([comp/N for comp in comps.sizes()],reverse=True)
    first_comp = all_comps[0]
    if len(all_comps) > 1:
        second_comp = all_comps[1]
    else:
        second_comp = 0
    return([N,R,alpha,theta,beta,K,mu,connectivity,first_comp,second_comp,all_comps[0:3],*dist_args])

def sim_SC_net(sim_parameters):
    N,R,theta,model = sim_parameters
    with open('SC_cities.pkl','rb') as infile:
        loaded_SC = pickle.load(infile)
    with open('NA_cities.pkl','rb') as infile:
        NA_g = pickle.load(infile)
    with open("distance_distrib.pkl","rb") as infile:
        est_dist = pickle.load(infile)
    first_SC = loaded_SC.vs[0]
    seed_g = igraph.Graph()
    seed_g.add_vertex(pos=first_SC['pos'],weight=first_SC['weight'],gh=first_SC['gh'])
    result_net = sim_SC_network(seed_g,NA_g,N,R,theta,est_dist,model,True)
    nodes = [node['gh'] for node in result_net.vs]
    return [model,nodes]

def lower_bound_R(N,R_limit,unchange_limit,max_N,save_file):
    unchange_count = 0
    while unchange_count < unchange_limit:
        R_GC = random.uniform(R_limit*0.999, R_limit)
        sim_parameters = [int(N),[None],R_GC,0,0,0,None]
        sim_data = simulation(sim_parameters)
        if sim_data[8] == 1 and R_GC < R_limit:
            print("Update count for N = %s reset at i = %s" %(N,unchange_count)) 
            R_limit = R_GC
            unchange_count = 0
        else:
            unchange_count = unchange_count + 1
    print ("New lower bound for N = %s = %s" %(N,R_limit))
    return [N,R_limit]  

def add_sim_data(data_file,new_data):
    with open(data_file, 'r') as infile:
        loaded_sim_data = json.load(infile)
    updated_dataset = loaded_sim_data + new_data
    with open(data_file, 'w') as outfile:
        json.dump(updated_dataset, outfile)
    return updated_dataset