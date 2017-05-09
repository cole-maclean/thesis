MAX_RANGE = 346 #max range of base class Tesla Model 3 in kms
MAJOR_CITY_POP = 15000 #population threshold for major city definition
PENETRATION_GH_PRECISION = 4 #base precision of penetration definition

import networkx as nx
import geo_tools
from operator import itemgetter
import math
import geohash
import numpy as np
import os
import pickle

POP_DICT = geo_tools.load_pop_dict()
#generate list of unique geohash "squares" of size dictated by GH_PRECISION and use this to build dict of populations grouped by the unique geohash "squares"
sub_ghs = list(set([gh[0:PENETRATION_GH_PRECISION] for gh in list(POP_DICT.keys())]))
sub_pop_dict = {sub_gh:sum([data['population'] for gh,data in POP_DICT.items() if gh[0:PENETRATION_GH_PRECISION] == sub_gh]) for sub_gh in sub_ghs}
tot_pop = sum(list(sub_pop_dict.values()))
major_cities = [city_gh for city_gh,data in POP_DICT.items() if data['population'] >= MAJOR_CITY_POP]
major_cities.remove('total')

#extend networkx Graph class with customized methods for SCNetwork class
class SCNetwork(nx.Graph):
    def __init__(self,network_data=None):
        nx.Graph.__init__(self,network_data)
        self.expansion_cache = {}
        self.expansion_city_ghs =[]
        self.newest_close_cities = [0]
        self.penetrated_sub_ghs = []
        self.util_params = [-1]
        self.util_attrbs = ["efficiency","connectivity"]
        self.connected_network = self.copy()

    #Network Metadata
    def newest_node(self):
        return max(self.nodes(),key=lambda n: int(self.node[n]['SC_index']))

    def reverse_node_lookup(self,SC_indexes):
        nodes = []
        for node,data in self.nodes_iter(data=True):
            if data["SC_index"] in SC_indexes:
                nodes.append(node)
        return nodes 

        #generate all possible sub_graphs of a SC network by building each stage of the network node by node
    def all_sub_graphs(self):
        return ([SCNetwork(self.subgraph([node for node,data in self.nodes_iter(data=True)
                                   if int(data["SC_index"]) <= node_count + 1]))
                     for node_count in range(self.number_of_nodes())])

    def add_SC(self,node_gh,util_params = []):
        if node_gh not in self.nodes():
            node_data = {}
            if len(self.nodes()) == 0:
                node_data['SC_index'] = 0
            else:
                node_data['SC_index'] = int(self.node[self.newest_node()]["SC_index"]) + 1
            node_data['GPS'] = geohash.decode(node_gh)
            node_data['lat'] = node_data['GPS'][0]
            node_data['lon'] = node_data['GPS'][1]
            node_data['GPS_lon_lat'] = [node_data['lon'],node_data['lat']]
            node_data['population'] = self.SC_population(node_gh)
            node_data['geohash'] = node_gh
            if util_params:
                node_data['util_params'] = {self.util_attrbs[i]:util_params[i] for i in range(len(self.util_attrbs))}
            self.add_node(node_gh,{key:node_data[key] for key in node_data.keys()})
            self.add_connections(node_gh)
        else:
            print ("node already in network")
        return self

    def add_connections(self,src_hash):
        connections = {}
        node_hashes = geo_tools.get_close_ghs(src_hash,self.nodes(),3,2,MAX_RANGE)
        for connected_node in node_hashes:
            if connected_node != src_hash:
                try:
                    self.add_edge(src_hash,connected_node,{key:data for key,data in self.connected_network[src_hash][connected_node].items()})
                    self[src_hash][connected_node]['first_node'] = str(min(int(self.node[src_hash]["SC_index"]), #order of connection 
                                                                            int(self.node[connected_node]["SC_index"])))
                    self[src_hash][connected_node]['second_node'] = str(max(int(self.node[src_hash]["SC_index"]), #order of connection 
                                                                            int(self.node[connected_node]["SC_index"])))
                except KeyError as e:
                    src_GPS = geo_tools.reverse_GPS(geohash.decode(src_hash))
                    connection = {}
                    connection['node'] = connected_node
                    connection['directions'] = geo_tools.get_geohash_directions(src_hash,connected_node)
                    if connection['directions']['distance']/1000 <= MAX_RANGE:
                        edge_weight = self.get_edge_weight(src_hash,connection['node'])
                        self.add_edge(src_hash,connection['node'],{'weight':edge_weight,'distance':connection['directions']['distance'],
                                                                    'steps':connection['directions']['steps'],
                                                                    'first_node':str(min(int(self.node[src_hash]["SC_index"]), #order of connection 
                                                                    int(self.node[connection['node']]["SC_index"]))),
                                                                    'second_node':str(max(int(self.node[src_hash]["SC_index"]), #order of connection 
                                                                    int(self.node[connection['node']]["SC_index"]))),
                                                                    'lon_lat_1':geo_tools.reverse_GPS(geohash.decode(src_hash)),
                                                                    'lon_lat_2':geo_tools.reverse_GPS(geohash.decode(connection['node']))})
                        self.connected_network.add_edge(src_hash,connection['node'],{key:data for key,data in self[src_hash][connection['node']].items()})
        return self

    def get_edge_weight(self,src_hash,connection_hash):
        try:
            pop1 = self.node[src_hash]['population']
            pop2 = self.node[connection_hash]['population']
            return (pop1+pop2)/POP_DICT['total']['population']
        except KeyError as e:
            print(e)
            return 0 

    #population/geographic tools
    def SC_population(self,node_gh):#function uses geohash precision of 3 (ie radius of 73km) and sums population within this radius
        total_close_pop = (sum([data['population'] for gh,data in POP_DICT.items()
                    if gh[0:3] in geohash.expand(node_gh[0:3])]))
        return total_close_pop

    #custom defined graph attributes
    def geo_area(self):
        node_GPS_list = [(data["lat"],data["lon"]) for node,data in self.nodes_iter(data=True)]
        sort_lat_GPS = sorted(node_GPS_list,key=itemgetter(0))
        sort_lon_GPS = sorted(node_GPS_list,key=itemgetter(1))
        NS_dist = geo_tools.haversine(-100,sort_lat_GPS[0][0],-100,sort_lat_GPS[-1][0])
        WE_dist = geo_tools.haversine(sort_lon_GPS[0][1],50,sort_lon_GPS[-1][1],50)
        return NS_dist*WE_dist*math.pi

    def largest_subcomponent(self):
        return SCNetwork(max(nx.connected_component_subgraphs(self), key=len))

            #calculate the unique sum total populatition of a network by developing the set of represented geohashes in the network and performing a lookup in the gh_pop_dict
    #for the respective population, normalized by the total population in the gh_pop_dict
    def penetration(self):
        G_sub_ghs = list(set([gh[0:PENETRATION_GH_PRECISION] for gh in self.nodes_iter()]))
        self.penetrated_sub_ghs = list(set([gh for sub_gh in G_sub_ghs for gh in geo_tools.gh_expansion(sub_gh[0:PENETRATION_GH_PRECISION],3)]))
        tot_graph_pop = sum([sub_pop_dict[sub_gh] for sub_gh in self.penetrated_sub_ghs if sub_gh in sub_ghs])
        return tot_graph_pop/tot_pop

    def connectivity(self):
        max_sg = self.largest_subcomponent()#get the maximum sub_graph component
        return max_sg.penetration()

    def robustness(self):
        max_sg = self.largest_subcomponent()#get the maximum sub_graph component
        return nx.node_connectivity(max_sg)

    def efficiency(self):
        max_sg = self.largest_subcomponent()
        if len(max_sg) > 1:
            return (math.sqrt(max_sg.geo_area()/math.pi))/nx.average_shortest_path_length(max_sg)/(MAX_RANGE*3) #Efficieny normalized by the theoretical max efficiency of traveling 3x the max range with 2 SC's
        return 0

    def breadth(self):
        max_sg = self.largest_subcomponent()#get the maximum sub_graph component
        return max_sg.geo_area()/(2762*6425*math.pi) #breadth normalized by maximum theoretical network breadth using extreme North America geographical points

    def density(self):
        max_sg = self.largest_subcomponent()#get the maximum sub_graph component
        G_area = max_sg.geo_area()
        if G_area ==0:
            return 0
        return (len(max_sg)/G_area)*100 #denisty normalized to 1 SC for every 100km^2 in the network

    def SC_expansion_search(self):
        if self.penetrated_sub_ghs == []:
            self.penetration()
        unadded_major_cities = [city_gh for city_gh in major_cities if city_gh not in self.nodes()]
        unpen_cities = [city_gh for city_gh in unadded_major_cities if city_gh[0:PENETRATION_GH_PRECISION] not in self.penetrated_sub_ghs]
        if self.newest_close_cities != [0]:
            new_exp_cities = [gh for gh in unpen_cities if gh not in self.expansion_city_ghs]
            self.newest_close_cities = geo_tools.get_close_ghs(self.newest_node(),new_exp_cities,3,2,MAX_RANGE)
        else:
            close_net_cities = []
            for node in self:
                close_net_cities = close_net_cities + geo_tools.get_close_ghs(node,unpen_cities,3,2,MAX_RANGE)
            self.newest_close_cities = [city_gh for city_gh in list(set(close_net_cities)) if city_gh not in self.penetrated_sub_ghs or self.nodes()]
        exp_cities = set(self.expansion_city_ghs + self.newest_close_cities)
        self.expansion_city_ghs = [city_gh for city_gh in exp_cities if city_gh in unpen_cities]
        print ("expansion search cities = " + str(len(self.expansion_city_ghs)))
        return self.expansion_city_ghs

    def calc_utilities(self,node,util_params):
        #store overall utility values of current network
        newest_node = self.newest_node()
        cur_pen = self.penetration()
        cur_eff = self.efficiency()
        cur_con = self.connectivity()
        #cur_breadth = self.breadth()
        #cur_dens = self.density()

        self.add_SC(node)
        self.expansion_cache[node] = []
        self.expansion_cache[node].append(self.efficiency() - cur_eff)
        try:
            self.expansion_cache[node].append(-math.log10((self.connectivity() - cur_con)))
        except ValueError:
            self.expansion_cache[node].append(0)
        #self.expansion_cache[node].append(self.breadth() - cur_breadth)
        #self.expansion_cache[node].append(self.density() - cur_dens)
        trans_util = self.clasif_scaler.transform(np.array(self.expansion_cache[node]).reshape(1,-1))
        if len(self[node].keys()) == 0: #set utility to -100 if node did not make any connections
            self.expansion_cache[node].append(-100)
            node_utility = -100
        elif int(self.clasif_model.predict(trans_util)) == 0:
            self.expansion_cache[node].append(0)
            node_utility = 0
        else:
            node_utility = trans_util.flatten()*np.array(util_params)
            self.expansion_cache[node].append(sum(node_utility))
        self.remove_node(node)
        return node_utility

    def expansion_utilities(self,util_params):
        if list(self.util_params) != list(util_params):#recalc utils with new params
            for node in self.expansion_cache:
                node_utility = np.array(self.expansion_cache[node].pop())*np.array(util_params)
                self.expansion_cache[node].append(sum(node_utility))
            print ("util params changed from " + str(self.util_params) + " to " + str(util_params))
            self.util_params = util_params
        expansion_nodes = self.SC_expansion_search()
        update_space = geo_tools.gh_expansion(self.newest_node()[0:3],2)
        for node in expansion_nodes:
            if node not in list(self.expansion_cache.keys()):
                util = self.calc_utilities(node,util_params)
            elif node[0:3] in update_space:
                trans_util = self.clasif_scaler.transform(np.array(self.expansion_cache[node][0:2]).reshape(1,-1))
                if int(self.clasif_model.predict(trans_util)) == 1:
                    util = self.calc_utilities(node,util_params)
        return self.expansion_cache
