from math import radians, cos, sin, asin, sqrt,inf,ceil
import json
import configparser
import googlemaps
import time
import geohash
import os
import fasteners

config = configparser.ConfigParser()
config.read("config.ini")
API_key = config['Keys']['google_API']
gmaps = googlemaps.Client(key=API_key)

#Resolution errors in Km lookup table based on geohash string length (ie geohash of length 1 has +/- error if 2500km 
#source=https://en.wikipedia.org/wiki/Geohash)
gh_resolution_lookup = [inf,2500,630,78,20,2.4,0.61,0.076,0.019] 
def load_pop_dict():
    with open('populations.json', 'r') as f:
        p_dict = json.load(f)
    return p_dict

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km
#reverse lon/lat positions of a GPS tuple
def reverse_GPS(GPS):
    return [GPS[1],GPS[0]]


def get_geohash_directions(gh_A,gh_B):
    try:
        with open("google_directions_cache.json","r") as f:
            google_directions_cache = json.load(f)
    except ValueError:
        GPS_A = geohash.decode(gh_A)
        GPS_B = geohash.decode(gh_B) 
        directions_result = gmaps.directions(GPS_A,
                                             GPS_B,
                                             mode="driving")
        connection_data =({'distance':directions_result[0]['legs'][0]['distance']['value'],
                           'steps':len(directions_result[0]['legs'][0]['steps'])})
        time.sleep(1)
        print ("failed cache read")
        return connection_data
    sorted_hashes = sorted([gh_A,gh_B])
    connection_key = sorted_hashes[0] + sorted_hashes[1]
    if connection_key in list(google_directions_cache.keys()):
        connection_data = google_directions_cache[connection_key]
    else:
        GPS_A = geohash.decode(gh_A)
        GPS_B = geohash.decode(gh_B) 
        directions_result = gmaps.directions(GPS_A,
                                             GPS_B,
                                             mode="driving")
        connection_data =({'distance':directions_result[0]['legs'][0]['distance']['value'],
                           'steps':len(directions_result[0]['legs'][0]['steps'])})
        google_directions_cache[connection_key] = connection_data
        with open("google_directions_cache.json","w") as f:
                json.dump(google_directions_cache,f)
        time.sleep(1)
    return connection_data

def gh_expansion(seed_gh,exp_iters):
    expansion_ghs = {0:[seed_gh]}
    ghs = []
    for i in range(1,exp_iters+1):
        expansion_ghs[i] = []
        for gh in expansion_ghs[i-1]:
            expansion_ghs[i] = expansion_ghs[i] + geohash.expand(gh)
            ghs = ghs + geohash.expand(gh)
    return list(set(ghs))

def get_best_expansion(distance,max_expansions=3,precision=8):
    if precision > 0:
        required_expansions = ceil(distance/(2*gh_resolution_lookup[precision]))
        if required_expansions <= max_expansions:
            return precision, required_expansions
        else:
            return get_best_expansion(distance,max_expansions,precision-1)
    else:
        return 0,0
def get_close_ghs(src_hash,lookup_hash_list,max_distance):
    '''This function takes in a geohash and searches for close geohashes in the supplied geohash lookup list, but filtering out
       geohashes exceeding the max_distance parameter. For computational efficiency, the list is first filtered using geohash expansions
       where the resolution is selected using the best expansion precision that requires less then max_expansions (default = 3). This resolution is then expanded to the
       max distance, and geohashes not in this expansion list are filtered out. The remaining geohashes are finetuned by calculating the haverstine
       distance of the remaining geohashes, and filtering out any exceeding the max_distance parameter. '''

    resolution,expansions = get_best_expansion(max_distance)
    exp_src_hash = gh_expansion(src_hash[0:resolution],expansions)
    return  [gh for gh in lookup_hash_list
                    if gh[0:resolution] in exp_src_hash
                    and haversine(*reverse_GPS(geohash.decode(src_hash)),*reverse_GPS(geohash.decode(gh))) <= max_distance]

def get_gh_city(gh):
    with open("google_geocity_cache.json","r") as f:
            google_geocity_cache = json.load(f)
    if gh in list(google_geocity_cache.keys()):
         city = google_geocity_cache[gh]
    else:
        location = gmaps.reverse_geocode(geohash.decode(gh))
        city = location[0]["formatted_address"].split(",")[1]
        time.sleep(1)
        google_geocity_cache[gh] = city
        with open("google_geocity_cache.json","w") as f:
                json.dump(google_geocity_cache,f)
        time.sleep(1)
    return city