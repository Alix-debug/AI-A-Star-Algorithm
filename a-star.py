# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 16:08:52 2022

_______________________________________________________________________________________________________________________
CECS 451 - Artificial Intelligence 
Chapter 4 : Solving Problems by Searching 



Implementation of A* pathfinding algorithm adapted to a case study using simplified California's map 


    Definition (source : Wikipedia )
    
"A* is a graph traversal and path search algorithm, 
used in many fields of computer science due to its completeness, optimality, and optimal efficiency. 
One major practical drawback is its O(b^d) space complexity, a it stores all generated nodes in memory. 
In practical travel-routing systems, it is generally outperformed by algorithms which can pre-process the graph 
to attain better performance as well as memory-bounded approaches.
However, A* is still the best solution in many cases."

________________________________________________________________________________________________________________________

@author: alixp
"""

# Import libraries
import os
import numpy as np
import pandas as pd
from collections import defaultdict


# Set constant variables
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

coordinates_file = "coordinates.txt"

map_file = "map.txt"

r = 3958.8 # Radius of the earth in miles
    
    
    
def degree_to_radian(degree):
    """
    This function returns the equivalent value of degree in radian
        :para degree (float) : latitude or longitude of a city in degree
        :return: the equivalent value in radian
    """
    return (np.pi/180)*degree

def Haversine(city1, city2) :
    """
    This function returns the straight line distance between two cities using the Haversine formula
        :para city1 (tuple) : latitude and longitude of city 1 in radian
        :para city2 (tuple) : latitude and longitude of city 2 in radian
        :return: the distance between city 1 and 2 in miles
    """    
    body = np.sin((city2[0]-city1[0])/2)**2 + np.cos(city1[0])*np.cos(city2[0])*(np.sin((city2[1]-city1[1])/2))**2 
    return 2*r*np.arcsin(np.sqrt(body))

def return_coordinates(coordinates_file):
    """
    This function returns the cities and their associated coordinates
      :para coordinates_file (string): the name of the file
      :return: a pandas dataframe that contain cities' coordinates
    """

    df = pd.read_csv(os.path.join(__location__, coordinates_file),
                     names= ['City','coordinates'], 
                     sep = ':')
    
    # df = df.assign(Latitude = lambda x: (x['coordinates'].str.strip('()').str.split(',')[0]))
    # df = df.assign(Longitude = lambda x: (x['coordinates'].str.strip('()').str.split(',')[1])
    
    df['Latitude_rad'], df['Longitude_rad']  = None, None
    
    for i in range(len(df)):
      lat, lng = map(float, df.iloc[i].coordinates.strip('()').split(','))
      df.at[i,"Latitude_rad"] = degree_to_radian(lat)
      df.at[i,"Longitude_rad"] = degree_to_radian(lng)
        
    return df.drop('coordinates', axis=1)

def Compute_haversine_distance(cityName1, cityName2):
    """
    This function returns the computation of Haversine distance between 2 cities
    :para cityName1 (string): the name of the first city
    :para cityName2 (string): the name of the second city
    :return: the Haversine distance between city1 and city2
    """
    city1 = (Coordinates[Coordinates.City == cityName1].Latitude_rad.item(),Coordinates[Coordinates.City == cityName1].Longitude_rad.item())
    city2 = (Coordinates[Coordinates.City == cityName2].Latitude_rad.item(),Coordinates[Coordinates.City == cityName2].Longitude_rad.item())
    return Haversine(city1, city2)

Coordinates = return_coordinates(coordinates_file)

def return_map(map_file):
    """
    This function returns the linked cities and their associated distances 
      :para map_file (string): the name of the file
      :return (dict): graph a dictionary that contains list of nearby cities and their relative distance for each map point
    """
    with open(os.path.join(__location__, map_file)) as f:
      map_points = [line.rstrip('\n') for line in f]
      
    graph = {}
    
    for i in range(len(map_points)):
        departure = map_points[i].split('-')[0].split(',')[0]
        graph[departure] = list()
        for i in map_points[i].split('-')[1].split(',') :
            graph[departure].append((str(i.replace(')','').split('(')[0]),float(i.replace(')','').split('(')[1])))
    
    return graph

def optimal_path(graph, parent, curr_position, start):
    """
    This function returns the linked cities and their associated distances 
      :para parent (dict): the dictionary of parent cities
      :para curr_position (string): the name of the actual position (same as the desination)
      :para start (string): the name of the start city
      :return (dict): a list that contains the optimal path to go from destination to start 
    """
    # Create the list that will collect the best options
    optimal_path = list()
    
    # Initialize the optimal distance
    optimal_dist = 0
    
    while parent[curr_position] != curr_position:
        # collect the optimal path
        optimal_path.append(curr_position)
        
        # compute the optimal distance
        for i in range(len(graph[curr_position])):
            if graph[curr_position][i][0] == parent[curr_position] :
                optimal_dist += graph[curr_position][i][1]
    
        curr_position = parent[curr_position]
    
    optimal_path.append(start)
    # reverse the path to give optimal map points from start to destination
    optimal_path.reverse()
    
    return optimal_path, optimal_dist

def find_lowest_cost(list_to_sort):
    """
    This function returns the name of the city that have the lower cost in list_to_sort
      :para list_to_sort (list): the list of tuples that contains the city point information
      :return: the name of the city with lower estimated total cost f
    """
    return sorted(list_to_sort, key=lambda x: x[1])[0][0]

def Remove_tuple(tup_lst, cityName):
    """
    This function returns the linked cities and their associated distances 
      :para tup_lst (list): the dictionary of parent cities
      :para cityName (string): the name of city point we want to remove from the list
      :return: the new tuple list
    """
    return [i for i in tup_lst if i[0] != cityName]

def AstarAlgorithm(start, destination):
    """
    Implementation of the A* pathfinding algorithm adapted to our case study (city map).
      :para start (string): the name of the start city 
      :para destination (string): the name of the desination (end point)
      :return : optimal_path (list) and optimal_dist (float), respectively the list that contains the optimal path to go from start point to destination and the optimal path value in miles.
    """
    # Initialize the map 
    graph = return_map(map_file)
    
    # Create the list to collect paths information (cost and parent city)
    open_list = list() 
    
    # Initialize the list that contains the distances driven to reach the current position
    g_cost = defaultdict(lambda: float('+inf'))
    g_cost[start] = 0
    
    f_cost = dict()
    f_cost[start] = Compute_haversine_distance(start, destination)
     
    # Initialization of the open list : the first visited map point is the start city
    open_list.append((start, f_cost[start],None))
    
    # Initialize the set of parent cities so we can find the optimal path
    parent = {}
    parent[start] = start
    
    # While all posible paths have not been visited <=> len(open_list)>0
    while len(open_list)>0 :
       
        curr_position = find_lowest_cost(open_list)
        # print('curr_position', curr_position)
        if curr_position == destination :
            return optimal_path(graph, parent, curr_position, start)
        
        # open_list.remove(curr_position)
        open_list = Remove_tuple(open_list, curr_position)
    
        
        for (neighbor, dist) in graph[curr_position]:
            
            g_cost_neighbor = g_cost[curr_position] + dist
            
            if g_cost_neighbor < g_cost[neighbor] :
                # optimal path found, save it
                parent[neighbor] = curr_position
                g_cost[neighbor] = g_cost_neighbor
                f_cost[neighbor] = g_cost_neighbor + Compute_haversine_distance(neighbor, destination)
                
                if neighbor not in open_list :
                    
                    open_list.append((neighbor,f_cost[neighbor],curr_position))
  			
  		
    return "Failure : No path found"
 

def output(city1, city2, best_route, score):
    """
    This function 
    :para city1 (string): the name of the first city
    :para city2 (string): the name of the second city
    """
    print('From city: ' + str(city1), end=('\n'))
    print('To city: ' + str(city2), end=('\n'))
    print('Best Route: ' + ' - '.join(x for x in best_route))
    print('Total distance: '+ str(score) +' mi')
    
def main():
    try:
        c1 , c2 = input().split(' ')
        if set([c1, c2]).issubset(Coordinates.City) :
            best_route = AstarAlgorithm(c1, c2)[0]            
            return output(c1, c2, best_route, AstarAlgorithm(c1, c2)[1]) 
        else : 
            print("ERROR: No such cities available\nNOTE: city 1 and city2 must appear in map.txt and coordinates.txt")  
      
    except:
        print("ERROR: Please provide valid input format :city1 city2\nNOTE :city 1 and city2 must appear in map.txt and coordinates.txt")
    
    
if __name__ == '__main__':
    main()