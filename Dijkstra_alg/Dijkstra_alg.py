import os
import sys
import time
import math

import pandas as pd
import numpy as np
from numpy.linalg import norm

import random

import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd

import networkx as nx

import folium
from folium.plugins import MousePosition
from folium import IFrame

import requests

import shapely
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry import LinearRing
from shapely.ops import unary_union
from shapely.validation import make_valid

from scipy.spatial import ConvexHull

from tqdm import tqdm

from datetime import datetime
from datetime import timedelta

import meteostat

from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans


from maps.Ocean_map import Ocean_map
from Weather_Map.Weather_map import weather_data, gpd_weather_data, extrimly_weather_data
from sub_functions.subfunctions import haversine, smoothing_path, curr_speed_and_fuel, destination_point, analys_path, nearest_data_in_weather_data

def decstra_path(
    desript_graph,
    grid,
    start_index,
    end_index,
    dist_coef = 1 / 9000,
    time_coef = 1 / 400,
    fuel_coef = 1 / 500,
    dist_degree = 1,
    time_degree = 1,
    fuel_degree = 1,
):
    # Создаем граф
    G = nx.DiGraph()
    for index1 in desript_graph:
        for index2 in desript_graph[index1]:
            speed = desript_graph[index1][index2]['speed']
            dist = desript_graph[index1][index2]['distance']
            fuel = desript_graph[index1][index2]['fuel']
            time = dist / speed
            
            cost = dist_coef * (dist ** dist_degree) + time_coef * (time ** time_degree) + fuel_coef * (fuel ** fuel_degree)
            G.add_edge(index1, index2, weight = cost)
    
    # Находим кратчайший путь
    shortest_path = nx.shortest_path(G, source=start_index, target=end_index, weight='weight')
    
    time = sum([
        desript_graph[shortest_path[i]][shortest_path[i+1]]['distance'] / desript_graph[shortest_path[i]][shortest_path[i+1]]['speed']
        for i in range(len(shortest_path) - 1)
    ])
    dist = sum([desript_graph[shortest_path[i]][shortest_path[i+1]]['distance'] for i in range(len(shortest_path) - 1)])
    fuel = sum([desript_graph[shortest_path[i]][shortest_path[i+1]]['fuel'] for i in range(len(shortest_path) - 1)])
    rang = (time + dist / 27.8 + fuel / (36 * 24)) / 3

    path = [grid[i][j] for i, j in shortest_path]

    return path, dist, fuel, time, rang

if __name__ == 'main':
    big_circle_path, grid = create_ACO_grid(
        add_distations = config['add_distations'], 
        step_km = config['step_km'], 
        north_south_distance = config['north_south_distance'], 
        north_south_step = config['north_south_step']
    )
    graph, desript_graph, start_index, end_index = create_discript_graph(
        grid = grid, 
        start_point = config['start_point'], 
        end_point = config['end_point'], 
        tao0 = config['tao0'],
        speed_degree = config['speed_degree'],
        weather_degree = config['weather_degree'],
        dist_degree = config['dist_degree'],
        dist_step = config['dist_step'],
        can_diag_move = config['can_diag_move'],
        type_work = config['type_work'],
        weather_data = config['weather_data'],
    )

    '''
    Оптимизация разных параметров
    '''
    names_functions = {
        'time' : {'dist_coef' : 0,'time_coef' : 1,'fuel_coef' : 0,
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
        'dist': {'dist_coef' : 1,'time_coef' : 0,'fuel_coef' : 0,
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
        'fuel': {'dist_coef' : 0,'time_coef' : 0,'fuel_coef' : 1,
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
        'time_dist' : {'dist_coef' : 1 / 27.8,'time_coef' : 1,'fuel_coef' : 0,
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
        'time_fuel' : {'dist_coef' : 0,'time_coef' : 1,'fuel_coef' : 1 / (36 * 24),
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
        'dist_fuel': {'dist_coef' : 1 / 27.8,'time_coef' : 0,'fuel_coef' : 1 / (36 * 24),
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
        'time_dist_fuel': {'dist_coef' : 1 / 27.8,'time_coef' : 1,'fuel_coef' : 1 / (36 * 24),
            'dist_degree' : 1,'time_degree' : 1,'fuel_degree' : 1,
        }, 
    }
       
    for name_function in names_functions:
        path, dist, fuel, time, rang = decstra_path(
            desript_graph = desript_graph,
            grid = grid,
            start_index = start_index,
            end_index = end_index,
            **names_functions[name_function],
        )
        os.makedirs('results_pathes', exist_ok=True)
        np.save(os.path.join('results_pathes', f'path_optim_{name_function}_{dist:.1f}_{time:.1f}_{fuel:.1f}_{rang:.1f}.npy'), np.array(path))
            
