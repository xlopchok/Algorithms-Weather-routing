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

from collections import deque

from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans

from maps.Ocean_map import Ocean_map
from Weather_Map.Weather_map import weather_data, gpd_weather_data, extrimly_weather_data
from sub_functions.subfunctions import haversine, smoothing_path, curr_speed_and_fuel, destination_point, analys_path, nearest_data_in_weather_data

config = {
    'start_point' : Point((3, 55)),
    'end_point' : Point((-80, 15)),
    'Ocean_map' : Ocean_map,
    'add_distance' : 900,
    'step_km' : 150,
    'north_south_distance' : 2100,
    'north_south_step' : 150,
    'tao0' : 1,
    'speed_degree' : 1.5,
    'weather_degree' : 2,
    'dist_degree' : 1.2,
    'dist_step' : 1,
    'can_diag_move' : True,
    'type_work' : 'const_fuel',
    'weather_data' : weather_data,
    'time_step_isohrons' : 24,
    'rendering_grid_points' : False,
}

def BFS_path(
    start_index = start_index,
    end_index = end_index,
    desript_graph = desript_graph,
    dist_coef = 1 / 27.8, 
    time_coef = 1, 
    fuel_coef = 1 / (36 * 24),
    dist_degree = 1, 
    time_degree = 1,
    fuel_degree = 1,
    verbose = -1,
):
    '''
    Функция анализирует граф, и ищет путь до каждой точки используя алгоритм BFS
    '''
    def comparison_path(curr_dist, min_dist, curr_time, min_time, curr_fuel, min_fuel,
        dist_coef, time_coef, fuel_coef, dist_degree, time_degree, fuel_degree):

        res = (dist_coef * (curr_dist ** dist_degree) + time_coef * (curr_time ** time_degree) + fuel_coef * (curr_fuel ** fuel_degree) < 
              dist_coef * (min_dist ** dist_degree) + time_coef * (min_time ** time_degree) + fuel_coef * (min_fuel ** fuel_degree))
        return res
    
    predict_dist_time_fuel = {
        start_index : {
            'time' : 0,
            'dist' : 0,  
            'fuel' : 0,
            'parent' : (-1, -1)
        }
    }
    
    queue = deque()
    queue.append(start_index) 
    
    count_points = 0
    while queue:
        # Берем последний доваленный в стек элемент
        curr_point = queue.popleft()
        
        # Анализируем его соседей
        for next_index in desript_graph[curr_point].keys():
            if next_index in predict_dist_time_fuel:
                continue
            min_dist = float('inf')
            min_time = float('inf')
            min_fuel = float('inf')
            best_speed = 0
            best_neighbour = (-1, -1)
    
            # Проверим всех соседений этой следующей точки, чтобы выбрать оптимальный путь дойти до нее
            for neighbour in desript_graph[next_index].keys():
                # Если сосед еще не обработан то пропускаем
                if neighbour not in predict_dist_time_fuel or next_index not in desript_graph[neighbour]:
                    continue
                    
                curr_dist = predict_dist_time_fuel[neighbour]['dist'] + desript_graph[neighbour][next_index]['distance']
                curr_time = predict_dist_time_fuel[neighbour]['time'] + desript_graph[neighbour][next_index]['distance'] / desript_graph[neighbour][next_index]['speed']
                curr_fuel = predict_dist_time_fuel[neighbour]['fuel'] + desript_graph[neighbour][next_index]['fuel']
                mean_speed = curr_dist / curr_time
    
                if comparison_path(curr_dist, min_dist, curr_time, min_time, curr_fuel, min_fuel,
                                            dist_coef, time_coef, fuel_coef, dist_degree, time_degree, fuel_degree):
                    best_speed = mean_speed
                    min_dist = curr_dist
                    min_time = curr_time
                    min_fuel = curr_fuel
                    best_neighbour = neighbour
                    
            predict_dist_time_fuel[next_index] = {
                'time' : min_time,
                'dist' : min_dist,  
                'fuel' : min_fuel,
                'parent' : best_neighbour
            }
            queue.append(next_index)
            count_points += 1
    
    path = [end_index]
    curr_index = path[-1]
    
    while predict_dist_time_fuel[curr_index]['parent'] != (-1, -1):
        curr_index = path[-1]
        next_index = predict_dist_time_fuel[curr_index]['parent']
        path.append(next_index)
    
    path = path[::-1]
    path = path[1:]
    
    return path, predict_dist_time_fuel


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
    '''
    Функция анализирует граф, и ищет путь до каждой точки используя алгоритм Дейкстры
    '''
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
    shortest_path = nx.dijkstra_path(G, source=start_index, target=end_index, weight='weight')
    
    time = sum([
        desript_graph[shortest_path[i]][shortest_path[i+1]]['distance'] / desript_graph[shortest_path[i]][shortest_path[i+1]]['speed']
        for i in range(len(shortest_path) - 1)
    ])
    dist = sum([desript_graph[shortest_path[i]][shortest_path[i+1]]['distance'] for i in range(len(shortest_path) - 1)])
    fuel = sum([desript_graph[shortest_path[i]][shortest_path[i+1]]['fuel'] for i in range(len(shortest_path) - 1)])
    rang = (time + dist / 27.8 + fuel / (36 * 24)) / 3
    
    path = [grid[i][j] for i, j in shortest_path]
    
    return path, dist, fuel, time, rang

def BFS_based_isochrone_rendering(
    Ocean_map,
    grid,
    predict_dist_time_fuel,
    start_point, 
    end_point,
    time_step = 24,
    rendering_grid_points = False,
    map_path = 'isohrons_based_BFS.html'
):
    '''
    На основе анализа графа с помощью алгоритма BFS строятся изохроны

    time_step - временной промежутов для каждой изохроны
    '''
    muliti_polygons_time = {}
    for index in predict_dist_time_fuel:
        time = predict_dist_time_fuel[index]['time'] // time_step + 1
        if type(time) != np.float64 and type(time) != int:
            continue
        point = grid[index[0]][index[1]]
    
        time = int(time)
        if time not in muliti_polygons_time:
            muliti_polygons_time[time] = []
    
        muliti_polygons_time[time].append(point)

    for day in muliti_polygons_time:
        muliti_polygons_time[day] = MultiPoint(muliti_polygons_time[day]).convex_hull
        if not muliti_polygons_time[day].is_valid:
            muliti_polygons_time[day] = make_valid(muliti_polygons_time[day])
        muliti_polygons_time[day] = muliti_polygons_time[day].intersection(Ocean_map)
        
    if rendering_grid_points:
        grid_points = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                grid_points.append(grid[i][j])

    m = folium.Map(tiles="cartodbpositron")
    group_1 = folium.FeatureGroup("first group").add_to(m)
    folium.GeoJson(Ocean_map).add_to(group_1)
    
    Color = ['green', 'red', 'yellow', 'purple', 'orange', 'blue', 'pink']
    
    folium.Marker((start_point.coords[0][1], start_point.coords[0][0]), tooltip="start_point").add_to(m)
    folium.Marker((end_point.coords[0][1], end_point.coords[0][0]), tooltip="end_point").add_to(m)
    
    min_day = min(list(muliti_polygons_time.keys()))
    max_day = max(list(muliti_polygons_time.keys()))
    for day in range(min_day, max_day + 1):
        isohron = muliti_polygons_time[day]
        if not isohron.is_valid:
            isohron = make_valid(isohron)
            
        isohron = isohron.intersection(Ocean_map)
        if not isohron.is_valid:
            isohron = make_valid(isohron)
            
        if day == 1:
            all_last_isohrones = isohron
        else:
            isohron = isohron.difference(all_last_isohrones)
            if not isohron.is_valid:
                isohron = make_valid(isohron)
            all_last_isohrones = all_last_isohrones.union(isohron)
            if not all_last_isohrones.is_valid:
                all_last_isohrones = make_valid(all_last_isohrones)
        folium.GeoJson(isohron, tooltip=f"{day}", name = f'step{day}', color = Color[day % len(Color)], weight = 5).add_to(m)

    if rendering_grid_points:
        folium.GeoJson(
            MultiPoint(grid_points), tooltip= f"point {1}", 
            marker=folium.Circle(radius=12, fill_color="orange", fill_opacity=0.4, color="black", weight=4), 
            name = f'point_of_isohrones{step}', color = 'red', weight = 4,
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)
    
    os.makedirs('results_visual', exist_ok=True)
    if rendering_grid_points:
        m.save(os.path.join('results_visual', 'grid_points_' + map_path))
    else:
        m.save(os.path.join('results_visual', map_path))


def visual_path(
    start_point, 
    end_point, 
    Ocean_map,
    path,
    color = 'purple',
    map_path = 'path.html',
):
    '''
    Визуализаця полученных маршрутов
    '''
    m = folium.Map(tiles="cartodbpositron")
    group_1 = folium.FeatureGroup("first group").add_to(m)
    folium.GeoJson(Ocean_map).add_to(group_1)
    
    folium.Marker((start_point.coords[0][1], start_point.coords[0][0]), tooltip="start_point").add_to(m)
    folium.Marker((end_point.coords[0][1], end_point.coords[0][0]), tooltip="end_point").add_to(m)
    
    folium.PolyLine([point[::-1] for point in path], tooltip="ant_path", color = color, weight = 5).add_to(m)
    MousePosition().add_to(m)

    os.makedirs('results_visual', exist_ok=True)
    m.save(os.path.join('results_visual', map_path))

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
        # Визуализация маршрута полученного в результате работы алгоритма Дейкстры
        visual_path(
            start_point = config['start_point'], 
            end_point = config['end_point'],
            Ocean_map = config['Ocean_map'],
            path = path,
            map_path = f'{name_function}_decstra_path.html'
        )
        
        os.makedirs('results_pathes', exist_ok=True)
        np.save(os.path.join('results_pathes', f'decstra_path_optim_{name_function}_{dist:.1f}_{time:.1f}_{fuel:.1f}_{rang:.1f}.npy'), np.array(path))

        path, predict_dist_time_fuel = BFS_path(
            start_index = start_index,
            end_index = end_index,
            desript_graph = desript_graph,
            **names_functions[name_function]
        )
        # Визуализация маршрута в результате работы алгоритма BFS
        visual_path(
            start_point = config['start_point'], 
            end_point = config['end_point'],
            Ocean_map = config['Ocean_map'],
            path = path,
            map_path = f'{name_function}_BFS_path.html'
        )
        
        time = predict_dist_time_fuel[end_index]['time']
        dist = predict_dist_time_fuel[end_index]['dist']
        fuel = predict_dist_time_fuel[end_index]['fuel']
        os.makedirs('results_pathes', exist_ok=True)
        np.save(os.path.join('results_pathes', f'BFS_path_optim_{name_function}_{dist:.1f}_{time:.1f}_{fuel:.1f}.npy'), np.array(path))

        # Визуализация изохрон
        BFS_based_isochrone_rendering(
            Ocean_map = config['Ocean_map'],
            grid = grid,
            predict_dist_time_fuel = predict_dist_time_fuel,
            start_point = config['start_point'], 
            end_point = config['end_point'], 
            time_step = config['time_step_isohrons'],
            rendering_grid_points = config['rendering_grid_points'],
            map_path = f'isohrons_based_BFS_{name_function}.html'
        ):
