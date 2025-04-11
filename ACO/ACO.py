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

config = {
    'start_point' : Point((3, 55)),
    'end_point' : Point((-80, 15)),
    'Ocean_map' : Ocean_map,
    'add_distations' : 900,
    'step_km' : 150,
    'north_south_distance' : 2100,
    'north_south_step' : 150,
    'tao0' : 1,
    'speed_degree' : 1.5, 
    'weather_degree' : 2,
    'dist_degree' : 1.2,
    'can_diag_move' : True,
    'type_work' : 'const_fuel',
    'weather_data' : weather_data,
    'count_try_get_path' : 15000,
    'max_count_points_in_path' : 1500, 
    'alpha' : 1.75,
    'beta' : 5,
    'min_feromon' : 1,
    'max_feromon' : 75,
    'ro_start' :  0.15,
    'ro_end' : 0.6, 
    'count_path_for_epoch' : 250,
    'start_count_path_for_epoch' : 5,
    'start_epochs' " 50,
    'epochs' : 1050,
    'Q' : 5000,
    'count_epoch_without_update' : 100,
    'all_pathes' : False,
    'dist_degree' : 2,
}

class ACO():
    def __init__(
        self, 
        grid,
        desript_graph, 
        start_index, 
        end_index,
        count_try_get_path = 20000,
        max_count_points_in_path = 1500, 
        alpha = 1.75,
        beta = 5,
        min_feromon = 0.01,
        max_feromon = 50,
        ro_start = 0.5,
        ro_end = 0.8, 
        
    ):
        self.grid = grid
        self.desript_graph = desript_graph
        self.start_index = start_index
        self.end_index = end_index
        self.count_try_get_path = count_try_get_path
        self.max_count_points_in_path = max_count_points_in_path
        self.alpha = alpha
        self.beta = beta
        self.min_feromon = min_feromon
        self.max_feromon = max_feromon
        self.ro_start = ro_start
        self.ro_end = ro_end

        self.best_path = None
      
    def create_path_for_one_ant(self):
        path_one_ant = []
        for curr_iter in range(self.count_try_get_path):
            path_one_ant.clear()
            path_one_ant = [self.start_index]
            for _ in range(self.max_count_points_in_path):
                curr_pos = path_one_ant[-1]
                next_posses = [pos for pos in self.desript_graph[path_one_ant[-1]].keys() if pos not in path_one_ant]
                if len(next_posses) == 0:
                    break
    
                feromons = [
                    self.desript_graph[curr_pos][next_pos]['feromon']
                    for next_pos in next_posses
                ]
                
                costs = [
                    self.desript_graph[curr_pos][next_pos]['cost']
                    for next_pos in next_posses
                ]

                probabilities = [
                    (self.desript_graph[curr_pos][next_pos]['feromon']**self.alpha) * (self.desript_graph[curr_pos][next_pos]['cost']**self.beta)
                    for next_pos in next_posses
                ]
                
                next_pos = np.random.choice(range(len(next_posses)), p=np.array(probabilities) / np.sum(probabilities))
                path_one_ant.append(next_posses[next_pos])
                
                if path_one_ant[-1] == self.end_index:
                    return path_one_ant
        return path_one_ant

    # Функция подсчета характеристик полученного маршрута
    def dist_time_meanBF_rang_path(self, path_one_ant):
        time = sum([
            self.desript_graph[path_one_ant[i]][path_one_ant[i+1]]['distance'] / self.desript_graph[path_one_ant[i]][path_one_ant[i+1]]['speed']
            for i in range(len(path_one_ant) - 1)
        ])
        dist = sum([self.desript_graph[path_one_ant[i]][path_one_ant[i+1]]['distance'] for i in range(len(path_one_ant) - 1)])
        fuel = sum([self.desript_graph[path_one_ant[i]][path_one_ant[i+1]]['fuel'] for i in range(len(path_one_ant) - 1)])
        mean_BF = np.mean([self.desript_graph[path_one_ant[i]][path_one_ant[i+1]]['BF'] for i in range(len(path_one_ant) - 1)])
        rang_path = 1000 / (np.log(1 + dist) * np.log(1 + time) * np.log(1 + fuel) * mean_BF**0.5 * (len(path_one_ant) / 100))
        
        return dist, time, mean_BF, fuel, rang_path
    
    def setup_graph_and_create_path(
        self,
        count_path_for_epoch = 40,
        start_count_path_for_epoch = 5,
        start_epochs = 10,
        epochs = 20,
        Q = 500,
        count_epoch_without_update = 50,
        all_pathes = False,
        dist_degree = 2,
    ):
        epoch_without_updates = count_epoch_without_update

        # Будем сохранять лучший путь         dist          time            fuel      rang
        self.best_path = {'path' : [], 'rang' : [float('inf'), float('inf') ,float('inf'), 0]}
        pathes = []
    
        pbar = tqdm(range(epochs + start_epochs), desc = f'')
        # Пройдемся по эпохам
        for epoch in pbar:
            # В каждой эпохе будем создавать несколько путей
            pathes.clear()
            
            if epoch < start_epochs:
                count_path = start_count_path_for_epoch + epoch + 1
            else:
                count_path = count_path_for_epoch
                
            for _ in range(count_path):
                path_one_ant = self.create_path_for_one_ant()
                if not all_pathes and path_one_ant[-1] != end_index:
                    continue
                pathes.append(path_one_ant)
                
            # Добавим описание
            dist = self.best_path['rang'][0]
            time = self.best_path['rang'][1]
            fuel = self.best_path['rang'][2]
            rang_path = self.best_path['rang'][3]
            pbar.set_description(f'epoch: {epoch + 1}, dist: {dist:.3f}, time : {time:.3f}, fuel : {fuel:.3f}, rang : {rang_path:.3f}, count_pathes : {len(pathes)}')
    
            # Проверим что есть хотя бы один путь
            if len(pathes) == 0:
                continue
    
            # Удалим лишние точки в маршрутах, если это возможно, чтобы избежать зацикливаний, 
            # Будем брать индекс и проверять нет ли его сосеей в дальнейшем пути на рсстоянии более 1 (так как на расстоянии 1 есть : сосед следующая точка)
            for path_idx, path in enumerate(pathes):
                new_path = []
                curr_index = 0
                # Проверим чтобы не выходить за пределы
                while curr_index < len(path):
                    # Добавим текущую точку
                    new_path.append(path[curr_index])
                    # Проверим ее соседей
                    max_dist, best_index = -1, -1
                    for next_indexes in self.desript_graph[path[curr_index]].keys():
                        if next_indexes in path and path.index(next_indexes) - curr_index > 1:
                            if path.index(next_indexes) - curr_index > max_dist:
                                max_dist = path.index(next_indexes) - curr_index
                                best_index = path.index(next_indexes)
    
                    # Проверим были ли изменения 
                    # Если да, то меняем следующую точку
                    if max_dist != -1 and best_index != -1:
                        curr_index = best_index
                    else:
                        curr_index += 1
                        
                # Новый путь поставим на место старого
                pathes[path_idx] = new_path
                
            # Сначала понизим кол-во феромонов на путях, и отредактируем кол-во феромонов
            used_edges = set()
            for path in pathes:
                for i in range(len(path) - 1):
                    used_edges.add((path[i], path[i + 1]))
            
            if epoch < start_epochs:
                ro = self.ro_start
            else:
                ro = self.ro_end
                
            for first_index, second_index in used_edges:
                desript_graph[first_index][second_index]['feromon'] *= (1 - ro)
            
            # Пройдем по полученным маршрутам
            update_best_path = False
            count_feromons = 0
            summ_feromons = 0
            
            for path_one_ant in pathes:
                dist, time, mean_BF, fuel, rang_path = self.dist_time_meanBF_rang_path(path_one_ant)
                # Проверим является ли путь лучшим:
                curr_path_best = False
                if ((self.best_path['rang'][0] > dist or self.best_path['rang'][1] > time or self.best_path['rang'][2] > fuel) and 
                    self.best_path['rang'][3] < rang_path and path_one_ant[-1] == self.end_index):
                    self.best_path['rang'] = [dist, time, fuel, rang_path]
                    self.best_path['path'] = path_one_ant.copy()
                    curr_path_best = True
                    update_best_path = True
    
                # Соединим лучший маршрут и текущий, по общей точке если она есть, и если текущий маршрут не является лучшим
                if not curr_path_best:
                    for i in range(len(path_one_ant)):
                        # Проверяем каждую точку есть ли она в лучшем маршрутме
                        curr_index = path_one_ant[i]
                        # если есть соединяем как первая часть лучшего + вторая часть текущего и наоборот
                        if curr_index in self.best_path:
                            best_path_index = self.best_path.index(curr_index)
                            new_path_1 = path_one_ant[:i] + self.best_path[best_path_index:]
                            new_path_2 = self.best_path[:best_path_index] + path_one_ant[i:]
                            dist1, time1, mean_BF1,fuel1, rang_path1 = self.dist_time_meanBF_rang_path(new_path_1)
                            dist2, time2, mean_BF2, fuel1, rang_path2 = self.dist_time_meanBF_rang_path(new_path_2)
                            
                            # Проверим какой из маршрутов является лучшим
                            if ((self.best_path['rang'][0] > dist1 or self.best_path['rang'][1] > time1 or self.best_path['rang'][2] > fuel1) and 
                                self.best_path['rang'][3] < rang_path1 and new_path_1[-1] == self.end_index):
                                self.best_path['rang'] = [dist1, time1, fuel1, rang_path1]
                                self.best_path['path'] = new_path_1.copy()
                                update_best_path = True
        
                            if ((self.best_path['rang'][0] > dist2 or self.best_path['rang'][1] > time2 or  self.best_path['rang'][2] > fuel2) and 
                                self.best_path['rang'][3] < rang_path2 and new_path_2[-1] == self.end_index):
                                self.best_path['rang'] = [dist2, time2, fuel2, rang_path2]
                                self.best_path['path'] = new_path_2.copy()
                                update_best_path = True
                            
                # Изменим кол-во феромонов на ребрах по которым прошли
                for i in range(len(path_one_ant)-1):
                    first_index = path_one_ant[i]
                    second_index = path_one_ant[i+1]
    
                    delta_feromons = (
                            Q * self.desript_graph[first_index][second_index]['speed'] / 
                            ( 
                                (dist * 1e-3)**dist_degree * self.desript_graph[first_index][second_index]['distance'] * 0.01  * 
                                self.desript_graph[first_index][second_index]['BF'] * mean_BF *
                                self.desript_graph[first_index][second_index]['fuel'] * fuel * 1e-3 * 
                                mean_BF * (len(path_one_ant) / 100)
                            )
                        )
                    count_feromons += 1
                    summ_feromons += delta_feromons
                    
                    if path_one_ant[-1] != end_index:
                        delta_feromons = 0.02 * delta_feromons
                        
                    desript_graph[first_index][second_index]['feromon'] += delta_feromons
                            
                    curr_feromon = self.desript_graph[first_index][second_index]['feromon']
                    if curr_feromon < self.min_feromon:
                        self.desript_graph[first_index][second_index]['feromon'] = self.min_feromon
                    if curr_feromon > self.max_feromon or np.isnan(curr_feromon):
                        self.desript_graph[first_index][second_index]['feromon'] = self.max_feromon
                    
                        
            mean_delta_feromons = summ_feromons / count_feromons if count_feromons != 0 else 0.5
            if not 0.1 < mean_delta_feromons < 1:
                # print(f'mean_delta_feromons : {mean_delta_feromons} and Q : {Q * 0.1 / mean_delta_feromons}')
                Q *= 0.25 / mean_delta_feromons
    
            if update_best_path:      
                epoch_without_updates = count_epoch_without_update
                save_path = np.array([self.grid[i][j] for i, j in self.best_path['path']])
                dist = self.best_path['rang'][0]
                time = self.best_path['rang'][1]
                fuel = self.best_path['rang'][2]
                rang = self.best_path['rang'][3]
                os.makedirs('results_pathes', exist_ok=True)
                
                np.save(os.path.join('results_pathes', f'path_{dist:.1f}_{time:.1f}_{rang:.1f}_{fuel:.1f}.npy'), save_path)
            else:
                epoch_without_updates -= 1
    
            if epoch_without_updates < 0:
                return self.best_path
            
            # Добавим описание
            dist = self.best_path['rang'][0]
            time = self.best_path['rang'][1]
            fuel = self.best_path['rang'][2]
            rang_path = self.best_path['rang'][3]
            
            # print(f'epoch: {epoch + 1}, dist: {dist:.3f}, time : {time:.3f}, rang : {rang_path:.3f}, count_pathes : {len(pathes)}')
            pbar.set_description(
                f'epoch: {epoch + 1}, dist: {dist:.3f}, time : {time:.3f}, fuel : {fuel:.3f},  rang : {rang_path:.3f}, count_pathes : {len(pathes)}'
            )
        return self.best_path
        
def visual_ACO_path(aco, start_point, end_point, Ocean_map = Ocean_map)
    m = folium.Map(tiles="cartodbpositron", world_copy_jump=True)
    
    group_1 = folium.FeatureGroup("first group").add_to(m)
    folium.GeoJson(Ocean_map).add_to(group_1)
    
    folium.Marker((start_point.coords[0][1], start_point.coords[0][0]), tooltip="start_point").add_to(m)
    folium.Marker((end_point.coords[0][1], end_point.coords[0][0]), tooltip="end_point").add_to(m)
    
    folium.PolyLine([grid[i][j][::-1] for i, j in aco.best_path['path']], color = 'red', weight = 5).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)

    m.save('ACO_path.html')

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
    
    aco = ACO(
        grid = grid,
        desript_graph = desript_graph, 
        start_index = start_index, 
        end_index = end_index,
        count_try_get_path = config['count_try_get_path'],
        max_count_points_in_path = config['max_count_points_in_path'], 
        alpha = config['alpha'],
        beta = config['beta'],
        min_feromon = config['min_feromon'],
        max_feromon = config['max_feromon'],
        ro_start =  config['ro_start'],
        ro_end = config['ro_end'], 
    )
    
    aco.setup_graph_and_create_path(
        count_path_for_epoch = config['count_path_for_epoch'],
        start_count_path_for_epoch = config['start_count_path_for_epoch'],
        start_epochs = config['start_epochs'],
        epochs = config['epochs'],
        Q = config['Q'],
        count_epoch_without_update = config['count_epoch_without_update'],
        all_pathes = config['all_pathes'],
        dist_degree = config['dist_degree'],
    )
    
    visual_ACO_path(aco, start_point = config['start_point'], end_point = config['end_point'], Ocean_map = config['Ocean_map'])
