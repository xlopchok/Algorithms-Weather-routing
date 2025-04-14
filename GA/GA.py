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
    'pathes_dir' : 'results_pathes',
    'Ocean_map' : Ocean_map,
    'epochs' : 50,  
    'max_count_indeividuals' : 25, 
    'p_1_point' : 0.15, 
    'p_2_point' : 0.15, 
    'p_skip_candidate' : 0.9
}


def One_point_Crossover(path1, path2, p_1_point = 0.2, good_dist = 10):
    '''
    Функция реализует скрещивание двух путей по одной точке,
    возвращает несколько полученных маршрутов
    
    path1, path2 - два пути
    p_1_point - вероятность скрещивание
    good_dist - расстояние в километрах, если две точки находятся на расстоянии не больше выбранного, то считаются совпадающими
    '''
    new_pathes = []
    
    for index_point1, point1 in enumerate(path1):
        if index_point1 == 0 or index_point1 == len(path1) - 1:
            continue
        for index_point2, point2 in enumerate(path2):
            if index_point2 == 0 or index_point2 == len(path2) - 1:
                continue
            if haversine(point1[1], point1[0], point2[1], point2[0]) < good_dist:
                posib_val = np.random.uniform()
                if posib_val < p_1_point:
                    new_path = list(path1[:index_point1]) + list(path2[index_point2:])
                    new_pathes.append(new_path)

                    new_path =  list(path2[:index_point2]) + list(path1[index_point1:])
                    new_pathes.append(new_path)
    return new_pathes

def Two_point_Crossover(path1, path2, p_2_point, good_dist = 10):
    '''
    Функция реализует скрещивание двух путей по двум точке,
    возвращает несколько полученных маршрутов
    
    path1, path2 - два пути
    p_2_point - вероятность скрещивание
    good_dist - расстояние в километрах, если две точки находятся на расстоянии не больше выбранного, то считаются совпадающими
    '''
    new_pathes = []
     # Берем первую точку в первом пути
    for index_point1_1, point1_1 in enumerate(path1):
        if index_point1_1 == 0 or index_point1_1 == len(path1) - 1:
            continue
        # Берем первую тчоку во втором пути
        for index_point2_1, point2_1 in enumerate(path2):
            if index_point2_1 == 0 or index_point2_1 == len(path2) - 1:
                continue
            # Проверяем что они находтся рядом
            dist_between_11 = haversine(point1_1[1], point1_1[0], point2_1[1], point2_1[0])
            if dist_between_11 > good_dist:
                continue
            # Берем вторую точку в первом пути
            for index_point1_2, point1_2 in enumerate(path1[index_point1_1 + 1:]):
                index_point1_2 += index_point1_1 + 1
                if index_point1_2 == 0 or index_point1_2 == len(path1) - 1:
                    continue
                # Берем вторую тчоку во втором пути
                for index_point2_2, point2_2 in enumerate(path2[index_point2_1 + 1 : ]):
                    index_point2_2 += index_point2_1 + 1
                    if index_point2_2 == 0 or index_point2_2 == len(path2) - 1:
                        continue
                    dist2_between_22 = haversine(point1_2[1], point1_2[0], point2_2[1], point2_2[0])
                    
                    if dist_between_11 < good_dist and dist2_between_22 < good_dist:
                        posib_val = np.random.uniform()
                        if posib_val < p_2_point:
                            new_path = list(path2[: index_point2_1]) + list(path1[index_point1_1 + 1 : index_point1_2]) + list(path2[index_point2_2 + 1 : ])
                            new_pathes.append(new_path)

                            new_path = list(path1[: index_point1_1]) + list(path2[index_point2_1 + 1 : index_point2_2]) + list(path1[index_point1_2 + 1 : ])
                            new_pathes.append(new_path)
    return new_pathes

def Walk_mutauions(path, big_dist, curr_time_now = None, weather_data = None):
    '''
    Функция реализует мутацию Walk Mutations
    
    path - рассматриваемый путь
    big_dist - максимальная допустимая длина ребра
    curr_time_now - текущий момент времени
    weather_data - погодные данные
    '''
    new_path = []
    curr_index = 0
    # Будем в цикле идти по индексам точек 
    while curr_index < len(path):
        point1 = path[curr_index]
        new_path.append(point1)
        best_index = -1
        best_rang = -1
        for next_index in range(curr_index + 1, len(path)):
            point2 = path[next_index]
            curr_dist = haversine(point1[1], point1[0], point2[1], point2[0])
            line = LineString([point1, point2])
            if curr_dist < big_dist and Ocean_map.contains(line):
                was_path = list(path[curr_index:next_index + 1])
                now_path = [path[curr_index] , path[next_index]]
                was_dist, was_fuel, was_time = analys_path([point[::-1] for point in was_path], type_work = type_work, curr_time_now = curr_time_now, weather_data = weather_data)
                now_dist, now_fuel, now_time = analys_path([point[::-1] for point in now_path], type_work = type_work, curr_time_now = curr_time_now, weather_data = weather_data)
                
                now_rang = now_dist * 1e-3 / 9 + now_time * 1e-2 / 4
                was_rang = was_dist * 1e-3 / 9 + was_time * 1e-2 / 4
                if now_rang < was_rang and was_rang - now_rang > best_rang:
                    best_index = next_index
                    best_rang = was_rang - now_rang
        if best_index != -1:
            curr_index = best_index
        else:
            curr_index += 1
    return new_path

def Random_Walk_mutations(path, p_move, noise_step = 5):
    '''
    Функция релизующуя мутаци Random_Walk_mutations, добавляет шум в маршруты, и сдвигает точки
    path - рассматриваемый путь
    p_move - вероятность добавления шума
    noise_step - максимальное допустимое смещение точки по координатам
    '''
    new_path = []
    for index, point in enumerate(path):
        if index == 0 or index == len(path) - 1:
            new_path.append(point)
            continue
        new_point = point.copy()
        p_val = np.random.uniform()
        if p_val < p_move:
            rand_lon = np.random.uniform(-noise_step, noise_step)
            rand_lat = np.random.uniform(-noise_step, noise_step)

            new_point[0] += rand_lon
            new_point[1] += rand_lat

            if not Ocean_map.contains(Point(new_point)):
                new_point = point.copy()
        new_path.append(new_point)
    return new_path
    
def GA(
    pathes_dir = 'start_populations',
    type_work = 'const_fuel',
    p_1_point = 0.2, # Вероятность добавить маршрут после OnePointCrossover
    p_2_point = 0.5, # Вероятность добавить маршрут после Two Point Crossover
    good_dist = 10, # Точки находящиеся расстоянии меньше этого считаются совпадающими
    p_change_point = 0.25, # 
    p_move = 0.5, # Вероятность смещения точки в пути
    big_dist = 300, # Допустимое расстояние для длины ребра
    epochs = 25, # Кол-во эпох
    max_count_indeividuals = 35, # Максимальное кол-во представитеей в эпохе
    eps = 10e-3, 
    noise_step = 5, # Радиус добавленного шума
    count_random_walk = 5, # Сколько раз к одному маршруту применятьеся Random Walk
    p_skip_candidate = 0.5, # Вероятность выживания к моменту селекции
    curr_time_now = None, 
    weather_data = None
):
    '''
    Генетический алгоритм
    pathes_dir - Директория с маршрутами стартовой популяции
    type_work - стратегия движения
    p_1_point - вероятность создать новый маршрут после OnePointCrossover 
    p_2_point - Вероятность создать новый маршрут после Two Point Crossover
    good_dist - Точки находящиеся расстоянии меньше этого считаются совпадающими
    p_change_point - вероятность поменять точки местами
    p_move - Вероятность смещения точки в пути
    big_dist -  Допустимое расстояние для длины ребра
    epochs - Кол-во эпох
    max_count_indeividuals - Максимальное кол-во представитеей в эпохе 
    noise_step - Радиус добавленного шума
    count_random_walk - Колличество применений  Random Walk к одному маршруту
    p_skip_candidate - Вероятность выживания к моменту селекции
    curr_time_now - текущий момент времени
    weather_data - погодные данные
    '''
    start_pathes = [np.load(path) for path in os.listdir(pathes_dir) if path.endswith('.npy')]
    curr_individs = []
    
    for i, path in enumerate(start_pathes):
        distance, fuel, time = analys_path([point[::-1] for point in path], type_work = type_work, curr_time_now = curr_time_now, weather_data = weather_data)
        curr_individs.append({
            'id' : i + 1,
            'path' : path,
            'time' : time,
            'dist' : distance,
            'fuel' : fuel,
        })

    for epoch in range(epochs):
        start_pathes = [individ['path'] for individ in curr_individs]
        new_pathes = []
        
        '''
        Сначала будем скрещивать маршруты по одной общей точке
        Считаем точки общими если они находятся на расстоянии меньше good_dist
        Будем делать скрещивание с вероятностью p_1_point
        '''
        for i, path1 in tqdm(enumerate(start_pathes), desc = f'epoch : {epoch + 1} OnePointCrossover, count_new pathes {len(new_pathes)}'):
            for path2 in start_pathes[i + 1 :]:
                res_pathes = One_point_Crossover(path1, path2, p_1_point = p_1_point, good_dist = good_dist)
                new_pathes.extend(res_pathes)
        '''
        Теперь будем скрещивать маршруты по двум общим точкам, меняя середину
        Считаем точки общими если они находятся на расстоянии меньше good_dist
        Будем делать скрещивание с вероятностью p_2_point
        '''
        # Берем первый маршрут
        for i, path1 in tqdm(enumerate(start_pathes), desc = f'epoch : {epoch + 1} Two_point_Crossover, count_new pathes {len(new_pathes)}'):
            # Берем второй маршрут
            for path2 in start_pathes[i + 1:]:
                posib_val = np.random.uniform()
                if posib_val < p_2_point:
                    res_pathes = Two_point_Crossover(path1, path2, p_2_point = p_2_point, good_dist = good_dist)
                    new_pathes.extend(res_pathes)

        '''
        Теперь добавим мутации
        - Удлаение точек (Если улучшается маршрут) 
        - Перемещаем точки в случайную сторону с вероятностью p_move
        '''
    
        '''
        Удаление 
        '''
        for path in tqdm(start_pathes, desc = f'epoch : {epoch + 1} Walk_mutauions, count_new pathes {len(new_pathes)}'):
            res_path = Walk_mutauions(path, big_dist)
            new_pathes.append(res_path)

        '''
        Перемещение
        '''
        for path in tqdm(start_pathes, desc = f'epoch : {epoch + 1} Random_Walk_mutations, count_new pathes {len(new_pathes)}'):
            # Будем для каждого маршрутов создавать 2 его смещенных версии
            for _ in range(count_random_walk):
                res_path = Random_Walk_mutations(path, p_move = p_move, noise_step = noise_step)
                new_pathes.append(res_path)

        '''
        Теперь проанализируем полученные маршруты и выберем лучшие
        '''
        curr_idx = max([item['id'] for item in curr_individs])
        dist_time_fuel_unique = set()
        
        for individ in curr_individs:
            dist = individ['dist']
            time = individ['time']
            fuel = individ['fuel']
    
            dist_time_fuel_unique.add((dist, time, fuel))
            
        for path in tqdm(new_pathes, desc = f'epoch : {epoch + 1}'):
            posib_val = np.random.uniform()
            if posib_val < p_skip_candidate:
                continue
            distance, fuel, time = analys_path([point[::-1] for point in path], type_work = type_work, curr_time_now = curr_time_now, weather_data = weather_data)
            if (distance, time, fuel) not in dist_time_fuel_unique:
                curr_idx += 1
                curr_individs.append({
                    'id' : curr_idx,
                    'path' : path,
                    'time' : time,
                    'dist' : distance,
                    'fuel' : fuel,
                })
                dist_time_fuel_unique.add((distance, time, fuel))
    
        curr_individs.sort(key = lambda x: 1e-3 * x['dist'] / 9 + 1e-2 * x['time'] / 4 + 1e-2 * x['fuel'] / 5)
        
        # Сначала возьмем max_count_indeivids Лучших
        curr_individs = curr_individs[:max_count_indeividuals]
        time, dist, fuel = curr_individs[0]['time'], curr_individs[0]['dist'], curr_individs[0]['fuel']
        print(f'epoch : {epoch + 1} \ntype_work : {type_work} \ntime : {time}; dist : {dist}; fuel : {fuel}\n\n')

        best_path_epoch = np.array(curr_individs[0]['path'])
        time = curr_individs[0]['time']
        dist = curr_individs[0]['dist']
        fuel = curr_individs[0]['fuel']
        
        np.save(os.path.join(pathes_dir, f'GA_res_Epoch_{epoch+1}_type_{type_work}_time_{time:.2f}_dist_{dist:.2f}_fuel_{fuel:.2f}'), best_path_epoch)
    return curr_individs

def visual_path(
    path, 
    Ocean_map,
    start_point, 
    end_point,
    map_path = 'GA_path.html',
):
    m = folium.Map(tiles="cartodbpositron", world_copy_jump=True)
    
    group_1 = folium.FeatureGroup("first group").add_to(m)
    folium.GeoJson(Ocean_map).add_to(group_1)
    
    folium.Marker((start_point.coords[0][1], start_point.coords[0][0]), tooltip="start_point").add_to(m)
    folium.Marker((end_point.coords[0][1], end_point.coords[0][0]), tooltip="end_point").add_to(m)
    
    lines = folium.FeatureGroup("Lines").add_to(m)
    folium.PolyLine([point[::-1] for point in path], color = 'red').add_to(lines)
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)

    os.makedirs('results_visual', exist_ok=True)
    m.save(os.path.join('results_visual', map_path))



if __name__ == 'main':
    for type_work in [
        'const_speed',
        'const_fuel',
        'max_persent_speed',
        'max_persent_fuel',
    ]:
         res = GA(
             pathes_dir = config['pathes_dir'],
             epochs = config['epochs'], 
             type_work = type_work, 
             max_count_indeividuals = config['max_count_indeividuals'], 
             p_1_point = config['p_1_point'], 
             p_2_point = config['p_2_point'], 
             p_skip_candidate = config['p_skip_candidate'],
         )

    visual_path(
        path = res[0]['path'],
        Ocean_map = config['Ocean_map'],
        start_point = config['start_point'], 
        end_point = config['end_point'],
        map_path = 'GA_path.html',
    )
