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

def Random_Walk_mutations(path, p_move, start_point = start_point, end_point = end_point, noise_step = 5, step_x = 10, step_y = 10):
    '''
    Функция релизующуя мутаци Random_Walk_mutations, добавляет шум в маршруты, и сдвигает точки
    path - рассматриваемый путь
    p_move - вероятность добавления шума
    start_point, end_point - начальная и конечная точки
    noise_step - максимальное допустимое смещение точки по координатам
    step_x, step_y - параметры исплользованные для создания погодной карты, используется для контроля за положением новых точек
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

            if (
                not Ocean_map.contains(Point(new_point)) or  # Если находится на суше
                new_point[0] > max(start_point.coords[0][0], end_point.coords[0][0]) or # Если находится за перделами построенной погодной карты
                new_point[1] > max(start_point.coords[0][1], end_point.coords[0][1]) or # Если находится за перделами построенной погодной карты
                new_point[0] < min(start_point.coords[0][0], end_point.coords[0][0]) or # Если находится за перделами построенной погодной карты
                new_point[1] < min(start_point.coords[0][1], end_point.coords[0][1])
            ):
                new_point = point.copy()
        new_path.append(new_point)
    return new_path

def tournament_selection(curr_individs, num_winners=100, tournament_size=15, seed=None):
    '''
    Функция реализует турнирную селекцию
    curr_individs - текущие представители
    num_winners - кол-во элементов после отбора
    tournament_size- кол-во участников в одном турнире
    seed - можно установить для воспроизводимости результатов
    '''
    if seed is not None:
        np.random.seed(seed)
    
    arr = np.array(range(len(curr_individs)))
    winners = []
    
    if num_winners > len(curr_individs):
        return curr_individs
        
    for _ in range(num_winners):
        if len(arr) == 0:
            return [curr_individs[index] for index in winners]
            
        if tournament_size > len(arr):
            tournament_size = int(0.8 * len(arr))
            if tournament_size == 0:
                tournament_size = 1
        # выбираем случайно `tournament_size` элементов
        indices = np.random.choice(len(arr), tournament_size, replace=False)
        tournament = arr[indices]
        
        winner_index = indices[np.argmin([curr_individs[index]['rang'] for index in indices])]
        
        # добавляем победителя
        winners.append(arr[winner_index])
        
        # удаляем его из массива, чтобы не повторялся
        arr = np.delete(arr, winner_index)

    return [curr_individs[index] for index in winners]

def roulette_wheel_selection(curr_individs, num_select, replace = True, seed=None):
    '''
    Функция реализует селекцию с выбором элемента по вероятности, зависящей от значения 
    curr_individs - текущие представители
    num_select - кол-во элементов после отбора
    replace - разрешение на выбор одинаковых элементов
    seed - можно установаить для воспроизводимости результатов
    '''
    if seed is not None:
        np.random.seed(seed)

    arr = np.array([1 / np.exp(individ['rang']) for individ in curr_individs])
    probs = arr / np.sum(arr)
    
    selected_indices = np.random.choice(len(arr), size = num_select, replace = replace, p = probs)
    return [curr_individs[index] for index in selected_indices]

def GA(
    start_point = start_point,
    end_point = end_point,
    pathes_dir = '/kaggle/working/',
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
    weather_data = None,
    dist_coef = 9e-3,
    time_coef = 4e-2,
    fuel_coef = 5e-2,
    num_winners = 100,
    tournament_size = 15,
    seed = None,
    replace = False,
    step_x = 10, 
    step_y = 10,
    top_k = 5,
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
    dist_coef, time_coef, fuel_coef - коэффициенты для оценки качества маршрутов
    num_winners - кол-во выбранных элементов после турнирной селекции
    seed - можно установить для воспроизводимости результатов
    replace - разрешение на повтроение элементов при селекции основанной на вероятности выбрать тот или иной элемент
    step_x, step_y - шаги, установленные для контроля построения новых точек, чтобы они не выходили за допустимые границы
    top_k - кол-во обяхательно выбранных лучших маршрутов, этот параметр позволяет гарантировать не ухудшение качества маршрутов на следующих эпохах
    '''
    start_pathes = [np.load(os.path.join(pathes_dir, path)) for path in os.listdir('/kaggle/input/weathedata') if path.endswith('.npy')]
    curr_individs = []
    
    for i, path in enumerate(start_pathes):
        distance, fuel, time = analys_path([point[::-1] for point in path], type_work = type_work, curr_time_now = curr_time_now, weather_data = weather_data)
        curr_individs.append({
            'id' : i + 1,
            'path' : path,
            'time' : time,
            'dist' : distance,
            'fuel' : fuel,
            'rang' : dist_coef * distance + time_coef * time + fuel_coef * fuel,
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
                res_path = Random_Walk_mutations(path, p_move, start_point = start_point, end_point = end_point, noise_step = 5, step_x = step_x, step_y = step_y)
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
                    'rang' : dist_coef * distance + time_coef * time + fuel_coef * fuel,
                })
                dist_time_fuel_unique.add((distance, time, fuel))
                
        print(f'lost {len(curr_individs)} individs\n')
        curr_individs.sort(key = lambda x: x['rang'])

        '''
        Выберем top_k лучших, это гарантирует не понижение качества на каждой эпохе
        '''
        top_k_individs = curr_individs[:top_k]

        '''
        Применение селекции к полученным маршрутам
        '''
        curr_individs = tournament_selection(curr_individs[top_k:], num_winners = num_winners, tournament_size = tournament_size, seed = seed)
        curr_individs = roulette_wheel_selection(curr_individs, num_select = max_count_indeividuals, replace = replace, seed = seed)
        
        curr_individs = top_k_individs + curr_individs
        
        curr_individs.sort(key = lambda x: x['rang'])
        time, dist, fuel, rang = curr_individs[0]['time'], curr_individs[0]['dist'], curr_individs[0]['fuel'], curr_individs[0]['rang'],
        print(f'epoch : {epoch + 1} \ntype_work : {type_work} \ntime : {time:.2f}; dist : {dist:.2f}; fuel : {fuel:.2f}, rang : {rang:.2f}\n\n')
        
        best_path_this_epoch = np.array(curr_individs[0]['path'])
        time = curr_individs[0]['time']
        dist = curr_individs[0]['dist']
        fuel = curr_individs[0]['fuel']
        
        np.save(f'GA_res_Epoch_{epoch+1}_type_{type_work}_time_{time:.1f}_dist_{dist:.2f}_fuel_{fuel:.2f}', best_path_this_epoch)
    return curr_individs
    
def visual_pathes_res(
    res, 
    Ocean_map,
    start_point,  
    end_point,
    map_path = 'GA.html',
):
    
    Color = ['green', 'red', 'yellow', 'purple', 'orange', 'blue', 'pink']
    count_pathes = int(0.5 * len(res))
    if count_pathes < 3 and len(res) <= 3:
        count_pathes = 3
    if len(res) == 0:
        return
    if len(res) != 0 and count_pathes == 0:
        count_pathes = 1
    res.sort(key = lambda x : x['rang'])
    
    m = folium.Map(tiles="cartodbpositron", world_copy_jump=True)
    
    group_1 = folium.FeatureGroup("first group").add_to(m)
    folium.GeoJson(Ocean_map).add_to(group_1)
    
    folium.Marker((start_point.coords[0][1], start_point.coords[0][0]), tooltip="start_point").add_to(m)
    folium.Marker((end_point.coords[0][1], end_point.coords[0][0]), tooltip="end_point").add_to(m)
    
    lines = folium.FeatureGroup("Lines").add_to(m)
    for i in range(count_pathes):
        path = res[i]['path']
        folium.PolyLine([point[::-1] for point in path], color = Color[i % len(Coloe)]).add_to(lines)
        
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)
    os.makedirs('results_visual', exist_ok=True)
    m.save(os.path.join('results_visual', map_path))


config = {
    'start_point' : Point((3, 55)),
    'end_point' : Point((-80, 15)),
    'pathes_dir' : 'results_pathes',
    'Ocean_map' : Ocean_map,
    'epochs' : 50,  
    'max_count_indeividuals' : 25, 
    'p_1_point' : 0.15, 
    'p_2_point' : 0.15, 
    'p_skip_candidate' : 0.9,
    'p_move' : 0.75,
    'p_change_point' : 0.5,
    'good_dist' : 7,
    'big_dist' : 200,
    'eps' : 1e-6,
    'noise_step' : 7.5,
    'count_random_walk' : 25,
    'curr_time_now' : None,
    'weather_data' : None,
    'seed' : None,
    'dist_coef' : 0,
    'time_coef' : 1e-2 / 4,
    'fuel_coef' : 1e-2 / 8,
    'num_winners' : 100,
    'tournament_size' : 50,
    'replace' : False,
    'step_x' : 10,
    'step_y' : 10,
    'top_k' : 5,
    
}


if __name__ == 'main':
    for type_work in [
        'const_speed',
        'const_fuel',
        'max_persent_speed',
        'max_persent_fuel',
    ]:
        res = GA(
            start_point = config['start_point'],
            end_point = config['end_point'],
            pathes_dir = confi['pathes_dir'],
            type_work = type_work,
            p_1_point = config['p_1_point'], 
            p_2_point = config['p_2_point'], 
            p_skip_candidate = config['p_skip_candidate'], 
            good_dist = config['good_dist'], 
            p_change_point = config['p_change_point'], 
            p_move = config['p_move'], 
            big_dist = config['big_dist'], 
            epochs = config['epochs'],
            max_count_indeividuals = config['max_count_indeividuals'], 
            eps = config['eps'], 
            noise_step = config['noise_step'], 
            count_random_walk = config['curr_time_now'],
            curr_time_now = config['curr_time_now'], 
            weather_data = config['weather_data'],
            dist_coef = config['dist_coef'],
            time_coef = config['time_coef'],
            fuel_coef = config['fuel_coef'],
            num_winners = config['num_winners'],
            tournament_size = config['tournament_size'],
            seed = config['seed'],
            replace = config['replace'],
            step_x = config['step_x'], 
            step_y = config['step_y'],
            top_k = config['top_k'],
        )

    visual_path(
        res = res, 
        Ocean_map = config['Ocean_map'],
        start_point = config['start_point'],
        end_point = config['end_point'],
        map_path = f'{type_work}_GA.html', 
    )
