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

from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans

from maps.Ocean_map import Ocean_map
from Weather_Map.Weather_map import weather_data, gpd_weather_data, extrimly_weather_data
from sub_functions.subfunctions import haversine, smoothing_path, curr_speed_and_fuel, destination_point, analys_path, nearest_data_in_weather_data


def check_point(lat1, lon1, lat2, lon2, work_space, all_isohrons_area = None, dist_to_glob_boundary = 0.03): 
    last_point = Point(lon2, lat2)
    if all_isohrons_area is not None:
        work_space = work_space.difference(all_isohrons_area)
        if not work_space.is_valid:
            work_space = make_valid(work_space)
            
    if work_space.contains(LineString([[lon1, lat1], [lon2, lat2]])):
        if max([Point([lon1, lat1]).distance(Ocean_map.boundary), 
            Point([lon1, lat1]).distance(work_space.boundary)]) < dist_to_glob_boundary:
            return lat1, lon1, True
        return lat1, lon1, True

    line = LineString([
        [lon1, lat1],
        [lon2, lat2]
    ]).intersection(work_space)
    if type(line) == shapely.geometry.multilinestring.MultiLineString or type(line) == shapely.geometry.collection.GeometryCollection:
        curr_line = None
        min_dist = float('inf')
        for sub_line in line.geoms:
            curr_dist = sub_line.distance(Point(lon2, lat2))
            if curr_dist < min_dist:
                min_dist = curr_dist
                curr_line = sub_line
        line = curr_line
    else:
        if type(line) == shapely.geometry.linestring.LineString and len(line.coords) == 0:
            return None, None, False
    if type(line) == shapely.geometry.linestring.LineString:
        result = line.coords[0]
        return result[1], result[0], False
    else:
        return None, None, False

def create_isohrons_array(data, step, all_isohrons_area = None, algorithm = 'optics', n_clusters = 3, coef_clustering = 0.2):
    data = data[data['step'] == step]['curr_point']
    if algorithm == 'optics':
        # Кластеризация
        optics = OPTICS(min_samples=int(len(data) * coef_clustering)).fit(list(data))
        
        # Метки кластеров
        labels = optics.labels_
    elif algorithm == 'kmeans':
        # Количество кластеров
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Кластеризация
        kmeans.fit(list(data))
        
        # Метки кластеров
        labels = kmeans.labels_
    
    multy_points_arr = []
    if all_isohrons_area is None:
        for label in set(labels):
            multy_points_arr.append(
                MultiPoint(list(data.iloc[np.where(labels == label)])).convex_hull.intersection(Ocean_map)
            )
    else:
        for label in set(labels):
            multy_points_arr.append(
                MultiPoint(list(data.iloc[np.where(labels == label)])).convex_hull.intersection(Ocean_map).difference(all_isohrons_area)
            )
    return multy_points_arr

def choose_best_points(
    all_isohrons_area,
    next_step_data, 
    min_coef_value = 0.02,
    radius = 0.25,
    coef_nearest = 0.9,
    count_epochs = 800,
    min_count_points = 150,
    max_count_points = 1000,
    valid_mean_distance = 0.65,
    dist_to_glob_boundary = 0.005,
    coef_clustering = 0.2, 
    abs_min_count_points = 120,
):
    # Выберем те точки которые будут лежат в раннее построенной области 
    # (такое возможно после нескольких поворотов на суммарный угол > 180)
    # также уберем те точки которые получились билзкими к заданной границе
    curr_step = int(max(next_step_data['step']))
    
    drop_indexes = [idx 
                    for idx in next_step_data.index 
                    if Point(next_step_data.loc[idx, 'curr_point']).distance(all_isohrons_area) < dist_to_glob_boundary
                   ]
    if len(next_step_data) > abs_min_count_points:
        if len(next_step_data) - len(drop_indexes) < abs_min_count_points:
            drop_indexes = np.random.choice(
                drop_indexes, 
                len(next_step_data) - int(abs_min_count_points * 1.3) if len(next_step_data) - int(abs_min_count_points * 1.3) > 0 else 50
            )
        next_step_data = next_step_data.drop(drop_indexes)

    # выделим соответсвующую область для новых точек, чтобы учитывать только новые внешниии границы
    # Объединим эту область с ранее созданной

    multy_polygon_array = create_isohrons_array(
        next_step_data, 
        next_step_data['step'].max(), 
        all_isohrons_area = all_isohrons_area, 
        coef_clustering = coef_clustering
    )
    
    poly = unary_union(multy_polygon_array)
    
    # Чтобы сделать объединение должно быть valid
    if not poly.is_valid:
        poly = make_valid(poly)

    if type(poly) != shapely.geometry.polygon.Polygon: 
        min_count_points = sum([
                poly.geoms[i].area for i in range(len(poly.geoms)) 
                if not (
                    type(poly.geoms[i]) == shapely.geometry.linestring.LineString
                    or type(poly.geoms[i]) == shapely.geometry.point.Point
                ) 
            ]) * 10
    else:
        min_count_points = poly.area * 10
    
    poly = poly.union(all_isohrons_area)
    if type(poly) != shapely.geometry.polygon.Polygon:
        poly = unary_union(
            [
                poly.geoms[i] for i in range(len(poly.geoms)) 
                if not (
                    type(poly.geoms[i]) == shapely.geometry.linestring.LineString
                    or type(poly.geoms[i]) == shapely.geometry.point.Point
                ) 
            ]
        )
    
    if not poly.is_valid:
        poly = make_valid(poly)
    # Найдем расстояния до ее границы для всех точек которые могут быть использованы дальше (flag == True)
    # И не будем учитывать точки которые находятся слишком близко к заданным границам
    all_distance = [
        Point(next_step_data.loc[idx, 'curr_point']).distance(poly.boundary) 
        for idx in next_step_data.index
        if next_step_data.loc[idx, 'can_move']
    ]
    # Найдем среднее расстояние для выбранных точек до внешней границы
    mean = np.mean(all_distance)

    # также отберем соотвествующие точки
    choosen_points = [True 
                      if Point(next_step_data.loc[idx, 'curr_point']).distance(poly.boundary) < mean * min_coef_value  
                      else False 
                      for idx in next_step_data.index
                     ]
    
    new_data = next_step_data[choosen_points]
    
    bound_data = new_data[new_data['can_move'] == False]

    new_data = new_data[new_data['can_move'] == True]

    
    min_count_points = max(min_count_points, abs_min_count_points)
    print(f'min_count_points = {min_count_points}')
    # Теперь удалим несколько точек по следующему алгоритму
    # Берем случайную точку из оставшихся
    # Для нее строим шар заданного радиуса
    # Рассматриваем все точки в данном шаре и упорядочиваем их
    # Оставляем заданную часть точек не меньше хотя бы одной
    # Если точек будет меньше чем допустимый минимум останавливаемся
    pbar = tqdm(range(count_epochs), desc = f'now have {len(new_data)} elements')
    for iterations in pbar:
        if len(new_data) < min_count_points and len(new_data) < max_count_points:
            bound_data = bound_data.loc[np.random.choice(bound_data.index, size=min(120, int(0.02 * len(bound_data))), replace=False)]
            new_data = pd.concat([new_data, bound_data], ignore_index = True)
            return new_data
        # выбираем случайную точку
        idx = random.choice(new_data.index)
        point = new_data.loc[idx, 'curr_point']

        # Создаем шар
        circe = Point(point).buffer(radius)

        # Выбираем точки лежащие в шаре
        
        near_points = [
            [i, new_data.loc[i, 'fuel'], new_data.loc[i, 'Dist2End'], new_data.loc[i, 'DistFromStart']]
            for i in new_data.index 
            if circe.contains(Point(new_data.loc[i, 'curr_point']))
        ]

        # Сортируем выбранные точки
        if curr_step < 22:
            near_points.sort(key = lambda x: [x[3], x[1]])
        else:
            near_points.sort(key = lambda x: [x[2], x[1]])

        # Оставляем лучшие точки
        count_drop = max(2, int(coef_nearest * len(near_points)))
        drop_indexes = [i[0] for i in near_points[count_drop: ]]
        new_data = new_data.drop(drop_indexes)

        if iterations % 20 == 0:
            pbar.desc = f'now have {len(new_data)} elements'
    pbar.desc = f'now have {len(new_data)} elements'
    # Если после прохода по всем эпохам кол-во точек привышает допустимый максимум, то повторяем
    return new_data

def create_isohrones(
    data = None,
    start_point = start_point, 
    end_point = end_point, 
    time = 7,
    Ocean_map = Ocean_map,
    count_steps = 50,
    radius = 0.5,
    coef_nearest = 0.9,
    max_angel = 60,
    angel_step = 3,
    min_coef_value = 0.2,
    count_epochs = 100,
    step_for_count_points = 50,
    max_count_points = 1000,
    valid_mean_distance = 0.65,
    dist_to_glob_boundary = 0.03,
    step_x  = 10,
    step_y = 10,
    coef_clustering = 0.2, 
    abs_min_count_points = 100,
    step_count_points = 15,
):
    # Если данные не заданы, то создаем набор данных
    if data is None:
        isohrons = pd.DataFrame(
            {
                'curr_point' : [np.array([start_point.coords[0][0], start_point.coords[0][1]])],
                'last_point' : [np.nan], 
                'time' : [0], 
                'fuel' : [0], 
                'step' : [0],
                'Dist2End' : [haversine(start_point.coords[0][1], start_point.coords[0][0], end_point.coords[0][1], end_point.coords[0][0])],
                'can_move' : [True],
                'DistFromStart' : [0],
            },
            columns = ['curr_point', 'last_point', 'time', 'fuel', 'step', 'Dist2End', 'can_move', 'DistFromStart'],
        )

    # Cоздаем рабочую прямоугольную область отталкиваясь от расположения начальной и конечной точки
    min_x, min_y, max_x, max_y =  LineString([start_point, end_point]).bounds
    
    max_x = np.round(max_x + step_x) + 1
    max_y = np.round(max_y + step_y) + 1
    min_x = np.round(min_x - step_x) - 1
    min_y = np.round(min_y - step_y) - 1
    
    work_space = Polygon([
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
        [max_x, min_y]
    ]).intersection(Ocean_map)

    # Если данные не заданы, то делаем первый шаг, поворачиваясь вокруг начальной точки на 360 градусов
    if data is None:
        for gradus in range(0, 360, angel_step):
            rad = np.radians(gradus)
            move_vector = np.array([np.cos(rad), np.sin(rad)])
            speed, fuel = curr_speed_and_fuel(start_point, move_vector)
            fuel = fuel * time / 24
            move_vector = np.array([np.sin(rad), np.cos(rad)])
            res_lat, res_lon = destination_point(start_point.coords[0][1], start_point.coords[0][0], move_vector, 
                                                 speed, time)
        
            res_lat, res_lon, flag = check_point(res_lat, res_lon, start_point.coords[0][1], start_point.coords[0][0], 
                                                 work_space, dist_to_glob_boundary = dist_to_glob_boundary)  
            if res_lat is None or res_lon is None:
                continue
            
            isohrons.loc[len(isohrons)] = [
                np.array([res_lon, res_lat]), 
                np.array([start_point.coords[0][0], start_point.coords[0][1]]),
                time, 
                fuel, 
                1, 
                haversine(res_lat, res_lon, np.array(end_point.coords)[0][1], np.array(end_point.coords)[0][0]),
                flag,
                haversine(res_lat, res_lon, np.array(start_point.coords)[0][1], np.array(start_point.coords)[0][0])
            ]
            all_isohrons_area = MultiPoint(
                list(isohrons['curr_point'])).convex_hull.intersection(Ocean_map)
    # Если данные заданы то нужно создать область в которой не должно быть новых точек : all_isohrons_area
    else:
        # Копируем входные данные
        isohrons = data.copy()
        
        array_multi_polygon = []
        for step in range(1, max(isohrons['step']) + 1):
            array_multi_polygon.extend(create_isohrons_array(isohrons, step, coef_clustering = coef_clustering))
        # Из полученного набора полигонов строим мультиполигон
        all_isohrons_area = unary_union(array_multi_polygon)
        
        # В случае если границы пересекаются или совпадают применяем функцию make_valid
        if not all_isohrons_area.is_valid:
            all_isohrons_area = make_valid(all_isohrons_area)

    # строим новые изохроны иттеративно
    while count_steps > max(isohrons['step']):
        curr_step = max(isohrons['step']) + 1

        # Будем менять время на один шаг в зависимости от номера шага
        curr_time = time
        if 10 < curr_step < 15 or 26 >= curr_step >= 24:
            curr_time = time * 2
        elif 15 <= curr_step <= 24:
            curr_time= time * 5

        
        next_step_data = pd.DataFrame(
            columns = ['curr_point', 'last_point', 'time', 'fuel', 'step', 'Dist2End', 'can_move', 'DistFromStart'],
        )
        
        last_isohron = isohrons[isohrons['step'] == curr_step - 1].copy()
        pbar = tqdm(last_isohron.index, desc = f'create {len(next_step_data)} new elements and now step {curr_step}')
        for idx in pbar:
            data_of_point = last_isohron.loc[idx]
            if not data_of_point['can_move']:
                continue
            curr_start = data_of_point['curr_point']
            last_point = data_of_point['last_point']
            
            last_move_vector = curr_start - last_point
            
            last_rad = np.arctan2(last_move_vector[1], last_move_vector[0]) 
                
            for gradus in range(-max_angel, max_angel + angel_step, angel_step):
                rad = np.radians(gradus)
                
                rad += last_rad 
                move_vector = np.array([np.cos(rad), np.sin(rad)])
                speed, fuel = curr_speed_and_fuel(curr_start, move_vector)
                
                fuel = fuel * curr_time / 24
                move_vector = np.array([np.sin(rad), np.cos(rad)])
                res_lat, res_lon = destination_point(curr_start[1], curr_start[0], move_vector, speed, curr_time)
        
                res_lat, res_lon, flag = check_point(res_lat, res_lon, last_point[1], last_point[0], work_space, all_isohrons_area = all_isohrons_area)
                if res_lat is None or res_lon is None:
                    continue
                    
                next_step_data.loc[len(next_step_data)] = [
                    np.array([res_lon, res_lat]), 
                    curr_start,
                    curr_time + data_of_point['time'], 
                    fuel + data_of_point['fuel'], 
                    curr_step, 
                    haversine(res_lat, res_lon, np.array(end_point.coords)[0][1], np.array(end_point.coords)[0][0]),
                    flag,
                    data_of_point['DistFromStart'] + haversine(res_lat, res_lon, curr_start[1], curr_start[0])
                ]
                if len(next_step_data) % 100 == 0:
                    pbar.desc = f'create {len(next_step_data)} new elements and now step = {curr_step}'
        
        next_step_data.to_pickle(f'next_step_data_{curr_step}.pkl')

        result_after_step = choose_best_points(
            next_step_data = next_step_data, 
            radius = radius, 
            coef_nearest = coef_nearest, 
            min_coef_value = min_coef_value, 
            count_epochs = count_epochs,
            max_count_points = max_count_points + curr_step * step_for_count_points,
            valid_mean_distance = valid_mean_distance * (1 + curr_step / count_steps),
            all_isohrons_area = all_isohrons_area,
            dist_to_glob_boundary = dist_to_glob_boundary * (curr_time / time),
            abs_min_count_points = abs_min_count_points + curr_step * step_count_points
        )
        
        isohrons = pd.concat([isohrons, result_after_step], ignore_index = True)
        
        array_multi_polygon = create_isohrons_array(isohrons, curr_step, all_isohrons_area = all_isohrons_area, coef_clustering = coef_clustering)
        res_isohrons = unary_union(array_multi_polygon)

            
        if not res_isohrons.is_valid:
            res_isohrons = make_valid(res_isohrons)

        if not all_isohrons_area.is_valid:
            all_isohrons_area = make_valid(all_isohrons_area)
            
        all_isohrons_area = res_isohrons.union(all_isohrons_area)
        
        if not all_isohrons_area.is_valid:
            all_isohrons_area = make_valid(all_isohrons_area)

        isohrons.to_pickle(f'isohrons_step{curr_step}.pkl')
            
        if all_isohrons_area.contains(end_point):
            return isohrons
          
    return isohrons

def find_isohron_path(
    data,
    start_point = start_point,
    end_point = end_point,
):
    isohron_path = []
    last_step = data['step'].max()
    data_of_nearest_point = data[data['Dist2End'] == data[data['step'] == last_step]['Dist2End'].min()]
    last_point = np.array(data_of_nearest_point['curr_point'])[0]

    isohron_path.append(np.array(end_point.coords[0]))
    while last_step > 0:
        isohron_path.append(last_point)

        last_step -= 1
        
        new_last_point = np.array(data_of_nearest_point['last_point'])[0]

        data_of_nearest_point = data[data['step'] == last_step]
        data_of_nearest_point = data_of_nearest_point[
        (data_of_nearest_point['curr_point'].apply(lambda x: x[0]) == new_last_point[0]) &
        (data_of_nearest_point['curr_point'].apply(lambda x: x[1]) == new_last_point[1])
        ]
        
        if len(data_of_nearest_point) > 1:
            data_of_nearest_point = data_of_nearest_point.iloc[0]
        
        last_point = new_last_point.copy()
        if last_step == 0:
            isohron_path.append(last_point)
    isohron_path = np.array([[point[1], point[0]] for point in isohron_path])
    return isohron_path


if __name__ == 'main':
    result_isohrones = create_isohrones(
        #data = data,
        start_point =  Point((3, 55)), # Начальная точка
        end_point = Point((-80, 15)), # Коечная точка
        count_steps = 60, # Максимальное кол-во слоев
        time = 5, # Время на один слой
        radius = 0.7, # Смотрим точки в определенном радиусе от каждой 
        coef_nearest = 0.5, # от всех точек в круге оставим только указанную часть
        max_angel = 75, # максимальное допустимое значение угла поворота
        angel_step = 5, # Шаг с которым перебираются углы из указанной области
        min_coef_value = 2, # минимальное допустимое значение coef полученное mean * min_coef_value  
        count_epochs = 5000, # Кол-во эпох в каждой из которых рассматривается точка и ближайшие к ней в заданном радиусе из них выбирают часть
        valid_mean_distance = 0.75, # Допустимое среднее расстояние, в случае когда оно меньше указанного valid_mean_distance * (1 + curr_step / count_steps)
        dist_to_glob_boundary = 0.15, # Расстояние до ближайшей глобальной границы
        coef_clustering = 0.19, # используется для кластеризации полученных точек
        abs_min_count_points = 100, # Минимальное кол-во точек для каждой изохроны
        step_count_points = 10, # Используется для вычисления минимального кол-ва точек для каждой изохроны на каждом шаге abs_min_count_points + step_count_points * curr_step   
        step_for_count_points = 15,
        max_count_points = 200,
    )
  
  



