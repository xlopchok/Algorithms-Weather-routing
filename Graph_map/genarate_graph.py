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

def create_grid(
    start_point, # начальная точка
    end_point, # конечная точка
    step_km = 150, # расстояние между точками на прямой соединяющей начальную и конечную точки
    add_distations = 900, # для увеленчения области построения графика 
    R = 6371, # Радиус земли
    north_south_distance = 900, # Расстояние вверх и вниз от полученной прямой
    north_south_step = 150, # Шаг с которым строятся точки
    need_grid = True,
):
    '''
    Создает сначала кратчайшую дугу большого круга соединяющую начальную и конечную точки
    Затем на этой дуге строит сетку

    start_point - начльная точка
    end_point - конечная точка
    step_km - шаг с которым строятся точки на дуге большого круга
    
    add_distations - добавочное расстояние за начальную и конечную точку, увеличиват длину дуги
    в граничных ситуациях может быть слишком большим, из-за чего кривая начнет строиться в нарпавлении замыкания, до полного круга
    может привести к тому, что сетка будет накладываться сама на себя

    R - радиус Земли
    north_south_distance - добавочное расстояние на север и юг, для построения сетки
    north_south_step - шаг с которым строятся точки в направлени юга и севера
    
    need_grid - по дефолту стоит True, без него вернется только дуга большого круга, может пригодиться при настройке параметров, 
    чтобы избегать ситуаций когда сетка на себя накладывается
    '''
    L1 = np.radians(start_point.coords[0][1])    # широта первой точки
    L2 = np.radians(end_point.coords[0][1])   # широта второй точки
    lambda1 = np.radians(start_point.coords[0][0])  # долгота первой точки
    lambda2 = np.radians(end_point.coords[0][0]) # долгота второй точки
    
    DL0 = lambda2 - lambda1
    
    distance_km = haversine(start_point.coords[0][1], start_point.coords[0][0], end_point.coords[0][1], end_point.coords[0][0])
    
    # Начальный курс (азимут)
    C = np.arctan2(
        np.sin(DL0) * np.cos(L2),
        np.cos(L1) * np.sin(L2) - np.sin(L1) * np.cos(L2) * np.cos(DL0)
    )
    
    big_circle_path = [] #[[start_point.coords[0][1], start_point.coords[0][0]]]
    
    for Dx_km in range(-add_distations, int(distance_km) + add_distations + 1, step_km):
        lat, lon = destination_point(start_point.coords[0][1], start_point.coords[0][0], distance = Dx_km, angle = C)
        big_circle_path.append([lon, lat])

    pos_start = -1
    pos_end = -1
    for i in range(1, len(big_circle_path) - 1):
        if pos_start == -1 and ((big_circle_path[i][0] <= start_point.coords[0][0] <= big_circle_path[i+1][0] or 
             big_circle_path[i][0] >= start_point.coords[0][0] >= big_circle_path[i+1][0]) and
            (big_circle_path[i][1] <= start_point.coords[0][1] <= big_circle_path[i+1][1] or 
             big_circle_path[i][1] >= start_point.coords[0][1] >= big_circle_path[i+1][1])):
            if (haversine(big_circle_path[i][1], big_circle_path[i][0], start_point.coords[0][1], start_point.coords[0][0]) < 
                haversine(big_circle_path[i+1][1], big_circle_path[i+1][0], start_point.coords[0][1], start_point.coords[0][0])):
                pos_start = i
            else:
                pos_start = i + 1
        if pos_end == -1 and ((big_circle_path[i][0] <= end_point.coords[0][1] <= big_circle_path[i+1][0] or 
             big_circle_path[i][0] >= end_point.coords[0][0] >= big_circle_path[i+1][0]) and
            (big_circle_path[i][1] <= end_point.coords[0][1] <= big_circle_path[i+1][1] or 
             big_circle_path[i][1] >= end_point.coords[0][1] >= big_circle_path[i+1][1])):
            if (haversine(big_circle_path[i][1], big_circle_path[i][0], end_point.coords[0][1], end_point.coords[0][0]) <
                haversine(big_circle_path[i+1][1], big_circle_path[i+1][0], end_point.coords[0][1], end_point.coords[0][0])):
                pos_end = i
            else:
                pos_end = i + 1
           
    big_circle_path[pos_start] = [start_point.coords[0][0], start_point.coords[0][1]]
    big_circle_path[pos_end] = [end_point.coords[0][0], end_point.coords[0][1]]
    if need_grid:
        grid = []
    
        for Dx_km in range(north_south_distance, -north_south_distance - 1, -north_south_step):
            grid.append([])
            for point in big_circle_path:
                lat, lon = destination_point(point[1], point[0], distance = Dx_km, angle = np.radians(0))
                grid[-1].append([lon, lat])
        
        
        return big_circle_path, grid
        
    else:
        big_circle_path


def create_discript_graph(
    grid, 
    start_point, 
    end_point,
    eps = 1e-3,
    tao0 = 0.1,
    speed_degree = 1.5,
    weather_degree = 2,
    dist_degree = 1,
    max_cost = 75,
    dist_step = 1,
    can_diag_move = False,
    curr_time_now = None,
    weather_data = None,
    type_work = 'const_fuel',
):
    '''
    Строит ребра графа и их описания, сам граф это набор ребер, без явного указания вершин.
    grid - исходная сетка
    start_point, end_point - начальная, конечная точка соответственно 
    eps - используется для определния ближайшей точки в сетке, к начлаьной и конечной, 
    смещение может произойти когда добавочное к длине большой дуге расстояние не кратно шагу сетки

    tao0 - начальное значение феромонов на ребрах
    speed_degree - один из коэффициентов влияния скорости на стоимость прохода по ребру
    weather_degree - один из коэффициентов влияния погоды на стоимость прохода по ребру
    dist_degree - один из коэффициентов влияния расстония на стоимость прохода по ребру
    
    max_cost - максимальное значение скорости при проходу по ребру, 
    нужно для ограничения при движении по ребру до конечной точке

    dist_step - определяет условие соединения точек в сетку по принципу:
    точки в сетке (x1, y1) и (x2, y2) соединены ребром, если max(abs(x2 - x1), abs(y2 - y1)) < dist_step

    can_diag_move - определяет возможность движения по диагонали в сетке
    weather_data - погодные данные
    type_work - стратегия движения, определяет скорость и расход топлива при движении по ребрам
    '''
    start_index = (-1, -1)
    end_index = (-1, -1)
    dist_start2end = haversine(start_point.coords[0][1], start_point.coords[0][0], end_point.coords[0][1], end_point.coords[0][0])
    # Создадим граф как набор ребер и его описание
    graph = []
    desript_graph = {}
    # Сначала находим положение начальной и конечной точки проверяя все точки
    for i in tqdm(range(len(grid))):
        for j in range(len(grid[0])):
            curr_point = grid[i][j]
            if not Ocean_map.contains(Point(curr_point)):
                continue
            desript_graph[(i, j)] = {}
            
            if abs(curr_point[0] - start_point.coords[0][0]) < eps and abs(curr_point[1] - start_point.coords[0][1]) < eps:
                start_index = (i, j)
                
            if abs(curr_point[0] - end_point.coords[0][0]) < eps and abs(curr_point[1] - end_point.coords[0][1]) < eps:
                end_index = (i, j)
                
    # Затем зная расположения начальной и конечной точки описываем граф
    for i in tqdm(range(len(grid))):
        for j in range(len(grid[0])):
            # Фиксируем превую точку
            point1 = grid[i][j]
            # Проверяем что она находится в рабочей области
            if not Ocean_map.contains(Point(point1)):
                continue
            # В описании графа добавляем ребра с началом в этой точке
            desript_graph[(i, j)] = {}
            # Выбираем погодные данные
            curr_weather_data = gpd_weather_data.loc[:7][gpd_weather_data.loc[:7]['polygon'].contains(Point(point1))] 
            if len(curr_weather_data) == 0:
                    continue
            # Фиксируем погодные условия в этой точке
            BF = curr_weather_data['BF'].values[0]

            # Строим ребра с ближайшими в графе точками
            # Если нельзя ходить по диагонялям (понижает вероятнотность зациклиться и застрять но и вариативность ниже)
            if not can_diag_move:
                posib_steps = [(val, 0) for val in range(-dist_step, dist_step+1)] + [(0, val) for val in range(-dist_step, dist_step+1)]
            else:
                posib_steps = []
                for step_i in range(-dist_step, dist_step + 1):
                    for step_j in range(-dist_step, dist_step + 1):
                        posib_steps.append((step_i, step_j))
            for step_i, step_j in posib_steps:
                if step_i == step_j == 0 or i + step_i < 0 or j + step_j < 0 or i + step_i >= len(grid) or j + step_j >= len(grid[0]):
                    continue
                # Фиксируем конечную точку
                point2 = grid[i + step_i][j + step_j]
                # Строим линию соединияющую первую и вторую точку и проверяем что она не пересекает сушу.
                line = LineString([point1, point2])
                if not Ocean_map.contains(line):
                    continue
                # Добавляем в граф эту линию
                graph.append(line)
            
                # Получаем скорость
                if curr_time_now is None and weather_data is None:
                    speed, fuel = curr_speed_and_fuel(
                        point1, np.array([
                            point2[0] - point1[0], 
                            point2[1] - point1[1], 
                        ]), type_work = type_work,
                    )
                elif curr_time_now is None:
                    speed, fuel = curr_speed_and_fuel(
                        point1, np.array([
                            point2[0] - point1[0], 
                            point2[1] - point1[1], 
                        ]), 
                        type_work = type_work,
                        weather_data = weather_data,
                    )
                elif weather_data is None:
                    speed, fuel = curr_speed_and_fuel(
                        point1, np.array([
                            point2[0] - point1[0], 
                            point2[1] - point1[1], 
                        ]), 
                        type_work = type_work,
                        curr_time = curr_time_now,
                    )
                else:
                    speed, fuel = curr_speed_and_fuel(
                        point1, np.array([
                            point2[0] - point1[0], 
                            point2[1] - point1[1], 
                        ]), 
                        type_work = type_work,
                        weather_data = weather_data,
                        curr_time = curr_time_now,
                    )
                
                # Ищем расстояние до конечной точки
                dist_to_end = haversine(end_point.coords[0][1], end_point.coords[0][0], point2[1], point2[0]) + 1e-6
                dist_edge = haversine(point1[1], point1[0], point2[1], point2[0])
                time_edge = dist_edge / speed
                fuel_edge = time_edge * fuel / 24
                # Оцениваем стоимость (чем выше тем больше вероятность пройти по ребру)
                cost = (
                    speed**speed_degree / 
                    (((dist_to_end / dist_start2end)**dist_degree) * BF * (fuel_edge / 1.5))
                )
                if cost < 0:
                    print(f'point2 : {point2}, \nstart_index : {start_index} \ndist_to_end : {dist_to_end} \ndist_start2end : {dist_start2end} \nBF : {BF}')
                # Ограничиваем стоимость сверху
                if cost > max_cost:
                    cost = 75
                # Создаем полное описание графа
                desript_graph[(i, j)][(i + step_i, j + step_j)] = {
                    'feromon' : tao0,
                    'speed' : speed,
                    'BF' : BF,
                    'fuel' : fuel_edge,
                    'distance' : dist_edge,
                    'cost' : cost
                # Чем больше скорость тем лучше, чем меньше расстояние ло конечной точки, тем лучше, чем меньше BF тем лучше
                }

    return graph, desript_graph, start_index, end_index 
