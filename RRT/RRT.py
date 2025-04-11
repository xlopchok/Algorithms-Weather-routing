import os
import sys
import math

import pandas as pd
import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd

import networkx as nx

import folium
from folium.plugins import MousePosition
from folium import IFrame

import shapely
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry import LinearRing
from shapely.ops import unary_union
from shapely.validation import make_valid

from maps.Ocean_map import Ocean_map, projection_to_boundary
from sub_functions.subfunctions import haversine, smoothing_path

config = {
    'start_point' : Point((3, 55)),
    'end_point' : Point((-80, 15)),
    'Ocean_map' : Ocean_map,
    'r' : 0.3,
    'random_r' : True,
    'sub_radius' : 0.15,
    'verbose' : -1,
    'count_points' : 1000,
    'count_pathes' : 1,
    'max_edge_dist' : 300,
}

class RRT:
    def __init__(
        self, 
        start_point, 
        end_point, 
        Ocean_map,
        step_x = 10,
        step_y = 10,
        r = 0.7,
        random_r = False,
        sub_radius = 0.2
    ):
        '''
        start_point, start_point - начальная и конечная точки в формате : [lon, lat];
        Ocean_map - рабочая область;

        В процессе алгоритма строится минимальный прямоугольник содеражщий начальную и конечную точки, для ограничения рабочей области
        step_x, step_y - расширяют полученный прямоугольник

        При работе алгоритма создается случайная точка и переносится на допустимой расстояние к ближайшей точке
        r - допустимое максимальное расстояние

        sub_radius - используется для поиска ближайшей точке среди имеющихся в графе, к новой случайной. 
        Чем больше sub_radius, тем больше точек будет точек в множестве ближаших, при маломо значении может не найти ближайшую, из-за вычислительных ошибок

        random_r - если установлен в True, ранее заданный r, может выбираться случайено, такой подход повышает вариативность
        '''
        self.Ocean_map = Ocean_map
        self.start_point = start_point
        self.end_point = end_point
        
        if not self.Ocean_map.contains(self.start_point):
            self.start_point = projection_to_boundary(self.start_point, Map = self.Ocean_map)
        if not self.Ocean_map.contains(self.end_point):
            self.end_point = projection_to_boundary(self.end_point, Map = self.Ocean_map)
        
        end_point_left = Point((self.end_point.coords[0][0] - 360, self.end_point.coords[0][1]))
        end_point_right = Point((self.end_point.coords[0][0] + 360, self.end_point.coords[0][1]))
        
        curr_dist = self.start_point.distance(self.end_point)
        
        if self.start_point.distance(end_point_right) < curr_dist:
            self.end_point = end_point_right
        elif self.start_point.distance(end_point_left) < curr_dist:
            self.end_point = end_point_left
        
        self.r = r
        self.random_r = random_r
        self.sub_radius = sub_radius

        # Создадим граф как набор ребер MultiLineString, но пусть он будет просто массивом из ребер
        self.graph = []
    
        # Будем запоминать точки которые используем чтобы найти
        self.all_points = []

        #Создадим путь
        self.path = None

        # Выделим рабочее прострнство:
        self.start, self.end = np.array(self.start_point.coords[0]), np.array(self.end_point.coords[0])
    
        self.all_points.append(self.start)
    
        min_x, min_y, max_x, max_y =  LineString([self.start, self.end]).bounds
    
        max_x += step_x
        max_y += step_y
        min_x -= step_x
        min_y -= step_y
    
        self.work_space = Polygon([
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
            [max_x, min_y]
        ])

    def add_new_point_to_RRT(self):
        min_x, min_y, max_x, max_y =  self.work_space.bounds
        ''' Создаем случайную точку '''
        new_x = np.random.uniform(low = min_x, high = max_x + 1)
        new_y = np.random.uniform(low = min_y, high = max_y + 1)
    
        new_point = np.array([new_x, new_y])
        
        ''' Найдем для нее ближашую точку в графе'''
        # Найдем расстояние до ближайшей
        distance = Point(new_point).distance(MultiPoint(self.all_points)) + self.sub_radius
        # Создадим область в которой лежит ближайшая точка
        circle = Point(new_point).buffer(distance)
        # Найдем пересечение область и всех точек
        nearest_points = circle.intersection(MultiPoint(self.all_points))
        
        # Если в результате пересечения нашли только одну точку то она и есть ближайшая
        if nearest_points.geom_type == 'Point':
            nearest_point = np.array(nearest_points.coords[0])
        # Иначе среди выбраных точек найдем ближайшую (так можно было бы проверить все точки, но это заняло бы слишком много времени при большом кол-ве точек)
        else:
            min_distance = float('inf')
            nearest_point = nearest_points.geoms[0].coords[0]
            for point in nearest_points.geoms:
                curr_dist = point.distance(Point(new_point))
                if min_distance > curr_dist:
                    min_distance = curr_dist
                    nearest_point = np.array(point.coords[0])
        '''
        Теперь перенесем новую точку, так чтобы она осталась на отрезке соединяющем новую и ближайшую точки
        но при этом была на заданном расстоянии r от новой точки
        '''
        curr_dist = Point(new_point).distance(Point(nearest_point))
        
        if curr_dist > self.r:
            vector = new_point - nearest_point
            # Добавим элемент случаности в выбор расстояния до новой точки
            if self.random_r:
                vector = np.random.uniform(self.r / 5, self.r * 5) * vector / np.linalg.norm(vector)
            else:
                vector = self.r * vector / np.linalg.norm(vector)
            new_point = nearest_point + vector
    
        # После правильного расположения новой точки, проверим что отрезок соединяющий их полностью лежит в Ocean_map
        line = LineString([nearest_point, new_point])
        if self.Ocean_map.contains(line):
            self.all_points.append(new_point)
            self.graph.append(np.array(line.coords))

        # Проверим расстояние до конченой точки
        curr_dist = Point(new_point).distance(Point(self.end))
        line = LineString([new_point, self.end])
        flag = False
        if curr_dist <= 10 * self.r and self.Ocean_map.contains(line):
            self.all_points.append(self.end)
            self.graph.append(np.array(line.coords))
            flag = True
            
        return flag

    def RRT_algorithm(self, count_points = 10000, verbose = 1500):
        '''
        Функция реализует алгоритм создания графа

        count_points - ограничивает максимальное значение точек в графе
        verbose - если >0, то это шаг с которым воводится кол-во точек в полученном графе
        '''
        epoch = 0
        while len(self.all_points) < count_points:
            epoch += 1
            flag = self.add_new_point_to_RRT()
            if flag:
                break
            if verbose > 0 and epoch % verbose == 0:
                print(len(self.all_points))

    def RRT_find_path_in_graph(self):
        '''
        Функция для поиска маршрута в полученном графе

        Все точки в графе, задаются в порядке [lon, lat], но
        путь возвращается в формате последовательности точек [lat, lon]

        Это сделано для удобства визуализации
        '''
        multilines = MultiLineString(self.graph)

        # Создаем граф
        G = nx.Graph()
        
        # Добавляем ребра в граф
        for line in multilines.geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                G.add_edge(coords[i], coords[i + 1], weight=line.length)
        
        # Определяем начальную и конечную точки
        start = tuple(self.start_point.coords[0])
        end = tuple(self.end_point.coords[0])
        
        # Находим кратчайший путь
        shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
        
        self.path = np.array([[coords[1], coords[0]] for coords in shortest_path])

def visualisations_pathes(
    start_point = start_point, 
    end_point = end_point, 
    Ocean_map = Ocean_map, 
    pathes = pathes, 
    map_file = "rrt_pathes.html"
):
    colors = ['red', 'green', 'blue', 'black', 'purple', 'orange', 'yellow']
    m = folium.Map(tiles="cartodbpositron")
    
    folium.GeoJson(Ocean_map).add_to(m)

    folium.Marker(start_point.coords[0][::-1], tooltip="inside_point").add_to(m)
    folium.Marker(end_point.coords[0][::-1], tooltip="outside_point").add_to(m)
    
    for i, path in enumerate(pathes):
        folium.PolyLine(path, color = colors[i % len(colors)], weight = 5).add_to(m)
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)
    m.save(map_file)
    
def visualisations_graph_and_path(
    rrt, 
    start_point = start_point, 
    end_point = end_point, 
    Ocean_map = Ocean_map,
    map_file = 'RRT_path.html'
):
    m = folium.Map(tiles="cartodbpositron")
    
    folium.GeoJson(Ocean_map).add_to(m)
    
    folium.Marker(start_point.coords[0][::-1], tooltip="inside_point").add_to(m)
    folium.Marker(end_point.coords[0][::-1], tooltip="outside_point").add_to(m)
    
    folium.GeoJson(MultiLineString(rrt.graph)).add_to(m)
    folium.PolyLine(rrt.path, color = 'red', weight = 5).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)
    m.save(map_file)
    
if __name__ == "__main__":
    pathes = []
    for i in range(config['count_pathes']):
        rrt = RRT(
          config['start_point'], 
          config['end_point'], 
          Ocean_map = config['Ocean_map'], 
          r = config['r'], 
          random_r = config['random_r'], 
          sub_radius = config['sub_radius']
        )
        rrt.RRT_algorithm(
          count_points = config['count_points'],
          verbose = config['verbose']
        )
        
        rrt.RRT_find_path_in_graph()
        res_path = rrt.path
        np.save(f'{i+1}_rrt_path.npy', res_path)

        visualisations_graph_and_path(
            rrt, 
            start_point = start_point, 
            end_point = end_point, 
            Ocean_map = Ocean_map,
            map_file = f'{i + 1}_RRT_path.html'
        )
        pathes.append(res_path)
        
    visualisations_pathes(
        start_point = config['start_point'],
        end_point = config['end_point'],
        Ocean_map = config['Ocean_map'], 
        pathes, 
        map_file = 'rrt_pathes_before_smoothing.html'
    )
    
    for i, path in enumerate(pathes):
        new_path = smoothing_path(path, max_edge_dist = config['max_edge_dist'], Ocean_map = Ocean_map)
        pathes[i] = new_path
        
    visualisations_pathes(
        start_point = config['start_point'],
        end_point = config['end_point'],
        Ocean_map = config['Ocean_map'], 
        pathes, 
        map_file = 'rrt_pathes_after_smoothing.html'
    )
                          
    
