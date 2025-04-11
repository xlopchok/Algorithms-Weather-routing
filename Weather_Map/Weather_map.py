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

import xarray as xr

config = {
    'start_point' : Point((3, 55)),
    'end_point' : Point((-80, 15)),
    'file_path_wave' : 'january_2020.grib',
    'file_path_wind' : 'wind_january_2020.grib',
    'step_x' : 10,
    'step_y' : 10,
    'step' : 13,
}

def create_wethaer_map(
    start_point, 
    end_point,
    file_path_wave = 'january_2020.grib',
    file_path_wind = 'wind_january_2020.grib',
    step_x = 10, 
    step_y = 10,
    step = 13,
):
    '''
    Для ограничения рабочей области создается минимальный прямоугольник содержащий начальную и конечну точки
    step_x, step_y - расширяются этот прямоугольник

    Для заполнения пропусков, используются среднии значения из области, определяемой параметром
    step - определяет область из которой берутся значения для заполнения пропусков
    '''
    current_dir = os.path.dirname(__file__)
    file_path_wave = os.path.join(current_dir, 'weather_data', 'file_path_wave')

    current_dir = os.path.dirname(__file__)
    file_path_wave = os.path.join(current_dir, 'weather_data', 'file_path_wind')
    
    min_x, min_y, max_x, max_y =  LineString([start_point, end_point]).bounds
    
    max_x = np.round(max_x + step_x) + 1
    max_y = np.round(max_y + step_y) + 1
    min_x = np.round(min_x - step_x) - 1
    min_y = np.round(min_y - step_y) - 1
   
    wave_dataset = xr.open_dataset(file_path_wave, engine='cfgrib')
    
    wind_dataset = xr.open_dataset(file_path_wind, engine='cfgrib')

    # Преобразуем гриб в датафрейм данные о движении волн
    wave_data = wave_dataset.to_dataframe().reset_index()

    # Отберем только нужные точки и нужые колоник
    wave_data = wave_data[
    (min_y <= wave_data['latitude']) & 
    (wave_data['latitude'] <= max_y) &
    (min_x + 180 <= wave_data['longitude']) &
    (wave_data['longitude'] <= max_x + 180) 
    ].drop(columns = ['time', 'step', 'number', 'depthBelowSeaLayer', 'mswpt300m', 'oceanSurface', 'zos', 'sithick'])

    # Обновим индексацию и добавим колонку point
    wave_data.index = range(len(wave_data))
    wave_data['point'] = wave_data.apply(lambda x: Point(x['longitude'] - 180, x['latitude']), axis = 1)

    # Преобразуем погодные данные о ветре из гриб в датафрейм
    wind_data = wind_dataset.to_dataframe().reset_index()

    # Выберем нужные точки и колонки
    wind_data = wind_data[
    (min_y <= wind_data['latitude']) & 
    (wind_data['latitude'] <= max_y) &
    (min_x + 180 <= wind_data['longitude']) &
    (wind_data['longitude'] <= max_x + 180) 
    ].drop(columns = ['time', 'step', 'number', 'surface', 'sp', 'heightAboveGround'])
    
    wind_data.index = range(len(wind_data))


    # Выберем только те даты которые встречаются обоих датафреймах
    full_valid_time = set()
    
    for date in wind_data['valid_time'].unique():
        if date in wave_data['valid_time'].unique():
            full_valid_time.add(date)
    
    for date in wave_data['valid_time'].unique():
        if date in wind_data['valid_time'].unique():
            full_valid_time.add(date)

    # Выберем только те строки для которых дата имеется в обоих датафреймах
    wind_data = wind_data[wind_data['valid_time'].apply(lambda x : x in full_valid_time)]
    wave_data = wave_data[wave_data['valid_time'].apply(lambda x : x in wave_data)]
    # Соеденим две таблицы
    weather_data = pd.merge(wave_data, wind_data, on=['latitude', 'longitude', 'valid_time'], how='inner')

    # Уберем пропуски 
    while weather_data['uoe'].isna().sum() != 0 and weather_data['von'].isna().sum() != 0:
        for idx in range(len(weather_data)):
            row = weather_data.loc[idx]
            for col in ['uoe', 'von']:
                if np.isnan(row[col]):
                    weather_data.loc[idx, col] = weather_data[
                        (weather_data['latitude'] <= row['latitude'] + step) &
                        (weather_data['latitude'] >= row['latitude'] - step) &
                        (weather_data['longitude'] <= row['longitude'] + step) &
                        (weather_data['longitude'] >= row['longitude'] - step) &
                        (-weather_data[col].isna())
                    ][col].mean()
        
    
    # Добавим скорость движения воды:
    weather_data['water_speed'] = weather_data[['uoe', 'von']].apply(lambda x: (x['uoe']**2 + x['von']**2)**0.5, axis = 1)
    weather_data['wind_speed'] = weather_data[['u10', 'v10']].apply(lambda x: (x['u10']**2 + x['v10']**2)**0.5, axis = 1)
    
    def BF_func(x):
        thresholds = [0.3, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
        thresholds = [1 if x <= val else 0 for val in thresholds]
        if 1 in thresholds:
            return thresholds.index(1)
        else: 
            return 12
            
    # Добавим число Баффорта
    weather_data['BF'] = weather_data['wind_speed'].apply(lambda x: BF_func(x))

    # Для отрисовки преобразуем точки в квадратные полигоны
    def point2polygon(point, step = 1.5):
        xy = point.coords[0]
        polygon = Polygon([
            [xy[0]-step, xy[1]-step],
            [xy[0]+step, xy[1]-step],
            [xy[0]+step, xy[1]+step],
            [xy[0]-step, xy[1]+step],
            [xy[0]-step, xy[1]-step],
        ])
        return polygon

    # Сделаем колонки с квадратными полигонами
    weather_data['polygon'] = weather_data['point'].apply(lambda x: point2polygon(x))
    
    return weather_data

def safe_unary_union(geometries):
    '''
    Функция для объединения геометрий с использованием современных методов Shapely
    Это позволяет вместо отрисовки большого числа объектов, отображать один объединненый объект
    Также упрощает определния некоторых погдных параметров, например число Бофорта
    '''
    # Создаем список геометрий
    geometries_list = [geom for g in geometries for geom in (g.geoms if hasattr(g, 'geoms') else [g])]
    
    # Объединяем их с помощью unary_union
    res = unary_union(geometries_list)
    if not res.is_valid:
        res = make_valid(res)
    return res

weather_data = create_wethaer_map(**config)

# Группировка и объединение
group_by_BF_time_weather_data = (
    weather_data.groupby(['valid_time', 'BF'])
    .agg({'polygon': lambda x: safe_unary_union(x)})
    .reset_index()
)

gpd_weather_data = gpd.GeoDataFrame(
    group_by_BF_time_weather_data,
    geometry = 'polygon',
    crs = 'EPSG:4326'
)

def visual_weather_data_by_BF_and_time(
    gpd_weather_data = gpd_weather_data,
    map_file = 'BF_weather.html'
):
    FULL_GPD = []
    for time_data in gpd_weather_data['valid_time'].unique():
        GPD = gpd_weather_data[gpd_weather_data['valid_time'] == time_data].copy()
        
        GPD.index =  pd.Series(range(len(GPD)))
        
        GPD['valid_time_str'] = GPD['valid_time'].dt.strftime('%Y-%m-%d')
        
        GPD.drop('valid_time', axis = 1, inplace = True)
        
        GPD['cat'] = pd.Series(range(len(GPD)))
    
        FULL_GPD.append(GPD)

    m = folium.Map(tiles="cartodbpositron", world_copy_jump=True)
    
    group_1 = folium.FeatureGroup("first group").add_to(m)
    folium.GeoJson(Ocean_map).add_to(group_1)
    
    # Добавление слоя Choropleth
    for i, GPD in enumerate(FULL_GPD):
        folium.Choropleth(
            geo_data=GPD.to_json(),
            name = f'{i} weather map',
            data=GPD,
            columns=['cat', 'BF'],  # Используем колонку с временными метками
            key_on='feature.id',  # Сопоставляем ID геометрий
            fill_color='RdBu_r',
            fill_opacity=0.8,
            line_opacity=0.2,
            # legend_name='Beaufort Force (BF)',
            show = False,
            highlight = True,
            
        ).add_to(m)

    folium.Marker((start_point.coords[0][1], start_point.coords[0][0]), tooltip="start_point").add_to(m)
    folium.Marker((end_point.coords[0][1], end_point.coords[0][0]), tooltip="end_point").add_to(m)
    
    folium.LayerControl().add_to(m)
    
    MousePosition().add_to(m)

    m.save(map_file)

def create_extrimly_weather_data(weather_data = weather_data):
    extrimly_weather_data['BF'] = weather_data['BF'].map(lambda x : x if x < 5 else 7)
    return extrimly_weather_data

extrimly_weather_data = create_extrimly_weather_data()

if __name__ == 'main':
    visual_weather_data_by_BF_and_time()
    visual_weather_data_by_BF_and_time(weather_data = extrimly_weather_data, map_file = 'extrimly_BF_weather.html')
