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

def create_wethaer_map(
    start_point = start_point, 
    end_point = end_point,
    file_path_wave = '/kaggle/input/weathedata/january_2020.grib',
    file_path_wind = '/kaggle/input/weathedata/wind_january_2020.grib',
    step_x = 10, step_y = 10,
    step = 13
):
    min_x, min_y, max_x, max_y =  LineString([start_point, end_point]).bounds
    
    max_x = np.round(max_x + step_x) + 1
    max_y = np.round(max_y + step_y) + 1
    min_x = np.round(min_x - step_x) - 1
    min_y = np.round(min_y - step_y) - 1
   
    wave_dataset = xr.open_dataset(file_path_wave, engine='cfgrib')
    
    wind_dataset = xr.open_dataset(file_path_wind, engine='cfgrib')

    # Преобразуем гриб в датафрейм данные о движении волн
    weather_data = wave_dataset.to_dataframe().reset_index()

    # Отберем только нужные точки и нужые колоник
    weather_data = weather_data[
    (min_y <= weather_data['latitude']) & 
    (weather_data['latitude'] <= max_y) &
    (min_x + 180 <= weather_data['longitude']) &
    (weather_data['longitude'] <= max_x + 180) 
    ].drop(columns = ['time', 'step', 'number', 'depthBelowSeaLayer', 'mswpt300m', 'oceanSurface', 'zos', 'sithick'])

    # Обновим индексацию и добавим колонку
    weather_data.index = range(len(weather_data))
    weather_data['point'] = weather_data.apply(lambda x: Point(x['longitude'] - 180, x['latitude']), axis = 1)

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
        if date in weather_data['valid_time'].unique():
            full_valid_time.add(date)
    
    for date in weather_data['valid_time'].unique():
        if date in wind_data['valid_time'].unique():
            full_valid_time.add(date)

    # Выберем только те строки для которых дата имеется в обоих датафреймах
    wind_data = wind_data[wind_data['valid_time'].apply(lambda x : x in full_valid_time)]
    weather_data = weather_data[weather_data['valid_time'].apply(lambda x : x in full_valid_time)]
    # Соеденим две таблицы
    weather_data = pd.merge(weather_data, wind_data, on=['latitude', 'longitude', 'valid_time'], how='inner')

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
        else: return 12
    # Добавим число Баффорта
    weather_data['BF'] = weather_data['wind_speed'].apply(lambda x: BF_func(x))

    # Для отрисовки преобразуем точки в квадраты
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

    # Сделаем колонки с квадратами
    weather_data['polygon'] = weather_data['point'].apply(lambda x: point2polygon(x))
    
    return weather_data
