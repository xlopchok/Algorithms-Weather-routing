import math

from maps.Ocean_map import Ocean_map
from Weather_Map.Weather_map import weather_data, gpd_weather_data, extrimly_weather_data

def smoothing_path(path, max_edge_dist = 300, Ocean_map = Ocean_map):
    '''
    Функция сглаживает маршрут, удаляя не нужные точки
    Уменьшает кол-во поворотов и избавляется от циклов
    '''
    path = [point[::-1] for point in path] # Возвращаем маршрут в последовательность точек вида [lon lat]
    new_path = []
    curr_index = 0
    while curr_index < len(path):
        point1 = path[curr_index]
        new_path.append(point1)
        best_next_index = -1
        best_dist = -1
        for next_index in range(curr_index + 1, len(path)):
            point2 = path[next_index]
            line = LineString([point1, point2])
            if Ocean_map.contains(line):
                dist_coords = next_index - curr_index
                distance = haversine(point1[1], point1[0], point2[1], point2[0])
                if distance < max_edge_dist and dist_coords > best_dist:
                    best_dist = dist_coords
                    best_next_index = next_index
        if best_next_index == -1:
            curr_index += 1
        else:
            curr_index = best_next_index
    new_path = [point[::-1] for point in new_path] # Возвращаем маршрут в последовательность точек вида [lat lon]
    return new_path

def curr_speed_and_fuel(
    point, 
    displacement_vector, 
    curr_time = weather_data['valid_time'].unique()[0],
    design_speed = 27.8, 
    weather_data = weather_data, 
    L = 366, 
    f = 36, 
    DWT = 120_000,
    pho = 1025,
    C_B = 0.7,
    type_work = 'cost_fuel',
    per_speed = 0.5, # Процент скорости в случае падения ниже которой мощность двигателя увеличивается
    per_fuel = 3, # максимальное значение на сколько можно поднять расход топлива
    max_loss_val = 0.75,
):
    '''
    Возвращает скорость в км/ч и рассход топлива в тонн/сутки 
    При движении из выбранной точки в выбранном направлении
    point - выбранная точка
    displacement_vector - направление движения, нужно для определния скоости судна относительно течения
    curr_time - текущий момент времени (или ближайший к текущему, который есть в данных)
    design_speed - расчетная скорость
    weather_data - погодные данные
    L - длина судна
    f - расчетный расход топлива в тонн/сутки
    DWT - одна из характеристик корабля
    pho - плотность мосркой воды
    C_B - коэффициент определяемый как отношения объемов погруженной части судна к минимальному параллепиду в который такая часть поместится, 
    зависит от нагруженности или заполненности судна
    type_work - одна из 4х стратегий движения
    per_speed - в одном из сценариев движения ограничивает минимальное значение скорости
    per_fuel - в одном из сценариев ограничивает максимальный допустимый расход топлива
    max_loss_val - ограничивает максимальный процент потерь скорости, является допущением, упрощающим стратегию движения,
    на практике max_loss_val = 1, то есть рассматривает случай полной остановки судна
    '''
    type_of_work = [
        'const_speed',
        'const_fuel',
        'max_persent_speed',
        'max_persent_fuel',
    ]
    if type_work not in type_of_work:
        type_work = type_of_work[1]
    
    if type(point) != shapely.geometry.point.Point:
        point = Point(point.copy())

    data_for_point = weather_data[weather_data['valid_time'] == curr_time]
    data_for_point = data_for_point[data_for_point['polygon'].apply(lambda x: x.contains(point))]

    # Для соответсвия размерностей все данные приведм к метрам и секундам
    # на случай если точка окажется на границе двух полигонов
    if len(data_for_point) != 0:
        data_for_point = data_for_point.iloc[0]
        
        V_of_sea = np.array([data_for_point['uoe'], data_for_point['von']])
        V_of_sea = data_for_point['water_speed'] * V_of_sea / norm(V_of_sea) if norm(V_of_sea) != 0 else 0
    
        # print(f'speed of sea = {np.linalg.norm(V_of_sea)}')    
        V_relative_current = design_speed * displacement_vector / (3.6 * norm(displacement_vector)) - V_of_sea
        # print(f'speed of realative current = {np.linalg.norm(V_relative_current)}')
        Fn = norm(V_relative_current) / (np.sqrt(9.8 * L))

        alpha = 3.1 - 5.3 * Fn - 12.4 * (Fn ** 2)
        if alpha < 0:
            alpha = 0


        cos = V_of_sea @ displacement_vector / (np.linalg.norm(V_of_sea) * np.linalg.norm(displacement_vector))
        if abs(cos) > 1:
            cos = np.sign(cos)
            
        angel = np.arccos(cos) * 180 / np.pi
        
        mu = None
        if angel < 60:
            mu = (1.7 - 0.03 * (data_for_point['BF'] - 4) ** 2) / 2
        elif angel < 150:
            mu = (0.9 - 0.06 * (data_for_point['BF'] - 6) ** 2) / 2
        else:
            mu = (0.4 - 0.03 * (data_for_point['BF'] - 8) ** 2) / 2
        if mu < 0:
            mu = 0

        real_V = None
        # print(f'\nmu = {mu}, \nalpha = {alpha} \nFn = {Fn}\n\n')
        if alpha == 0 or mu == 0:
            real_V = design_speed / 3.6 - ((design_speed / 3.6) * (0.5 * data_for_point['BF'] + (data_for_point['BF']**(6.5)) / (22 * (C_B * DWT)/pho))) / 100
        else:
            V_loss = alpha * mu * (0.5 * data_for_point['BF'] + (data_for_point['BF']**(6.5)) / (22 * (C_B * DWT)/pho)) / 100
            # BF = data_for_point['BF']
            # print(f'\nBF = {BF}\n')
            # print(f'\nV loss = {V_loss}\n')
            if V_loss > 1:
                V_loss = max_loss_val
            real_V = (design_speed / 3.6) * (1 - V_loss)

        ''' 
        Будем возвращать разные значения скорсти и топлива, в зависимости от выбранного режима работы
        '''
        if type_work == 'const_fuel':
            # Не меняем расходов топлива
            return real_V * 3.6, f
            
        elif type_work == 'const_speed':
            real_f = f * (((design_speed / 3.6) / real_V)**3)
            return design_speed, real_f

        elif type_work == 'max_persent_speed':
            # Проверим что скорость упала ниже желаемой
            if real_V * 3.6 >= design_speed * per_speed:
                return real_V * 3.6, f
            else:
                real_f = f * (((per_speed * design_speed / 3.6) / real_V)**3)
                return per_speed * design_speed, real_f
        elif type_work == 'max_persent_fuel':
            # Сначала проверим какой будет расход топлива если поднять скорость до расчетной
            posib_real_f = f * (((design_speed / 3.6) / real_V)**3)
            if posib_real_f / f <= per_fuel:
                return design_speed, posib_real_f
            else:
                # Если же не удается достичь расчетной скорости, то посчитаем какой скорости удастся достичь
                posib_V = per_fuel ** (1/3) * real_V
                return posib_V * 3.6, f * per_fuel
    else:
        return -1, -1
    
# Функция для вычисления расстояния с использованием формулы Haversine
def haversine(lat1, lon1, lat2, lon2, unit='km'):
    '''
    Функция для вычисления расстояния с использованием формулы Haversine
    '''
    # Радиус Земли в зависимости от единицы измерения
    if unit == 'km':
        R = 6371  # Радиус Земли в километрах
    elif unit == 'm':
        R = 6371000  # Радиус Земли в метрах
    
    # Преобразуем градусы в радианы
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Разность широт и долгот
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Формула Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Расстояние
    distance = R * c
    return distance

def destination_point(lat1, lon1, move_vector = None, speed = None, time = None, R=6371, distance = None, angle = None):
    '''
    Возвращает конечную точку при движении из координат
    lat1 lon1, 
    move_vector - направление движения, может быть не задано
    speed - скорость движения, может быть не задано
    time - время движения, может быть не задано
    R - радиус Земли
    distance - расстояние в км
    angle - угол, определяющий нарпавление движения, может быть не задан

    Обязательно задано:
    - одно из двух angle или move_vector для определения направления
    - однин из двух вариантов (speed и time) или (distance) для определния расстояния
    '''
    # Преобразуем начальные координаты в радианы
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)

    if angle is None:
        # Вычисляем азимут (курс) в радианах
        bearing_rad = np.arctan2(move_vector[1], move_vector[0]) 
    else:
        bearing_rad = angle

    if distance is None:
        # Вычисляем угловое расстояние
        delta_sigma = (speed * time) / R
    else:
        delta_sigma = distance / R
    
    # Вычисляем новую широту
    lat2_rad = np.arcsin(np.sin(lat1_rad) * np.cos(delta_sigma) + 
                         np.cos(lat1_rad) * np.sin(delta_sigma) * np.cos(bearing_rad))
    
    # Вычисляем новую долготу
    # lon2_rad = lon1_rad + np.arctan2(np.sin(bearing_rad) * np.sin(delta_sigma) * np.cos(lat1_rad), 
    #                                  np.cos(delta_sigma) - np.sin(lat1_rad) * np.sin(lat2_rad))

    lon2_rad = lon1_rad + np.arcsin((np.sin(bearing_rad) * np.sin(delta_sigma)) / (np.cos(lat2_rad)))

    # Преобразуем обратно в градусы
    lat2 = np.degrees(lat2_rad)
    lon2 = np.degrees(lon2_rad)
    
    return lat2, lon2

def analys_path(
    path,
    Print = False,
    curr_time_now = None,
    type_work = 'const_fuel',
    weather_data = None,
):
    '''
    Анализирует полученный маршрут
    '''
    time = 0
    fuel = 0
    distance = 0
    if type(path) != np.array:
        path = np.array(path)
    for i in range(len(path) - 1):
        curr_distance = haversine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        move_vector = path[i + 1] - path[i]
        if curr_time_now is None:
            curr_speed, curr_fuel = curr_speed_and_fuel(np.array([path[i][1], path[i][0]]), move_vector, type_work = type_work)
        else:
            if weather_data is None:
                curr_speed, curr_fuel = curr_speed_and_fuel(np.array([path[i][1], path[i][0]]), move_vector, curr_time = curr_time_now, type_work = type_work)
            else:
                curr_speed, curr_fuel = curr_speed_and_fuel(
                    np.array([path[i][1], path[i][0]]), 
                    move_vector, curr_time = curr_time_now, 
                    type_work = type_work, weather_data = weather_data)
        curr_time = curr_distance / curr_speed
        curr_fuel = curr_fuel * curr_time / 24

        time += curr_time
        fuel += curr_fuel
        distance += curr_distance
    if Print:
        print(f'res distance = {distance} \nres fuel = {fuel} \nres time = {time}')

    if time >= 0:
        return distance, fuel, time
    else:
        # В случае если подан маршрут в неправильном формате, перепутаны местами lat и lon
        returnanalys_path(
            path = [point[::-1] for point in path],
            Print = False,
            curr_time_now = curr_time_now,
            type_work = type_work,
            weather_data = weather_data,
        )

