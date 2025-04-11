import math

from maps.Ocean_map import Ocean_map

# Функция для вычисления расстояния с использованием формулы Haversine
def haversine(lat1, lon1, lat2, lon2, unit='km'):
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

def smoothing_path(path, max_edge_dist = 300, Ocean_map = Ocean_map):
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
