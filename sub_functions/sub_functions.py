import math

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
