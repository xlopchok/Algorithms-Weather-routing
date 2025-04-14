# 📦Algorithms-Weather-routing

Algorithms-Weather-routing — это проект, посвящённый разработке и сравнению алгоритмов построения маршрутов морских судов с учётом погодных условий.
Цель проекта — исследовать, как различные алгоритмы планирования маршрутов могут использовать данные о погоде для построения безопасных и эффективных морских маршрутов.

## 🌊 Описание проекта
В рамках проекта реализованы и сравнены следующие алгоритмы маршрутизации:

- ACO (Ant Colony Optimization) — алгоритм муравьиной колонии

- GA (Genetic Algorithm) — генетический алгоритм

- Dijkstra — алгоритм Дейкстры

- RRT (Rapidly-exploring Random Tree)

- RRT (RRT Star) — улучшенная версия RRT*

- Isochrons method — метод изохрон

Алгоритмы ACO, GA, Dijkstra, Isohrons method использует данные о погодных условиях и карте океана для генерации маршрута с учётом таких факторов, как скорость и направление ветра и течений.

## Структура:
```markdown
📦 project
 ┣ 📂 ACO                  # Алгоритм муравьиной колонии
 ┃ ┣ 📂 result_visual      # Визуализация результатов
 ┃ ┣ 📂 results_pathes     # Сохранённые пути
 ┃ ┗ 📄 ACO.py             # Реализация алгоритма
 ┣ 📂 GA                   # Генетический алгоритм
 ┃ ┣ 📂 result_visual
 ┃ ┣ 📂 results_pathes
 ┃ ┗ 📄 GA.py
 ┣ 📂 Dijkstra_alg         # Алгоритм Дейкстры
 ┃ ┣ 📂 result_visual
 ┃ ┣ 📂 results_pathes
 ┃ ┗ 📄 Dijkstra_alg.py
 ┣ 📂 RRT                  # RRT
 ┃ ┣ 📂 result_visual
 ┃ ┣ 📂 results_pathes
 ┃ ┗ 📄 RRT.py
 ┣ 📂 RRT_star             # RRT*
 ┃ ┣ 📂 result_visual
 ┃ ┣ 📂 results_pathes
 ┃ ┗ 📄 RRT_star.py
 ┣ 📂 isochrons            # Метод изохрон
 ┃ ┣ 📂 result_visual
 ┃ ┣ 📂 results_pathes
 ┃ ┗ 📄 isochrons.py
 ┣ 📂 maps                 # Карта океана
 ┃ ┣ 📄 Ocean_map.html
 ┃ ┣ 📄 Ocean_map.png
 ┃ ┗ 📄 Ocean_map.py
 ┣ 📂 Weather_Map          # Работа с погодными данными
 ┃ ┣ 📂 result_visual
 ┃ ┣ 📂 Weather_Map
 ┃ ┗ 📄 Weather_map.py
 ┣ 📂 sub_finctions        # Вспомогательные функции
 ┃ ┗ 📄 sub_finctions.py
 ┣ 📂 Graph_map            # Генерация графа карты
 ┃ ┗ 📄 generate_graph.py
┣ 📄 README.md             # Документация проекта
...
```

## 📊 Результаты
Результаты выполнения алгоритмов сохраняются в директории result_visual в виде изображений и HTML-файлов для наглядности маршрутов.
Дополнительно пути сохраняются в формате .npy в папке results_pathes для последующего анализа.

## 🧩 Дополнительные модули
- Weather_Map — получение и обработка данных о погоде, визуализация.

- sub_finctions — вспомогательные функции для работы алгоритмов.

- Graph_map — генерация графа карты океана.

## 📖 Дополнительно
- Все алгоритмы реализованы в виде отдельных модулей для удобства тестирования и расширения.
