# 1 Основа таблиц
import pandas as pd
# Создаем таблицу ВРУЧНУЮ, чтобы понять структуру
# Это список списков - каждая строка = один цветок
data = [
    [5.1, 3.5, 1.4, 0.2, 'setosa'],
    [7.0, 3.2, 4.7, 1.4, 'versicolor'],
    [6.3, 3.3, 6.0, 2.5, 'virginica']
]

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


# Дата фрейм pd превращает обычную таблицу формата Excel в спец. таблицу с колоннами из columns
df = pd.DataFrame(data, columns=columns)
print("Моя самая первая таблица")
print(df)


# 2 Параметры

print("Размеры таблицы\n", df.shape, "\n")

print("Типы данных\n", df.dtypes, "\n")

print("Цветок\n", df.iloc[0], "\n") #iloc = index location(обращение по номеру строки)

for i in range(df.shape[0]):
    print("Цветок", i, "\n", df.iloc[i], "\n")

print(df.iloc[0]["sepal_length"])

# 3 Разделение
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

print("Признаки (X):")
print(X)
print("\nЦелевые значения (y):")
print(y)

new_flower = [[5.8, 1.0, 2.5, 3.5]]
new_df = pd.DataFrame(new_flower, columns=X.columns)
print(new_df)

# 4 Математика
import math
def euclidean_distance(row1, row2):
    """Евклидово расстояние между двумя точками"""
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

new_flower_values = new_flower[0]
distances = []
for i in range(len(df)):
    flower_values = df.iloc[i][['sepal_length', 'sepal_width',
                                'petal_length', 'petal_width']].values
    dist = euclidean_distance(new_flower_values, flower_values)
    distances.append((dist, df.iloc[i]['species']))

print("Расстояния до известных цветков")
for dist, species in distances:
    print(f"{species}: {dist:.2f}")

nearest_neighbor = min(distances, key=lambda x: x[0])
print(f"\nБлижайший сосед: {nearest_neighbor[1]} (расстояние: {nearest_neighbor[0]:.2f})")
print(f"Предсказание: новый цветок, вероятно, {nearest_neighbor[1]}")

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X,y)

prediction = model.predict(new_df)