import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch

# 1. Создаем данные
data = [
    [1.0, 2.0, 3.0, 'cat'],
    [4.0, 5.0, 6.0, 'dog'],
    [7.0, 8.0, 9.0, 'cat'],
    [10.0, 11.0, 12.0, 'bird'],
    [13.0, 14.0, 100., 'fish'],
]
df = pd.DataFrame(data, columns=['a', 'b', 'c', 'animal'])

print("Исходные данные:")
print(df)

# 2. Определяем типы колонок
numeric_features = ['a', 'b', 'c']
categorical_features = ['animal']

# 3. Создаем ColumnTransformer (главный инструмент!)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 4. Обучаем и преобразуем
X_processed = preprocessor.fit_transform(df)
X_new_processed = preprocessor.transform(df)
print(f"\nФорма после обработки: {X_processed.shape}")
print("\nОбработанные данные:")
print(X_processed)

# 5. Конвертируем в тензор PyTorch
X_tensor = torch.tensor(X_processed, dtype=torch.float32)
print(f"\nТензор PyTorch shape: {X_tensor.shape}")
print(X_tensor)