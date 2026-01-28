import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

medical_data = {
    'age': [25, 30, 35, 40, 45, 50],
    'blood_pressure': [120, 125, 130, 140, 135, 150],
    'cholesterol': ['high', 'medium', 'low', 'high', 'medium', 'low'],
    'smoker': ['yes', 'no', 'no', 'yes', 'no', 'yes'],
    'bmi': [22.0, np.nan, 27.0, np.nan, 30.0, 24.0],
    'diagnosis': [0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(medical_data)


def get_age_group(age):
    if age < 30:
        return 'young'
    elif age < 50:
        return 'middle'
    else:
        return 'old'


df['age_group'] = df['age'].apply(get_age_group)

# Разделение
train_df = df.iloc[:4]
test_df = df.iloc[4:]

X_train = train_df.drop('diagnosis', axis=1)
y_train = train_df['diagnosis']
X_test = test_df.drop('diagnosis', axis=1)
y_test = test_df['diagnosis']

# Препроцессинг
numeric_cols = ['age', 'blood_pressure', 'bmi']
categorical_cols = ['cholesterol', 'smoker', 'age_group']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Конвертация в тензоры
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print(f"Размерность X_train: {X_train_tensor.shape}")
print(f"Размерность y_train: {y_train_tensor.shape}")

import numpy as np

# Наши исходные тренировочные данные
print("Исходные тренировочные данные:")
print(train_df)


# Функция для создания синтетических пациентов
def create_synthetic_patients(original_df, n_synthetic=20):
    """
    Создаёт синтетических пациентов на основе существующих
    Добавляет небольшой шум к числовым признакам
    """
    synthetic_data = []

    for _ in range(n_synthetic):
        # Выбираем случайного реального пациента
        idx = np.random.randint(0, len(original_df))
        patient = original_df.iloc[idx].copy()

        # Добавляем шум к числовым признакам
        if np.random.random() > 0.3:  # 70% chance to add noise
            # Небольшое изменение возраста (±5 лет)
            if 'age' in patient:
                patient['age'] += np.random.randint(-5, 6)
                patient['age'] = max(20, min(80, patient['age']))  # границы

            # Небольшое изменение давления (±10 мм)
            if 'blood_pressure' in patient:
                patient['blood_pressure'] += np.random.randint(-10, 11)
                patient['blood_pressure'] = max(90, min(200, patient['blood_pressure']))

        # Иногда меняем категориальные признаки
        if np.random.random() > 0.8:  # 20% chance
            if 'smoker' in patient:
                patient['smoker'] = 'yes' if patient['smoker'] == 'no' else 'no'

        synthetic_data.append(patient)

    return pd.DataFrame(synthetic_data)


# Создаём синтетические данные
print("\nСоздаём синтетические данные...")
synthetic_train = create_synthetic_patients(train_df, n_synthetic=50)
print(f"Создано {len(synthetic_train)} синтетических пациентов")

# Объединяем с оригинальными
augmented_train = pd.concat([train_df, synthetic_train], ignore_index=True)
print(f"Всего тренировочных данных: {len(augmented_train)}")

# Пересоздаём age_group для новых данных
augmented_train['age_group'] = augmented_train['age'].apply(get_age_group)

# Проверяем распределение
print("\nРаспределение диагнозов в расширенных данных:")
print(augmented_train['diagnosis'].value_counts())


# 1. Раздели X и y для расширенных данных
X_augmented = augmented_train.drop('diagnosis', axis=1)
y_augmented = augmented_train['diagnosis']

# 2. Примени препроцессор (используй уже обученный на оригинальных данных!)
# Важно: transform, а не fit_transform!
X_augmented_processed = preprocessor.transform(X_augmented)

# 3. Конвертируй в тензоры
X_augmented_tensor = torch.tensor(X_augmented_processed, dtype=torch.float32)
y_augmented_tensor = torch.tensor(y_augmented.values, dtype=torch.float32).view(-1, 1)

print(f"\nРасширенные данные:")
print(f"X shape: {X_augmented_tensor.shape}")
print(f"y shape: {y_augmented_tensor.shape}")