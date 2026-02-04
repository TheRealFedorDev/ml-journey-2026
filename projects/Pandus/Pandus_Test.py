import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch

# 1. Создаём данные с реалистичными пропусками
medical_data = {
    'age': [25, 30, 35, 40, 45, 50],
    'blood_pressure': [120, 125, 130, 140, 135, 150],
    'cholesterol': ['high', 'medium', 'low', 'high', 'medium', 'low'],
    'smoker': ['yes', 'no', 'no', 'yes', 'no', 'yes'],
    'bmi': [22.0, np.nan, 27.0, np.nan, 30.0, 24.0],
    'diagnosis': [0, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(medical_data)

# 2. Добавляем age_group
def get_age_group(age):
    if age < 30:
        return 'young'
    elif age < 50:
        return 'middle'
    else:
        return 'old'

df['age_group'] = df['age'].apply(get_age_group)

# 3. Разделяем на train/test
train_df = df.iloc[:4]  # 0,1,2,3
test_df = df.iloc[4:]   # 4,5

print("Train данные:")
print(train_df)
print("\nTest данные:")
print(test_df)

# 4. Разделяем X и y
X_train = train_df.drop('diagnosis', axis=1)
y_train = train_df['diagnosis']
X_test = test_df.drop('diagnosis', axis=1)
y_test = test_df['diagnosis']

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# 5. Определяем типы колонок
numeric_cols = ['age', 'blood_pressure', 'bmi']
categorical_cols = ['cholesterol', 'smoker', 'age_group']  # ДОБАВИЛИ age_group!

# 6. Создаём препроцессор
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



# 7. Обучаем и преобразуем
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


def show_transformation(original_df, processed_array, preprocessor):
    """Показывает преобразование данных"""
    # Получаем имена признаков после обработки
    feature_names = preprocessor.get_feature_names_out()

    # Создаём DataFrame из обработанных данных
    processed_df = pd.DataFrame(processed_array, columns=feature_names)

    print("Исходные данные:")
    print(original_df)
    print("\nПосле обработки:")
    print(processed_df)

    return processed_df


# Используй функцию для train данных
processed_train_df = show_transformation(X_train, X_train_processed, preprocessor)

print(f"\nX_train после обработки shape: {X_train_processed.shape}")
print(f"X_test после обработки shape: {X_test_processed.shape}")

# 8. Конвертируем в тензоры
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print(f"\nТензоры PyTorch:")
print(f"X_train_tensor: {X_train_tensor.shape}")
print(f"y_train_tensor: {y_train_tensor.shape}")

# 9. Смотрим имена признаков
feature_names = preprocessor.get_feature_names_out()
print("\nИтоговые признаки:")
for i, name in enumerate(feature_names):
    print(f"{i}: {name}")


print(X_train_processed.shape)
print(train_df['age_group'].unique())