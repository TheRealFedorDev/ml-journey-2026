# cardio_sklearn_only.py - ТОЛЬКО sklearn, НИКАКОГО PyTorch!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Базовая информация
print("1. Загрузка данных...")
try:
    df = pd.read_csv('cardio_train.csv', sep=';')
    print(f" Успешно! Размер: {df.shape}")
except FileNotFoundError:
    print("  Файл 'cardio_train.csv' не найден!")
    print("   Помести файл в ту же папку, где этот скрипт")
    exit()

# 2. БАЗОВЫЙ АНАЛИЗ
print("\n2. Анализ данных...")
print(f"   Колонки: {list(df.columns)}")
print(f"   Пропуски: {df.isnull().sum().sum()}")
print(f"   Распределение cardio: {df['cardio'].value_counts().to_dict()}")

# 3. ОЧИСТКА
print("\n3. Очистка данных...")



df_copy_1 = df.copy()


print(f"\n=== АГРЕССИВНАЯ ОЧИСТКА ДАННЫХ ===")
print(f"До очистки: {len(df)} записей")

original_count = len(df)

# 1. ДАВЛЕНИЕ - только физиологически возможные значения
# Верхнее давление: 70-250, Нижнее: 40-180, Верхнее > Нижнее
df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 180)]
df = df[df['ap_hi'] > df['ap_lo']]  # Верхнее ВСЕГДА больше нижнего!

# 2. РОСТ И ВЕС - взрослые люди
df = df[(df['height'] >= 140) & (df['height'] <= 220)]  # 1.4м - 2.2м
df = df[(df['weight'] >= 40) & (df['weight'] <= 180)]    # 40кг - 180кг

# 3. ПЕРЕСЧИТАТЬ BMI с очищенными данными
df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
df = df[(df['bmi'] >= 16) & (df['bmi'] <= 45)]  # Реалистичный диапазон

# 4. УДАЛИТЬ ВЫБРОСЫ В ВОЗРАСТЕ (уже нормально, но проверим)
df['age_years'] = df['age'] // 365
df = df[(df['age_years'] >= 30) & (df['age_years'] <= 65)]

print(f"\nУдалено записей: {original_count - len(df)} ({100*(original_count - len(df))/original_count:.1f}%)")
print(f"После очистки: {len(df)} записей")

# Проверка после очистки
print("\n=== ПРОВЕРКА ПОСЛЕ ОЧИСТКИ ===")
print(f"ap_hi: [{df['ap_hi'].min()}, {df['ap_hi'].max()}]")
print(f"ap_lo: [{df['ap_lo'].min()}, {df['ap_lo'].max()}]")
print(f"Рост: [{df['height'].min()}, {df['height'].max()}] см")
print(f"Вес: [{df['weight'].min()}, {df['weight'].max()}] кг")
print(f"BMI: [{df['bmi'].min():.1f}, {df['bmi'].max():.1f}]")
print(f"Возраст: [{df['age_years'].min()}, {df['age_years'].max()}] лет")


print("\n=== СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ ===")

# 1. МЕДИЦИНСКИЕ ПРИЗНАКИ
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']  # Пульсовое давление
df['mean_arterial'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3  # Среднее артериальное

# 2. КАТЕГОРИИ ДАВЛЕНИЯ (гипертония по стадиям)
def categorize_pressure(row):
    sys, dia = row['ap_hi'], row['ap_lo']
    if sys < 120 and dia < 80:
        return 0  # Нормальное
    elif sys < 130 and dia < 80:
        return 1  # Повышенное
    elif sys < 140 or dia < 90:
        return 2  # Гипертония 1 степени
    elif sys < 180 or dia < 120:
        return 3  # Гипертония 2 степени
    else:
        return 4  # Гипертонический криз

df['pressure_stage'] = df.apply(categorize_pressure, axis=1)

# 3. ВЗАИМОДЕЙСТВИЕ ПРИЗНАКОВ
df['age_pressure_risk'] = df['age_years'] * (df['ap_hi'] / 100)
df['chol_bmi'] = df['cholesterol'] * df['bmi']

# 4. ЛОГАРИФМИРОВАНИЕ skewed признаков
for col in ['ap_hi', 'ap_lo', 'weight']:
    if df[col].skew() > 0.5:
        df[f'log_{col}'] = np.log1p(df[col])
        print(f"Создан log_{col} (skew был {df[col].skew():.2f})")

print(f"Всего признаков после feature engineering: {len(df.columns)}")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Базовая информация
print("=== ИНФОРМАЦИЯ О ДАННЫХ ===")
print(f"Размер данных: {df.shape}")
print(f"Пропуски:\n{df.isnull().sum()}")
print(f"Типы данных:\n{df.dtypes}")
print(f"\nБаланс классов cardio:\n{df['cardio'].value_counts(normalize=True)}")

# 2. Статистика по числовым признакам
print("\n=== СТАТИСТИКА ===")
print(df.describe())

print("\n=== КРИТИЧЕСКИЕ ЗНАЧЕНИЯ ===")
print(f"ap_hi - мин: {df['ap_hi'].min()}, макс: {df['ap_hi'].max()}")
print(f"ap_lo - мин: {df['ap_lo'].min()}, макс: {df['ap_lo'].max()}")

# Проверка на абсолютно невозможные значения
print(f"\nНевозможные значения ap_hi > 300: {(df['ap_hi'] > 300).sum()}")
print(f"Невозможные значения ap_lo > 300: {(df['ap_lo'] > 300).sum()}")
print(f"ap_hi < ap_lo: {(df['ap_hi'] < df['ap_lo']).sum()}")

# Возраст в годах
df['age_years'] = df['age'] // 365
print(f"\nВозраст в годах: от {df['age_years'].min()} до {df['age_years'].max()} лет")

# Рост и вес (возможные ошибки)
print(f"\nРост: от {df['height'].min()} до {df['height'].max()} см")
print(f"Вес: от {df['weight'].min()} до {df['weight'].max()} кг")

# BMI для проверки
df['bmi_temp'] = df['weight'] / ((df['height']/100) ** 2)
print(f"\nBMI: от {df['bmi_temp'].min():.1f} до {df['bmi_temp'].max():.1f}")
print(f"Нереалистичный BMI (>50): {(df['bmi_temp'] > 50).sum()}")
print(f"Нереалистичный BMI (<15): {(df['bmi_temp'] < 15).sum()}")

print("\n=== МОЩНЫЙ FEATURE ENGINEERING ===")

# 1. Полиномиальные признаки (взаимодействия)
df['age_bmi_interaction'] = df['age_years'] * df['bmi']
df['pressure_product'] = df['ap_hi'] * df['ap_lo']
df['chol_age_squared'] = df['cholesterol'] * (df['age_years'] ** 2)

# 2. Медицинские индексы
df['hypertension_index'] = (df['ap_hi'] - 120) * (df['ap_lo'] - 80)
df['metabolic_age'] = df['bmi'] * df['age_years'] / 10

# 3. Категориальные взаимодействия
df['risk_cluster'] = df['cholesterol'] * df['pressure_stage'] * df['age_years']

# 4. Бинарные комбинации
df['high_risk'] = ((df['cholesterol'] > 1) & (df['ap_hi'] > 140)).astype(int)
df['sedentary_smoker'] = ((df['active'] == 0) & (df['smoke'] == 1)).astype(int)

# 5. Квантили и бининг
for col in ['ap_hi', 'ap_lo', 'bmi', 'age_years']:
    df[f'{col}_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
    # One-hot encoding для бинов
    dummies = pd.get_dummies(df[f'{col}_bin'], prefix=f'{col}_bin')
    df = pd.concat([df, dummies], axis=1)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Теперь признаков: {len(df.columns)}")
print("Новые признаки:", [col for col in df.columns if col not in numeric_cols][:10])

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

print("\n=== БЫСТРОЕ ТЕСТИРОВАНИЕ МОДЕЛЕЙ ===")

# Подготовка данных
X = df.drop(['cardio', 'id'], axis=1, errors='ignore')  # Удаляем ID и целевую
y = df['cardio']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование числовых признаков
scaler = StandardScaler()
num_cols = X.select_dtypes(include=[np.number]).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\n=== ГЛУБОКАЯ ДИАГНОСТИКА ===")

# 1. Корреляция всех признаков с целевой переменной
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['cardio'].sort_values(ascending=False)

print("\nТоп-10 признаков по корреляции с cardio:")
for feat, corr in correlations.head(10).items():
    print(f"  {feat}: {corr:.4f}")

print("\nХудшие 5 признаков по корреляции с cardio:")
for feat, corr in correlations.tail(5).items():
    print(f"  {feat}: {corr:.4f}")

# 2. Проверка на мультиколлинеарность
print("\n=== ПРОВЕРКА НА МУЛЬТИКОЛЛИНЕАРНОСТЬ ===")
high_corr_pairs = []
corr_matrix = df[numeric_cols].corr().abs()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.8:  # Очень высокая корреляция
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("Найдены сильно коррелирующие признаки (>0.8):")
    for col1, col2, corr in high_corr_pairs[:5]:  # Покажем первые 5
        print(f"  {col1} и {col2}: {corr:.4f}")
else:
    print("Сильно коррелирующих признаков не найдено")

# 3. Визуализация разделимости классов
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
top_features = correlations.index[1:7]  # Первые 6 кроме cardio

for i, feat in enumerate(top_features):
    ax = axes[i // 3, i % 3]

    # Разделяем по классам
    df_0 = df[df['cardio'] == 0][feat]
    df_1 = df[df['cardio'] == 1][feat]

    ax.hist(df_0, bins=30, alpha=0.5, label='No Disease', density=True)
    ax.hist(df_1, bins=30, alpha=0.5, label='Disease', density=True)
    ax.set_title(f'{feat}\n(corr={correlations[feat]:.3f})')
    ax.legend()

plt.tight_layout()
plt.show()

# Тестируем модели
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")

    # Быстрая кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Финальное обучение и оценка на тесте
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    results[name] = test_acc
    print(f"Test Accuracy: {test_acc:.4f}")

    # Если accuracy > 0.85, покажем важность признаков
    if test_acc > 0.85:
        print(f"\nТоп-10 важных признаков для {name}:")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_importance = sorted(zip(X.columns, importances),
                                     key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in feat_importance:
                print(f"  {feat}: {imp:.4f}")

print(f"\n=== ИТОГ ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

    print("\n=== МОЩНЫЙ FEATURE ENGINEERING ===")

    # 1. Полиномиальные признаки (взаимодействия)
    df['age_bmi_interaction'] = df['age_years'] * df['bmi']
    df['pressure_product'] = df['ap_hi'] * df['ap_lo']
    df['chol_age_squared'] = df['cholesterol'] * (df['age_years'] ** 2)

    # 2. Медицинские индексы
    df['hypertension_index'] = (df['ap_hi'] - 120) * (df['ap_lo'] - 80)
    df['metabolic_age'] = df['bmi'] * df['age_years'] / 10

    # 3. Категориальные взаимодействия
    df['risk_cluster'] = df['cholesterol'] * df['pressure_stage'] * df['age_years']

    # 4. Бинарные комбинации
    df['high_risk'] = ((df['cholesterol'] > 1) & (df['ap_hi'] > 140)).astype(int)
    df['sedentary_smoker'] = ((df['active'] == 0) & (df['smoke'] == 1)).astype(int)

    # 5. Квантили и бининг
    for col in ['ap_hi', 'ap_lo', 'bmi', 'age_years']:
        df[f'{col}_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
        # One-hot encoding для бинов
        dummies = pd.get_dummies(df[f'{col}_bin'], prefix=f'{col}_bin')
        df = pd.concat([df, dummies], axis=1)

    print(f"Теперь признаков: {len(df.columns)}")
    print("Новые признаки:", [col for col in df.columns if col not in numeric_cols][:10])

print("\n=== НЕЙРОСЕТЬ НА PYTORCH ===")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Преобразуем данные в тензоры
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

# Создаём Dataset и DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Архитектура нейросети
class CardioNet(nn.Module):
    def __init__(self, input_size):
        super(CardioNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


# Инициализация
model = CardioNet(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Валидация каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_preds = (test_outputs > 0.5).float()
            test_acc = (test_preds == y_test_tensor).float().mean()
            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Test Acc: {test_acc:.4f}')

# Финальная оценка
model.eval()
with torch.no_grad():
    final_preds = (model(X_test_tensor) > 0.5).float()
    final_acc = (final_preds == y_test_tensor).float().mean()
    print(f"\nФинальная точность нейросети: {final_acc:.4f}")