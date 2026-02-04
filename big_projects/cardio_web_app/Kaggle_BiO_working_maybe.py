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

print("=== КАРДИО ДАТАСЕТ - ТОЛЬКО SKLEARN ===\n")

# 1. ЗАГРУЗКА ДАННЫХ
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


def clean_data(df):
    df = df.copy()

    # Удаляем ID
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Очистка нереалистичных значений давления
    initial_size = len(df)
    df = df[(df['ap_hi'] >= 90) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 60) & (df['ap_lo'] <= 150)]
    df = df[df['ap_hi'] > df['ap_lo']]

    removed = initial_size - len(df)
    print(f"      Удалено {removed} строк с нереалистичным давлением")

    # Возраст в годах
    df['age_years'] = df['age'] // 365

    # BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Пример: биологически возможные диапазоны (настрой под данные!)
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 200)]
    df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]
    df = df[(df['height'] >= 140) & (df['height'] <= 210)]
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 45)]

    return df


df_clean = clean_data(df)
print(f"   После очистки: {df_clean.shape}")

# 4. РАЗДЕЛЕНИЕ
print("\n4. Разделение на train/val/test...")
X = df_clean.drop('cardio', axis=1)
y = df_clean['cardio']

# 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

print(f"   Train: {X_train.shape} ({len(X_train) / len(X) * 100:.1f}%)")
print(f"   Validation: {X_val.shape} ({len(X_val) / len(X) * 100:.1f}%)")
print(f"   Test: {X_test.shape} ({len(X_test) / len(X) * 100:.1f}%)")

# 5. ПРЕОБРАБОТКА ПРИЗНАКОВ
print("\n5. Подготовка признаков...")

# Определяем типы колонок
numeric_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'age_years', 'bmi']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Создаём препроцессор
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# Обучаем на train
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"   После обработки: {X_train_processed.shape[1]} признаков")

# 6. БАЛАНСИРОВКА КЛАССОВ
print("\n6. Балансировка классов (SMOTE)...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
print(f"   После SMOTE: {X_train_balanced.shape}")

# 7. ОБУЧЕНИЕ МОДЕЛЕЙ
print("\n" + "=" * 50)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 50)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n▶ {name}:")

    # Обучение
    model.fit(X_train_balanced, y_train_balanced)

    # Предсказания
    y_train_pred = model.predict(X_train_balanced)
    y_val_pred = model.predict(X_val_processed)

    # Метрики
    train_acc = accuracy_score(y_train_balanced, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    print(f"   Train Accuracy: {train_acc:.2%}")
    print(f"   Val Accuracy:   {val_acc:.2%}")

    # Переобучение?
    overfit = train_acc - val_acc
    if overfit > 0.1:
        print(f" Возможно переобучение (разница: {overfit:.2%})")

# 8. ВЫБОР ЛУЧШЕЙ МОДЕЛИ
print("\n" + "=" * 50)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 50)

# Таблица результатов
print("\nМодель                 Train Acc    Val Acc")
print("-" * 40)
for name, res in results.items():
    print(f"{name:<22} {res['train_acc']:>10.2%} {res['val_acc']:>10.2%}")

# Выбираем лучшую по validation accuracy
best_name = max(results, key=lambda x: results[x]['val_acc'])
best_model = results[best_name]['model']
best_val_acc = results[best_name]['val_acc']

print(f"\nЛучшая модель: {best_name} (Val Acc: {best_val_acc:.2%})")

# 9. ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ
print("\n" + "=" * 50)
print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ НА TEST SET")
print("=" * 50)

# Предсказания на тесте
y_test_pred = best_model.predict(X_test_processed)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTest Accuracy: {test_acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(f"           Predicted 0  Predicted 1")
print(f"Actual 0    {cm[0, 0]:8}    {cm[0, 1]:8}")
print(f"Actual 1    {cm[1, 0]:8}    {cm[1, 1]:8}")

# Детальный отчёт
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# 10. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ (если это Random Forest или GB)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 50)
    print("ВАЖНОСТЬ ПРИЗНАКОВ")
    print("=" * 50)

    # Получаем имена признаков после преобразования
    try:
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # Для one-hot encoded признаков
                encoder = transformer.named_steps['onehotencoder'] if isinstance(transformer, Pipeline) else transformer
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_features = encoder.get_feature_names_out(features)
                    feature_names.extend(cat_features)

        # Создаём DataFrame с важностью
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nТоп-10 самых важных признаков:")
        print(feature_importance.head(10))

        # Визуализация
        plt.figure(figsize=(10, 6))
        top10 = feature_importance.head(10)
        plt.barh(range(len(top10)), top10['importance'])
        plt.yticks(range(len(top10)), top10['feature'])
        plt.xlabel('Важность')
        plt.title('Топ-10 важных признаков')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Не удалось получить важность признаков: {e}")

# 11. СОХРАНЕНИЕ МОДЕЛИ
print("\n" + "=" * 50)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 50)

import joblib

# Сохраняем модель и препроцессор
joblib.dump(best_model, 'best_cardio_model.pkl')
joblib.dump(preprocessor, 'cardio_preprocessor.pkl')

print("Модель сохранена: 'best_cardio_model.pkl'")
print("Препроцессор сохранён: 'cardio_preprocessor.pkl'")

# Сохраняем результаты
results_df = pd.DataFrame({
    'model': list(results.keys()),
    'train_acc': [r['train_acc'] for r in results.values()],
    'val_acc': [r['val_acc'] for r in results.values()]
})
results_df.to_csv('model_results.csv', index=False)
print("Результаты сохранены: 'model_results.csv'")

print("\n" + "=" * 50)
print(" ВСЁ ЗАВЕРШЕНО!")
print("=" * 50)


