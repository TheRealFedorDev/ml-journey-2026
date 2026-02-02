import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Download latest version
#path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")

#print("Path to dataset files:", path)


df = pd.read_csv("cardio_train.csv", sep=';')
print(df.head())
print(df.info())
'''
    id;age(days);gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio
    0;18393;2;168;62.0;110;80;1;1;0;0;1;0
    1;20228;1;156;85.0;140;90;3;1;0;0;1;1
    2;18857;1;165;64.0;130;70;3;1;0;0;0;1
    3;17623;2;169;82.0;150;100;1;1;0;0;1;1
    4;17474;1;156;56.0;100;60;1;1;0;0;0;0
'''

print(f"Длина файла: {len(df)}")

if 'id' in df.columns:
    df = df.drop('id', axis=1)

X = df.drop('cardio', axis=1)
y = df['cardio']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,y,
    train_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    random_state=42,
    test_size=0.1765,
    stratify=y_train_val
)

print(f"Train: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nРаспределение классов в train: {y_train.value_counts(normalize=True).to_dict()}")
print(f"Распределение классов в val: {y_val.value_counts(normalize=True).to_dict()}")
print(f"Распределение классов в test: {y_test.value_counts(normalize=True).to_dict()}")


df_sorted = df.sort_values('age')

train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

train_df = df_sorted.iloc[:train_size]
val_df = df_sorted.iloc[train_size:train_size + val_size]
test_df = df_sorted.iloc[train_size + val_size:]

X_train, y_train = train_df.drop('cardio', axis=1), train_df['cardio']
X_val, y_val = val_df.drop('cardio', axis=1), val_df['cardio']
X_test, y_test = test_df.drop('cardio', axis=1), test_df['cardio']



# ==================== АНАЛИЗ ====================

numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
fix, axes = plt.subplots(2, 3, figsize=(15,10))

# Гистограммы для числовых признаков
for i, col in enumerate(numerical_features):
    ax = axes[i//3, i%3]
    ax.hist(df[col], bins=50, alpha=0.7)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Поиск выбросов (особенно в давлении!)
print("\nАнализ давления (должно быть в разумных пределах):")
print(f"ap_hi min: {df['ap_hi'].min()}, max: {df['ap_hi'].max()}")
print(f"ap_lo min: {df['ap_lo'].min()}, max: {df['ap_lo'].max()}")

def clean_blood_pressure(df):
    df_clean = df.copy()
    # Систолическое давление (ap_hi) обычно 90-250
    df_clean = df_clean[(df_clean['ap_hi'] >= 90) & (df_clean['ap_hi'] <= 250)]
    # Диастолическое давление (ap_lo) обычно 60-150
    df_clean = df_clean[(df_clean['ap_lo'] >= 60) & (df_clean['ap_lo'] <= 150)]
    # Систолическое должно быть выше диастолического
    df_clean = df_clean[df_clean['ap_hi'] > df_clean['ap_lo']]
    return df_clean

df_clean = clean_blood_pressure(df)
print(f"\nПосле очистки давления: {len(df_clean)} строк (было {len(df)})")

# Пайплайн для числовых данных
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Пайплайн для категориальных данных

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),    # заполняем модой
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')) # one-hot encoding
])
# Объединяем
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])
# Обучаем препроцессор НА ТРЕНИРОВОЧНЫХ ДАННЫХ
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"Train после обработки: {X_train_processed.shape}")
print(f"Validation после обработки: {X_val_processed.shape}")
print(f"Test после обработки: {X_test_processed.shape}")

class_counts = y_train.value_counts()
print(f"Здоровые (0): {class_counts[0]} ({class_counts[0]/len(y_train)*100:.1f}%)")
print(f"Больные (1): {class_counts[1]} ({class_counts[1]/len(y_train)*100:.1f}%)")


# Если дисбаланс сильный, применяем технику
from imblearn.over_sampling import SMOTE

if abs(class_counts[0] - class_counts[1]) > .1 * len(y_train):
    print("\nПрименяем SMOTE для балансировки классов...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    print(f"После балансировки: {X_train_balanced.shape}")
    # После применения SMOTE
    print(f"\nПосле балансировки: {X_train_balanced.shape}")

    # Проверяем и конвертируем y
    print(f"Тип y_train_balanced: {type(y_train_balanced)}")
    print(f"Форма y_train_balanced: {y_train_balanced.shape if hasattr(y_train_balanced, 'shape') else 'нет формы'}")

    # Конвертируем y в правильный формат
    if isinstance(y_train_balanced, pd.Series):
        y_train_balanced_np = y_train_balanced.values
    elif isinstance(y_train_balanced, np.ndarray):
        y_train_balanced_np = y_train_balanced
    else:
        # Пробуем преобразовать
        y_train_balanced_np = np.array(y_train_balanced)

    print(f"y_train_balanced_np shape: {y_train_balanced_np.shape}")

    # Теперь конвертируем ВСЕ данные
    X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_balanced_np, dtype=torch.float32).view(-1, 1)

    print(f"\nИтоговые размеры:")
    print(f"X_train_tensor: {X_train_tensor.shape}")
    print(f"y_train_tensor: {y_train_tensor.shape}")

    # Проверка
    if X_train_tensor.shape[0] != y_train_tensor.shape[0]:
        print(f"⚠️  ВНИМАНИЕ: размеры не совпадают!")
        print(f"   X имеет {X_train_tensor.shape[0]} примеров")
        print(f"   y имеет {y_train_tensor.shape[0]} примеров")

        # Исправляем: берём минимум из двух
        min_size = min(X_train_tensor.shape[0], y_train_tensor.shape[0])
        print(f"   Берём первые {min_size} примеров из каждого")

        X_train_tensor = X_train_tensor[:min_size]
        y_train_tensor = y_train_tensor[:min_size]

        print(f"   Исправленные размеры:")
        print(f"   X_train_tensor: {X_train_tensor.shape}")
        print(f"   y_train_tensor: {y_train_tensor.shape}")

    # Теперь создаём dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    print(f"\nTrain dataset создан: {len(train_dataset)} примеров")

    # Аналогично для validation и test (они не проходили SMOTE)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    print(f"\nValidation: X={X_val_tensor.shape}, y={y_val_tensor.shape}")
    print(f"Test: X={X_test_tensor.shape}, y={y_test_tensor.shape}")

    # Создаём DataLoader
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    print(f"\nDataLoader созданы:")
    print(f"  Train батчей: {len(train_loader)}")
    print(f"  Validation батчей: {len(val_loader)}")
    print(f"  Test батчей: {len(test_loader)}")
else:
    X_train_balanced, y_train_balanced = X_train_processed, y_train





X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_balanced.values, dtype=torch.float32).view(-1,1)
X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)


# Создаём новые признаки
def create_features(df):
    df = df.copy()

    # 1. Преобразуем возраст в годы
    df['age_years'] = df['age'] // 365

    # 2. BMI (индекс массы тела)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # 3. Разница давления (пульсовое давление)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # 4. Среднее артериальное давление
    df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

    # 5. Квадрат возраста (нелинейность)
    df['age_squared'] = df['age_years'] ** 2

    # Удаляем исходные колонки которые заменили
    df = df.drop(['age', 'height', 'weight'], axis=1, errors='ignore')

    return df


def create_better_features(df):
    """Создаём новые признаки из исходных"""
    df = df.copy()

    # 1. Возраст в годах (вместо дней)
    df['age_years'] = df['age'] // 365

    # 2. BMI (индекс массы тела)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # 3. Категории BMI
    df['bmi_category'] = pd.cut(df['bmi'],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3])

    # 4. Пульсовое давление (важный медицинский показатель)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # 5. Среднее артериальное давление
    df['mean_arterial_pressure'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

    # 6. Взаимодействия (features crossing)
    df['age_pressure'] = df['age_years'] * df['ap_hi'] / 100
    df['bmi_cholesterol'] = df['bmi'] * df['cholesterol']

    # 7. Квадраты признаков (для нелинейности)
    df['age_squared'] = df['age_years'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2

    # Удаляем исходные колонки
    cols_to_drop = ['age', 'id'] if 'id' in df.columns else ['age']
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    return df

# Применяем к данным
X_train_features = create_better_features(X_train)
X_val_features = create_better_features(X_val)
X_test_features = create_better_features(X_test)

print(f"Новое количество признаков: {X_train_features.shape[1]}")

print(f"X_test_tensor shape is {X_test_tensor.shape}")
print(f"y_test_tensor shape is {y_test_tensor.shape}")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# Используем батчи для экономии памяти
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Количество батчей в train: {len(train_loader)}")
print(f"Количество батчей в validation: {len(val_loader)}")

debug_size = .1
X_train_small, _, y_train_small, _ = train_test_split(
    X_train, y_train,
    train_size=debug_size,
    random_state=42,
    stratify=y_train
)

print(f"\nДля отладки используем {len(X_train_small)} примеров")

import joblib

joblib.dump(preprocessor, 'preprocessor.pkl')

np.savez_compressed(
    'processed_data.npz',
    X_train=X_train_processed,
    X_val=X_val_processed,
    X_test=X_test_processed,
    y_train=y_train.values,
    y_val=y_val.values,
    y_test=y_test.values,
)

print("Данные сохранены!")


class SimpleCardioNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            # Слой 1: input_size -> 32
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Слой 2: 32 -> 16
            nn.Linear(32, 16),  # Было 32 -> 16, а не 32 -> ?
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Слой 3: 16 -> 8
            nn.Linear(16, 8),
            nn.ReLU(),

            # Выходной слой: 8 -> 1
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
class AdvancedCardioNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(

            # Слой 1: больше нейронов
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),  # Ускоряет обучение
            nn.ReLU(),
            nn.Dropout(0.4), # Против переобучения

            # Слой 2
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Слой 3
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Слой 4
            nn.Linear(32, 16),
            nn.ReLU(),

            # Выходной слой
            nn.Linear(16,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        return self.network(x)

input_size = 13
simple_model = SimpleCardioNet(input_size)


class EnsembleModel(nn.Module):
    def __init__(self, input_size, n_models=3):
        super().__init__()
        self.models = nn.ModuleList([
            AdvancedCardioNet(input_size) for _ in range(n_models)
        ])

    def forward(self, x):
        # Предсказания всех моделей
        predictions = torch.stack([model(x) for model in self.models])
        # Усредняем
        return torch.mean(predictions, dim=0)

print("=== ПРОСТАЯ МОДЕЛЬ ===")
print(f"Входные признаки: {input_size}")
print(f"Слой 1: Linear({input_size}, 16)")
print(f"Слой 2: Linear(32, 16)")
print(f"Слой 3: Linear(16, 8)")
print(f"Выходной слой: Linear(8, 1)")
print(f"Всего параметров: {sum(p.numel() for p in simple_model.parameters()):,}")



print("Размеры параметров:")
for name, param in simple_model.named_parameters():
    print(f"  {name}: {param.shape}")


def train_better(model, train_loader, val_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # ===== ОБУЧЕНИЕ =====
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Прямой проход
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Обратное распространение
            loss.backward()
            optimizer.step()

            # Статистики
            train_loss += loss.item()

            # Accuracy
            preds = (predictions > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        # ===== ВАЛИДАЦИЯ =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

                preds = (predictions > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        # ===== ВЫЧИСЛЕНИЕ СРЕДНИХ =====
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # ===== ВЫВОД =====
        print(f"Эпоха {epoch + 1}/{epochs}:")
        print(f"  Train: loss={avg_train_loss:.4f}, acc={train_acc:.2%}")
        print(f"  Val:   loss={avg_val_loss:.4f}, acc={val_acc:.2%}")

        # Ранняя остановка если accuracy низкая
        if epoch >= 3 and val_acc < 0.65:
            print("⚠️  Accuracy низкая, пробуем другую архитектуру")
            break

    return model, history  # Возвращаем И модель, И историю


def train_simple(model, train_loader, val_loader, epochs=5):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    results = []

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += ((preds > 0.5).float() == y_batch).sum().item()
            train_total += len(y_batch)

        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)

                val_loss += loss.item()
                val_correct += ((preds > 0.5).float() == y_batch).sum().item()
                val_total += len(y_batch)

        # Считаем средние
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Сохраняем
        results.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })

        print(f"Эпоха {epoch + 1}: Train acc={train_acc:.2%}, Val acc={val_acc:.2%}")

    return model, results


# ВАРИАНТ 2: С ансамблем (нужно адаптировать функцию обучения)
def train_ensemble(model, train_loader, val_loader, epochs=10):
    """Обучение для ансамбля моделей"""
    criterion = nn.BCELoss()

    # Разные оптимизаторы для каждой модели
    optimizers = [
        torch.optim.Adam(m.parameters(), lr=0.0005, weight_decay=0.001)
        for m in model.models
    ]

    history = []

    for epoch in range(epochs):
        # Обучение каждой модели
        for i, (submodel, optimizer) in enumerate(zip(model.models, optimizers)):
            submodel.train()

            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                preds = submodel(X_batch)
                loss = criterion(preds, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(submodel.parameters(), max_norm=1.0)
                optimizer.step()

                train_correct += ((preds > 0.5).float() == y_batch).sum().item()
                train_total += len(y_batch)

        # Валидация ансамбля
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)  # Ансамбль усредняет предсказания
                val_correct += ((preds > 0.5).float() == y_batch).sum().item()
                val_total += len(y_batch)

        val_acc = val_correct / val_total

        # Примерная train accuracy (средняя по моделям)
        train_acc = 0
        for submodel in model.models:
            submodel.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in train_loader:
                    preds = submodel(X_batch)
                    correct += ((preds > 0.5).float() == y_batch).sum().item()
                    total += len(y_batch)
            train_acc += correct / total
        train_acc /= len(model.models)

        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        print(f"Эпоха {epoch + 1}: Train={train_acc:.2%}, Val={val_acc:.2%}")

    return model, history


def validate_update(model, val_loader):
    criterion = nn.BCELoss()

    model.eval() # Режим оценки
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad(): # Отключаем вычисление градиентов
        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            total_loss += loss.item()

            binary_preds = (predictions > .5).float()
            total_correct += (binary_preds == y_batch).sum().item()
            total_samples += len(y_batch)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model_comprehensive(model, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
        binary_preds = (predictions > 0.5).float()

        # Confusion matrix
        cm = confusion_matrix(y_tensor.numpy(), binary_preds.numpy())
        print("Confusion Matrix:")
        print(f"           Predicted 0  Predicted 1")
        print(f"Actual 0    {cm[0, 0]:8}    {cm[0, 1]:8}")
        print(f"Actual 1    {cm[1, 0]:8}    {cm[1, 1]:8}")

        # Подробный отчет
        print("\nClassification Report:")
        print(classification_report(y_tensor.numpy(), binary_preds.numpy()))

        # ROC-AUC (важно для несбалансированных данных)
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = roc_auc_score(y_tensor.numpy(), predictions.numpy())
        print(f"ROC-AUC Score: {auc:.4f}")

        # Precision, Recall, F1 для каждого класса
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nClass 1 (больные):")
        print(f"  Precision: {precision:.4f} - из предсказанных больных, сколько действительно больны")
        print(f"  Recall:    {recall:.4f} - из всех больных, сколько мы нашли")
        print(f"  F1-score:  {f1:.4f} - баланс между precision и recall")


# Применим к твоей модели
print("=== КОМПЛЕКСНАЯ ОЦЕНКА ===")
evaluate_model_comprehensive(simple_model, X_val_tensor, y_val_tensor)

'''
simple_model = AdvancedCardioNet(input_size=13)

batch_size = 128

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_tensor, y_val_tensor),
    batch_size=batch_size,
    shuffle=False
)

print("=== НАЧИНАЕМ ОБУЧЕНИЕ ===")
print(f"Размер батча: {batch_size}")
print(f"Батчей в train: {len(train_loader)}")
print(f"Батчей в validation: {len(val_loader)}")

for epoch in range(5):
    print(f"\nЭпоха {epoch + 1}/5:")
    # Обучение
    trained_model, history = train_better(simple_model, train_loader, val_loader)
    print(f"Лучшая train accuracy: {max(history['train_acc']):.2%}")
    print(f"Лучшая val accuracy: {max(history['val_acc']):.2%}")

    # Проверка
    val_loss, val_accur = validate_update(simple_model, val_loader)
    print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_accur:.2%}")

    # Анализ
    if val_accur > 0.7:
        print("✅ Хороший результат!")
    elif val_accur < 0.6:
        print("⚠️  Нужно улучшать модель")
    else:
        print("➡️  Средний результат")

print("\n=== АНАЛИЗ МОДЕЛИ ===")
# 1. Посмотрим на предсказания для нескольких примеров
simple_model.eval()

with torch.no_grad():

    sample_X = X_val_tensor[:5]
    sample_y = y_val_tensor[:5]

    predictions = simple_model(sample_X)

    print("Примеры предсказаний:")

    for i in range(5):
        prob = predictions[i].item()
        real = sample_y[i].item()
        pred_class = 1 if prob > .5 else 0
        correct = "✓" if pred_class == real else "✗"
        print(f"  Пациент {i + 1}: Вероятность={prob:.2%} → {pred_class} (реально: {real}) {correct}")


# 2. Проверим на всём тестовом наборе
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=batch_size,
    shuffle=False
)

test_loss, test_accur = validate_update(simple_model, test_loader)
print(f"\nФинальный тест:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accur:.2%}")

# 3. Сохраним модель
torch.save(simple_model.state_dict(), 'simple_cardio_model.pth')
print("Модель сохранена в 'simple_cardio_model.pth'")
'''
ensemble = EnsembleModel(input_size=13, n_models=3)
trained_ensemble, ensemble_history = train_ensemble(ensemble, train_loader, val_loader, epochs=10)
print("\n=== ИТОГИ ===")
best_result = max(ensemble_history, key=lambda x: x['val_acc'])
print(f"Лучшая эпоха: {best_result['epoch']}")
print(f"Лучшая валидационная accuracy: {best_result['val_acc']:.2%}")
print(f"Соответствующая train accuracy: {best_result['train_acc']:.2%}")


def detailed_error_analysis(model, X_tensor, y_tensor, feature_names=None):
    """Детальный анализ ошибок модели"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
        probs = predictions.numpy()
        preds = (predictions > 0.5).float().numpy()
        y_true = y_tensor.numpy()

    # Индексы разных типов ошибок
    tp_indices = np.where((preds == 1) & (y_true == 1))[0]  # True Positive
    tn_indices = np.where((preds == 0) & (y_true == 0))[0]  # True Negative
    fp_indices = np.where((preds == 1) & (y_true == 0))[0]  # False Positive
    fn_indices = np.where((preds == 0) & (y_true == 1))[0]  # False Negative

    print(f"\n=== ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК ===")
    print(f"Всего примеров: {len(y_true)}")
    print(
        f"Правильно предсказано: {len(tp_indices) + len(tn_indices)} ({((len(tp_indices) + len(tn_indices)) / len(y_true) * 100):.1f}%)")
    print(
        f"Ошибок: {len(fp_indices) + len(fn_indices)} ({((len(fp_indices) + len(fn_indices)) / len(y_true) * 100):.1f}%)")
    print(f"  - False Positive (здорового назвали больным): {len(fp_indices)}")
    print(f"  - False Negative (больного назвали здоровым): {len(fn_indices)}")

    # Анализ уверенности модели
    print(f"\n=== УВЕРЕННОСТЬ МОДЕЛИ ===")
    print(f"Средняя вероятность для правильных предсказаний: {probs[tp_indices].mean():.2%}")
    print(f"Средняя вероятность для ошибок FP: {probs[fp_indices].mean():.2%}")
    print(f"Средняя вероятность для ошибок FN: {probs[fn_indices].mean():.2%}")

    # Если есть feature_names, можно проанализировать признаки
    if feature_names is not None and len(fp_indices) > 0 and len(fn_indices) > 0:
        print(f"\n=== АНАЛИЗ ПРИЗНАКОВ ДЛЯ ОШИБОК ===")
        X_np = X_tensor.numpy()

        # Средние значения признаков для разных групп
        print("Средние значения признаков:")
        print(f"{'Признак':<20} {'Все':<10} {'FP':<10} {'FN':<10}")
        for i, name in enumerate(feature_names):
            if i < X_np.shape[1]:  # Проверяем границы
                print(
                    f"{name:<20} {X_np[:, i].mean():<10.3f} {X_np[fp_indices, i].mean():<10.3f} {X_np[fn_indices, i].mean():<10.3f}")


# Запускаем анализ
print("Анализ на validation данных...")
feature_names = preprocessor.get_feature_names_out()  # если препроцессор сохранён
detailed_error_analysis(trained_ensemble, X_val_tensor[:1000], y_val_tensor[:1000], feature_names)

# Правильный код для вывода результатов ансамбля
print("\n=== РЕЗУЛЬТАТЫ АНСАМБЛЯ ===")
if ensemble_history:
    best_result = max(ensemble_history, key=lambda x: x['val_acc'])
    print(f"Лучшая эпоха: {best_result['epoch']}")
    print(f"Лучшая валидационная accuracy: {best_result['val_acc']:.2%}")
    print(f"Train accuracy в эту эпоху: {best_result['train_acc']:.2%}")

    # График обучения
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in ensemble_history]
    train_accs = [h['train_acc'] for h in ensemble_history]
    val_accs = [h['val_acc'] for h in ensemble_history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label='Train', marker='o')
    plt.plot(epochs, val_accs, label='Validation', marker='s')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Обучение ансамбля моделей')
    plt.legend()
    plt.grid(True)
    plt.show()