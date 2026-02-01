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
else:
    X_train_balanced, y_train_balanced = X_train_processed, y_train

from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_balanced.values, dtype=torch.float32).view(-1,1)
X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)

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
        # Вход: 13 признаков → Скрытый слой: 8 нейронов → Выход: 1 нейрон
        self.layer1 = nn.Linear(input_size, 8)  # 13 → 8
        self.layer2 = nn.Linear(8,1)  # 8 → 1
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.output_activation(x)
        return x

input_size = 13
simple_model = SimpleCardioNet(input_size)

print("=== ПРОСТАЯ МОДЕЛЬ ===")
print(f"Входные признаки: {input_size}")
print(f"Слой 1: Linear({input_size}, 8)")
print(f"Слой 2: Linear(8, 1)")
print(f"Всего параметров: {sum(p.numel() for p in simple_model.parameters()):,}")
print()


print("Размеры параметров:")
for name, param in simple_model.named_parameters():
    print(f"  {name}: {param.shape}")

def train_one_epoch_simple(model, train_loader):
    criterion = nn.BCELoss()