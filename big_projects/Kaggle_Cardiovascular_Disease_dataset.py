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
    test_size=0.176,
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