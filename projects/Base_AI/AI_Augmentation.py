import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ==================== 1. ПОДГОТОВКА ДАННЫХ ====================
# (используем твой код с исправлениями)

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


def create_better_synthetic_data(n_samples=100):
    synthetic = []

    for _ in range(n_samples):
        patient = {}

        # Возраст: нормальное распределение
        patient['age'] = np.random.normal(40, 10)
        patient['age'] = max(20, min(80, patient['age']))

        # Давление: зависит от возраста
        base_pressure = 120
        age_effect = (patient['age'] - 40) * 0.5  # с возрастом давление растёт
        patient['blood_pressure'] = np.random.normal(base_pressure + age_effect, 10)

        # BMI: нормальное распределение
        patient['bmi'] = np.random.normal(25, 4)

        # Холестерин: зависит от возраста и BMI
        cholesterol_prob = 0.3 + 0.005 * (patient['age'] - 30) + 0.01 * (patient['bmi'] - 25)
        patient['cholesterol'] = 'high' if np.random.random() < cholesterol_prob else np.random.choice(
            ['medium', 'low'])

        # Курильщик: случайно, но чаще с возрастом
        smoker_prob = 0.2 + 0.005 * (patient['age'] - 30)
        patient['smoker'] = 'yes' if np.random.random() < smoker_prob else 'no'

        # Диагноз: ЗАВИСИМ ОТ ПРИЗНАКОВ (это важно!)
        risk_score = 0
        risk_score += 0.1 if patient['age'] > 50 else 0
        risk_score += 0.2 if patient['blood_pressure'] > 140 else 0
        risk_score += 0.3 if patient['cholesterol'] == 'high' else 0.1 if patient['cholesterol'] == 'medium' else 0
        risk_score += 0.2 if patient['smoker'] == 'yes' else 0
        risk_score += 0.1 if patient['bmi'] > 30 else 0

        # Вероятность болезни
        disease_prob = 1 / (1 + np.exp(-risk_score))
        patient['diagnosis'] = 1 if np.random.random() < disease_prob else 0

        synthetic.append(patient)

    return pd.DataFrame(synthetic)

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


print("Создаём реалистичные синтетические данные...")
realistic_synthetic = create_better_synthetic_data(200)
print(f"Создано {len(realistic_synthetic)} реалистичных пациентов")
print("\nРаспределение диагнозов:")
print(realistic_synthetic['diagnosis'].value_counts())
print("\nКорреляция возраста и давления:")
print(realistic_synthetic[['age', 'blood_pressure']].corr())


# 1. Создаём ВЕСЬ синтетический набор
realistic_synthetic['age_group'] = realistic_synthetic['age'].apply(get_age_group)

# 2. Разделяем на train и validation
from sklearn.model_selection import train_test_split

synth_train, synth_val = train_test_split(realistic_synthetic, test_size=0.2, random_state=42)

# 3. Обучаем препроцессор на синтетических train
preprocessor_synth = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_synth_train = synth_train.drop('diagnosis', axis=1)
y_synth_train = synth_train['diagnosis']
X_synth_val = synth_val.drop('diagnosis', axis=1)
y_synth_val = synth_val['diagnosis']

# Обучаем препроцессор
X_synth_train_processed = preprocessor_synth.fit_transform(X_synth_train)

# Преобразуем validation и оригинальные данные
X_synth_val_processed = preprocessor_synth.transform(X_synth_val)
X_orig_processed = preprocessor_synth.transform(X_train)  # оригинальные 4
X_test_processed = preprocessor_synth.transform(X_test)   # тестовые 2

# Конвертируем в тензоры
X_synth_train_tensor = torch.tensor(X_synth_train_processed, dtype=torch.float32)
y_synth_train_tensor = torch.tensor(y_synth_train.values, dtype=torch.float32).view(-1, 1)
X_synth_val_tensor = torch.tensor(X_synth_val_processed, dtype=torch.float32)
y_synth_val_tensor = torch.tensor(y_synth_val.values, dtype=torch.float32).view(-1, 1)
X_orig_tensor = torch.tensor(X_orig_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

print(f"\nРазмеры данных:")
print(f"Синтетические train: {X_synth_train_tensor.shape}")
print(f"Синтетические validation: {X_synth_val_tensor.shape}")
print(f"Оригинальные 4 примера: {X_orig_tensor.shape}")
print(f"Тестовые 2 примера: {X_test_tensor.shape}")



# Супер-простая модель для маленьких данных
class MicroNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # ПРОСТОЙ слой: вход -> 2 нейрона -> выход
        self.layer1 = nn.Linear(input_size, 2)
        self.layer2 = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

# Считаем параметры: Linear(7,2)=16, Linear(2,1)=3, всего 19

def train_with_all_checks(model, epochs=300):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        preds = model(X_synth_train_tensor)
        loss = criterion(preds, y_synth_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                # 1. На синтетических train
                train_acc = ((model(X_synth_train_tensor) > 0.5).float() == y_synth_train_tensor).float().mean()

                # 2. На синтетических val
                val_acc = ((model(X_synth_val_tensor) > 0.5).float() == y_synth_val_tensor).float().mean()

                # 3. На оригинальных 4 примерах
                orig_acc = ((model(X_orig_tensor) > 0.5).float() == y_train_tensor).float().mean()

                # 4. На тестовых 2 примерах
                test_acc = ((model(X_test_tensor) > 0.5).float() == y_test_tensor).float().mean()

                print(f"Эпоха {epoch + 1}:")
                print(f"  Synth Train: {train_acc:.1%}, Synth Val: {val_acc:.1%}")
                print(f"  Original 4: {orig_acc:.1%}, Test 2: {test_acc:.1%}")
                print()

    return model


print("\nОбучаем MicroNN на реалистичных синтетических данных...")
micro_model = MicroNN(input_size=8)
trained_micro = train_with_all_checks(micro_model)

print("\n" + "=" * 60)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ПРЕДСКАЗАНИЙ")
print("=" * 60)

with torch.no_grad():
    trained_micro.eval()

    # Вероятности для всех наборов
    synth_train_probs = trained_micro(X_synth_train_tensor)
    synth_val_probs = trained_micro(X_synth_val_tensor)
    orig_probs = trained_micro(X_orig_tensor)
    test_probs = trained_micro(X_test_tensor)

    print("\nТестовые предсказания с вероятностями:")
    for i in range(len(X_test_tensor)):
        prob = test_probs[i].item()
        real = y_test_tensor[i].item()
        pred = 1 if prob > 0.5 else 0
        correct = "✓" if pred == real else "✗"
        print(f"Пациент {i}: {prob:.1%} → {pred} (реально: {real}) {correct}")

    # Средняя уверенность модели
    print(f"\nСредняя уверенность в предсказаниях:")
    print(f"На синтетических train: {synth_train_probs.mean():.1%}")
    print(f"На синтетических val:   {synth_val_probs.mean():.1%}")
    print(f"На оригиналах:         {orig_probs.mean():.1%}")
    print(f"На тесте:              {test_probs.mean():.1%}")