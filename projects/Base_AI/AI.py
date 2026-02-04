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


print("Создаём реалистичные синтетические данные...")
realistic_synthetic = create_better_synthetic_data(200)
print(f"Создано {len(realistic_synthetic)} реалистичных пациентов")
print("\nРаспределение диагнозов:")
print(realistic_synthetic['diagnosis'].value_counts())
print("\nКорреляция возраста и давления:")
print(realistic_synthetic[['age', 'blood_pressure']].corr())


# ==================== 2. СОЗДАНИЕ НЕЙРОСЕТИ ====================
import torch.nn as nn
import torch

class MedicalNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Здесь создаём слои
        self.hidden = nn.Linear(input_size, 4)
        self.output = nn.Linear(4, 1)
    def forward(self, x):
        # Здесь определяем, как данные проходят через слои
        x = self.hidden(x) # через первый слой
        x = torch.relu(x) # функция активации
        x = self.output(x) # через второй слой
        x = torch.sigmoid(x) # для получения вероятности
        return x

# Узнаем размерность входных данных
# X_train_tensor.shape = (4, 8) - 4 пациента, 8 признаков
# Нам нужно 8 - это индекс 1

class SimpleMedicalNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Здесь создаём слои
        self.hidden = nn.Linear(input_size, 2)
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        # Здесь определяем, как данные проходят через слои
        x = self.hidden(x) # через первый слой
        x = torch.relu(x) # функция активации
        x = self.output(x) # через второй слой
        x = torch.sigmoid(x) # для получения вероятности
        return x

# Создаём и проверяем
simple_model = SimpleMedicalNN(input_size=7)
print("Упрощённая модель:")
print(simple_model)

# Считаем параметры
simple_params = sum(p.numel() for p in simple_model.parameters())
print(f"Параметров: {simple_params}")
print(f"Было: 37, Стало: {simple_params}")

# Создаём новую модель
augmented_model = SimpleMedicalNN(input_size=7)


# Обучаем на расширенных данных
def train_on_augmented(model, epochs=200):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Обучаем на ВСЕХ расширенных данных
        predictions = model(X_augmented_tensor)
        loss = criterion(predictions, y_augmented_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                # Проверяем на оригинальных train
                orig_train_preds = model(X_train_tensor)
                orig_train_acc = ((orig_train_preds > 0.5).float() == y_train_tensor).float().mean()

                # Проверяем на test
                test_preds = model(X_test_tensor)
                test_acc = ((test_preds > 0.5).float() == y_test_tensor).float().mean()

                print(f"Эпоха {epoch + 1}: Orig Train Acc = {orig_train_acc:.1%}, Test Acc = {test_acc:.1%}")

    return model


print("\nОбучаем на расширенных данных...")
trained_augmented = train_on_augmented(augmented_model)

class DropoutMedicalNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


# Важно: при оценке модели нужно отключить dropout!
def evaluate_model(model, X, y):
    model.eval()  # Переводим в режим оценки (выключаем dropout)
    with torch.no_grad():
        predictions = model(X)
        binary = (predictions > 0.5).float()
        accuracy = (binary == y).float().mean()
    model.train()  # Возвращаем в режим обучения
    return accuracy.item()


print("\nСоздаём модель с Dropout...")
dropout_model = DropoutMedicalNN(input_size=7)
print(dropout_model)





input_size = X_train_tensor.shape[1]  # = 8
print(f"Входной размер: {input_size}")

# Создаём модель
model = MedicalNN(input_size=7)
print("Модель создана!")
print(model)

with torch.no_grad():
    all_predictions = model(X_train_tensor)  # подставь X_train_tensor
    print("Предсказания для всех пациентов:")
    print(all_predictions)

    # Преобразуй в бинарные предсказания (0 или 1)
    binary_predictions = (all_predictions > 0.5).float()
    print("\nБинарные предсказания (порог 0.5):")
    print(binary_predictions)

    # Сравни с реальными диагнозами
    print("\nРеальные диагнозы:")
    print(y_train_tensor)

# Проверяем на одном пациенте
one_patient = X_train_tensor[0:1]  # берем первого пациента
print(f"\nПациент features (shape {one_patient.shape}):")
print(one_patient)

# Предсказание (пока модель не обучена!)
with torch.no_grad():
    prediction = model(one_patient)
    print(f"\nПредсказание (сырое): {prediction}")
    print(f"Это означает: {prediction.item():.1%} вероятность болезни")


# Посчитай реальное количество параметров
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров : {total_params}")

# Выведи веса первого слоя (до обучения)
print("\nВеса скрытого слоя (до обучения):")
print(model.hidden.weight)
print(f"Shape: {model.hidden.weight.shape}")

# Выведи смещения первого слоя
print("\nСмещения скрытого слоя:")
print(model.hidden.bias)

# Твои предсказания:
print(all_predictions)
# Бинарные предсказания (порог 0.5):
print(binary_predictions)
# Реальные диагнозы:
print(y_train_tensor)

# Считаем accuracy вручную
correct = (binary_predictions == y_train_tensor).sum().item()
total = len(y_train_tensor)
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.1f}%")


# ==================== 3. ОБУЧЕНИЕ НЕЙРОСЕТИ ====================

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

print(f"Функция потерь: {criterion}")
print(f"Оптимизатор: {optimizer}")

def train_one_epoch():
    predictions = model(X_train_tensor)  # модель делает предсказания
    loss = criterion(predictions, y_train_tensor)  # считаем ошибку

    optimizer.zero_grad() # ОБНУЛЯЕМ старые градиенты (важно!)
    loss.backward() # ВЫЧИСЛЯЕМ новые градиенты (производные)
    optimizer.step()  # ОБНОВЛЯЕМ веса используя градиенты и lr

    return loss.item()

# Запускаем одну эпоху
loss_value = train_one_epoch()
print(f"Loss после 1 эпохи: {loss_value:.4f}")

# Проверяем, изменились ли предсказания
with torch.no_grad():
    new_predictions = model(X_train_tensor)
    print(f"Предсказания после 1 эпохи: {new_predictions}")


def train_model(model, epochs=200):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Прямой проход
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)

        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                # Train accuracy
                train_preds = model(X_train_tensor)
                train_acc = ((train_preds > 0.5).float() == y_train_tensor).float().mean()

                # Test accuracy
                test_preds = model(X_test_tensor)
                test_acc = ((test_preds > 0.5).float() == y_test_tensor).float().mean()

                print(f"Эпоха {epoch + 1}: Train Acc = {train_acc:.1%}, Test Acc = {test_acc:.1%}")

    return model

model = train_model(model)


def train_with_early_stopping(model, patience=5, max_epochs=500):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_test_acc = 0
    no_improve = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Обучение
        model.train()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Каждые 10 эпох проверяем
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_tensor)
                test_acc = ((test_preds > 0.5).float() == y_test_tensor).float().mean()

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    no_improve = 0
                    best_model_state = model.state_dict().copy()
                    print(f"Эпоха {epoch + 1}: Новый лучший Test Acc = {test_acc:.1%}")
                else:
                    no_improve += 1

                # Проверяем раннюю остановку
                if no_improve >= patience:
                    print(f"⏹️  Ранняя остановка на эпохе {epoch + 1}")
                    model.load_state_dict(best_model_state)
                    break

    return model


print("\nОбучаем с ранней остановкой...")
early_stop_model = DropoutMedicalNN(input_size=7)
trained_early = train_with_early_stopping(early_stop_model)


def train_model(model, epochs=200):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Прямой проход
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)

        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                # Train accuracy
                train_preds = model(X_train_tensor)
                train_acc = ((train_preds > 0.5).float() == y_train_tensor).float().mean()

                # Test accuracy
                test_preds = model(X_test_tensor)
                test_acc = ((test_preds > 0.5).float() == y_test_tensor).float().mean()

                print(f"Эпоха {epoch + 1}: Train Acc = {train_acc:.1%}, Test Acc = {test_acc:.1%}")

    return model


print("\nОбучаем упрощённую модель...")
trained_simple = train_model(simple_model)


print("\n" + "=" * 50)
print("4. ТЕСТИРОВАНИЕ НА НОВЫХ ДАННЫХ:")
print("=" * 50)

print("\n" + "=" * 60)
print("СРАВНЕНИЕ: МОДЕЛЬ ОБУЧЕННАЯ НА РАСШИРЕННЫХ ДАННЫХ")
print("=" * 60)

# Проверяем на разных наборах
with torch.no_grad():
    trained_augmented.eval()

    # 1. Оригинальные train (4 пациента)
    orig_train_preds = trained_augmented(X_train_tensor)
    orig_train_acc = ((orig_train_preds > 0.5).float() == y_train_tensor).float().mean()

    # 2. Расширенные train (54 пациента)
    aug_train_preds = trained_augmented(X_augmented_tensor)
    aug_train_acc = ((aug_train_preds > 0.5).float() == y_augmented_tensor).float().mean()

    # 3. Test (2 пациента)
    test_preds = trained_augmented(X_test_tensor)
    test_acc = ((test_preds > 0.5).float() == y_test_tensor).float().mean()

    print(f"Accuracy на оригинальных train: {orig_train_acc:.1%}")
    print(f"Accuracy на расширенных train:  {aug_train_acc:.1%}")
    print(f"Accuracy на тестовых данных:    {test_acc:.1%}")
    print(f"Loss на тесте: {criterion(test_preds, y_test_tensor):.4f}")

    # Выводим предсказания с вероятностями
    print("\nПодробные предсказания на тесте:")
    for i in range(len(X_test_tensor)):
        prob = test_preds[i].item()
        real = y_test_tensor[i].item()
        pred = 1 if prob > 0.5 else 0
        correct = "✓" if pred == real else "✗"
        print(f"Пациент {i}: Вероятность = {prob:.2%}, Предсказано = {pred}, Реально = {real} {correct}")


with torch.no_grad():
    # Предсказания на test данных (которые модель не видела при обучении)
    test_preds = model(X_test_tensor)
    print(f"\nТестовые предсказания (вероятности):")
    print(test_preds)

    # Бинарные предсказания
    test_binary = (test_preds > 0.5).float()
    print(f"\nБинарные предсказания на тесте:")
    print(test_binary)

    # Реальные тестовые значения
    print(f"\nРеальные тестовые значения:")
    print(y_test_tensor)

    # Test accuracy
    test_accuracy = (test_binary == y_test_tensor).float().mean() * 100
    print(f"\nAccuracy на тестовых данных: {test_accuracy:.1f}%")

    # Loss на тесте
    test_loss = criterion(test_preds, y_test_tensor)
    print(f"Loss на тестовых данных: {test_loss:.4f}")

# Тестирование на тестовых данных
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_accuracy = ((test_preds > 0.5).float() == y_test_tensor).float().mean() * 100
    test_loss = criterion(test_preds, y_test_tensor)

    print(f"РЕЗУЛЬТАТЫ:")
    print(f"Train Accuracy: 100.0% (из твоего вывода)")
    print(f"Test Accuracy: {test_accuracy:.1f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Если test accuracy низкая — это переобучение
    if test_accuracy < 70:
        print("\n⚠️  ВОЗМОЖНОЕ ПЕРЕОБУЧЕНИЕ!")
        print("Train accuracy >> Test accuracy")

print("\n" + "=" * 60)
print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("=" * 60)

models = {
    "Базовая (переобученная)": model,  # твоя исходная модель
    "Упрощённая": trained_simple,
    "С Dropout": trained_early
}

for name, model_obj in models.items():
    # Проверяем на train
    train_acc = evaluate_model(model_obj, X_train_tensor, y_train_tensor)

    # Проверяем на test
    test_acc = evaluate_model(model_obj, X_test_tensor, y_test_tensor)

    print(f"\n{name}:")
    print(f"  Train Accuracy: {train_acc:.1%}")
    print(f"  Test Accuracy:  {test_acc:.1%}")
    print(f"  Разница: {abs(train_acc - test_acc):.1%}")

    if train_acc > 0.9 and test_acc < 0.7:
        print("  ❌ СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ")
    elif abs(train_acc - test_acc) < 0.2:
        print("  ✅ ХОРОШЕЕ ОБОБЩЕНИЕ")