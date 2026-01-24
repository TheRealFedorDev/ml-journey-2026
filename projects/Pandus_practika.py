# Добавляем новую строку с неизвестным мемом
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


new_row = {'x': 12.0, 'y': 13.0, 'z': 14.0, 'memes': 'unknown meme'}
scaler = StandardScaler()
new_df = pd.DataFrame([new_row])

numeric_columns = ['x', 'y', 'z']
# 1. Обрабатываем числовые признаки (используем уже обученный scaler!)
new_numeric = new_df[numeric_columns]
new_scaled = scaler.transform(new_numeric)
print("Масштабированные числовые признаки новой строки:")
print(new_scaled)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# 2. Обрабатываем текстовые признаки
new_text = new_df[['memes']]
try:
    # Попробуем преобразовать
    new_encoded = encoder.transform(new_text)
    print("\nЗакодированный текст новой строки:")
    print(new_encoded)
    print("Соответствующие колонки:", encoder.get_feature_names_out(['memes']))
except Exception as e:
    print(f"\nОШИБКА при кодировании: {e}")