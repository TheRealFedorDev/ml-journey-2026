import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)  # Показать все колонки
pd.set_option('display.width', 1000)        # Ширина вывода
data = [[0., 1., 2., 'please'],
        [3., 4., 5., 'speed'],
        [6., 7., 8., 'i_need_dis'],
        [9., 10., 11., 'meow'],
        [12., 13., 14., 'unknown_meme']]

columns = ['x', 'y', 'z', 'memes']
df = pd.DataFrame(data, columns=columns)
print("\n")
print(df, "\n", df.dtypes)

# 2
from sklearn.preprocessing import StandardScaler

numeric_columns = ['x', 'y', 'z']
X_numeric = df[numeric_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
print("\n")
print(X_numeric, "\n", X_numeric.shape)
print(X_scaled, "\n", X_scaled.shape)
print("\n")
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

X_new_1_numeric = df[numeric_columns]
X_new_1_scaled = scaler.transform(X_new_1_numeric)
X_new_1_scaled.mean(axis=0)
X_new_1_scaled.std(axis=0)

# 3
from sklearn.preprocessing import OneHotEncoder

text_col = df[['memes']]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(text_col)
print("\n")
print(text_col, "\n", X_encoded, "\n", encoder.categories_)

#4

scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns)
encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['memes']))

final_df = pd.concat([scaled_df, encoded_df], axis=1)

print(final_df, "\n", final_df.shape)

# 5

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), text_col)
])

# Делим данные
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

# Обучаем препроцессор ТОЛЬКО на train данных!
preprocessor.fit(df_train)

# Применяем к train и test
X_train = preprocessor.transform(df_train)
X_test = preprocessor.transform(df_test)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
