import tensorflow as tf
import numpy as np

print("="*60)
print("ПРОВЕРКА TENSORFLOW")
print("="*60)

# 1. Версия
print(f"TensorFlow версия: {tf.__version__}")

# 2. Доступные устройства
print("\nДоступные устройства:")
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print(f"  GPU: {len(gpus)} устройств")
print(f"  CPU: {len(cpus)} устройств")

if gpus:
    for gpu in gpus:
        print(f"    {gpu}")
    # Включим рост памяти GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 3. Создадим простой тензор
print("\nСоздание TF тензора:")
tensor = tf.constant([[1, 2], [3, 4]])
print(f"Tensor:\n{tensor}")
print(f"Shape: {tensor.shape}")
print(f"Dtype: {tensor.dtype}")
print(f"Numpy:\n{tensor.numpy()}")

# 4. Простейшие операции
print("\nОперации с тензорами:")
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])
print(f"a + b:\n{a + b}")
print(f"a * b (поэлементно):\n{a * b}")
print(f"matmul(a, b):\n{tf.matmul(a, b)}")

# 5. Автоматическое дифференцирование
print("\nАвтоматическое дифференцирование (GradientTape):")
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1
grad = tape.gradient(y, x)
print(f"f(x) = x² + 2x + 1")
print(f"f({x.numpy()}) = {y.numpy()}")
print(f"f'({x.numpy()}) = {grad.numpy()}")

# 6. Проверка Keras
print("\nПроверка Keras API:")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
print("Модель создана успешно!")
print(f"Архитектура: {model.summary()}")

print("\n" + "="*60)
print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! 🎉")
print("TensorFlow готов к работе!")
print("="*60)