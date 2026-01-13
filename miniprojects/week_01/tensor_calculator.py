import numpy as np


class TensorCalculator:
    def __init__(self):
        """Инициализация калькулятора тензоров"""
        self.operations_log = []

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Поэлементное сложение двух тензоров.

        Args:
            a: Первый тензор
            b: Второй тензор

        Returns:
            Сумма тензоров

        Raises:
            ValueError: Если формы тензоров не совпадают
        """
        if a.shape != b.shape:
            raise ValueError(f"Формы не совпадают: {a.shape} != {b.shape}")

        result = a + b
        self.operations_log.append(f"add: {a.shape} + {b.shape} = {result.shape}")
        return result

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Матричное умножение.

        Args:
            a: Первая матрица (n x m)
            b: Вторая матрица (m x k)

        Returns:
            Результат умножения (n x k)

        Raises:
            ValueError: Если размерности не совместимы
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Размерности не совместимы для умножения: "
                f"{a.shape} и {b.shape}"
            )

        # Три способа сделать матричное умножение
        result = np.matmul(a, b)  # Способ 1
        # result = a @ b           # Способ 2 (оператор @)
        # result = np.dot(a, b)    # Способ 3 (для 2D массивов)

        self.operations_log.append(f"matmul: {a.shape} @ {b.shape} = {result.shape}")
        return result

    def elementwise_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Поэлементное умножение (не матричное!).
        """
        if a.shape != b.shape:
            raise ValueError(f"Формы не совпадают: {a.shape} != {b.shape}")

        result = a * b
        self.operations_log.append(f"elementwise_mul: {a.shape} * {b.shape}")
        return result

    def reshape(self, tensor: np.ndarray, new_shape: tuple) -> np.ndarray:
        """
        Изменение формы тензора.

        Args:
            tensor: Исходный тензор
            new_shape: Новая форма

        Returns:
            Тензор новой формы

        Raises:
            ValueError: Если новое форма имеет другое количество элементов
        """
        if np.prod(tensor.shape) != np.prod(new_shape):
            raise ValueError(
                f"Нельзя изменить форму {tensor.shape} на {new_shape}. "
                f"Количество элементов должно совпадать."
            )

        result = tensor.reshape(new_shape)
        self.operations_log.append(f"reshape: {tensor.shape} -> {new_shape}")
        return result

    def transpose(self, tensor: np.ndarray) -> np.ndarray:
        """
        Транспонирование тензора.
        Для 2D матриц это обычное транспонирование.
        Для ND тензоров можно указать оси.
        """
        result = tensor.T  # Для 2D матриц
        self.operations_log.append(f"transpose: {tensor.shape} -> {result.shape}")
        return result

    def get_log(self) -> list:
        """Возвращает историю операций"""
        return self.operations_log

    def clear_log(self):
        """Очищает историю операций"""
        self.operations_log.clear()


def test_calculator():
    """Тестирование всех операций калькулятора"""
    calc = TensorCalculator()

    print("=== Тест 1: Сложение ===")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    try:
        result = calc.add(a, b)
        print(f"a + b =\n{result}")
        print(f"Ожидаемый результат:\n{[[6, 8], [10, 12]]}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\n=== Тест 2: Матричное умножение ===")
    a = np.array([[1, 2], [3, 4]])  # 2x2
    b = np.array([[5, 6], [7, 8]])  # 2x2
    try:
        result = calc.matmul(a, b)
        print(f"a @ b =\n{result}")
        print(f"Ожидаемый результат:\n{[[19, 22], [43, 50]]}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\n=== Тест 3: Поэлементное умножение ===")
    try:
        result = calc.elementwise_multiply(a, b)
        print(f"a * b (elementwise) =\n{result}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\n=== Тест 4: Изменение формы ===")
    tensor = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
    try:
        result = calc.reshape(tensor, (3, 2))
        print(f"reshape((2,3) -> (3,2)) =\n{result}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\n=== Тест 5: Транспонирование ===")
    try:
        result = calc.transpose(tensor)
        print(f"Транспонирование {tensor.shape} -> {result.shape}")
        print(f"Результат:\n{result}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\n=== История операций ===")
    for op in calc.get_log():
        print(f"  - {op}")


if __name__ == "__main__":
    test_calculator()

    # Дополнительные примеры
    print("\n" + "=" * 50)
    print("Дополнительные примеры для понимания:")

    # Разница между @ и *
    print("\n1. Разница между @ (матричное умножение) и * (поэлементное):")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 0], [1, 2]])

    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"A @ B (матричное) = \n{A @ B}")
    print(f"A * B (поэлементное) = \n{A * B}")

    # Broadcasting
    print("\n2. Broadcasting (автоматическое расширение размерностей):")
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    vector = np.array([10, 20, 30])
    print(f"matrix + vector = \n{matrix + vector}")

    # Разные формы
    print("\n3. Создание тензоров разной размерности:")
    scalar = np.array(5)  # 0D - скаляр
    vector = np.array([1, 2, 3])  # 1D - вектор
    matrix = np.array([[1, 2], [3, 4]])  # 2D - матрица
    tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D

    print(f"scalar shape: {scalar.shape}")  # ()
    print(f"vector shape: {vector.shape}")  # (3,)
    print(f"matrix shape: {matrix.shape}")  # (2, 2)
    print(f"tensor_3d shape: {tensor_3d.shape}")  # (2, 2, 2)