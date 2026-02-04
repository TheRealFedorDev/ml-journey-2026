def count_balls_2d_triangle(base):
    return base * (base + 1) // 2

base = 11
result = count_balls_2d_triangle(base)
print(f"Для треугольника с основанием {base} нужно {result} шариков")

