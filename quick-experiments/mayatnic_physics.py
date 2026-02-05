import pygame
import math
import sys

pygame.init()

# Размеры экрана
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Настраиваемый физический маятник")
clock = pygame.time.Clock()

# Физические константы
GRAVITY_EARTH = 9.81  # м/с²


class Pendulum:
    def __init__(self, pivot_x, pivot_y, length, angle, mass, damping, gravity, scale=100):
        self.pivot = pygame.Vector2(pivot_x, pivot_y)
        self.length = length  # в метрах (масштабируется)
        self.scale = scale  # пикселей на метр
        self.length_px = length * scale  # в пикселях
        self.angle = angle
        self.angle_vel = 0.0
        self.angle_acc = 0.0
        self.mass = mass  # в кг
        self.damping = damping
        self.gravity = gravity  # м/с²
        # Увеличиваем влияние массы на размер шара
        self.ball_radius = max(15, int(mass * 2.5))  # Более заметное изменение размера
        self.trail = []
        self.show_trail = True
        self.dragging = False
        self.color = (255, 100, 100)
        self.paused = False

    def update(self, dt):
        if not self.dragging and not self.paused:
            # Уравнение движения маятника
            self.angle_acc = -(self.gravity / self.length) * math.sin(self.angle)
            self.angle_vel += self.angle_acc * dt
            self.angle_vel *= self.damping
            self.angle += self.angle_vel * dt
        self.add_trail()

    def add_trail(self):
        pos = self.get_ball_pos()
        self.trail.append((int(pos.x), int(pos.y)))
        if len(self.trail) > 200:
            self.trail.pop(0)

    def get_ball_pos(self):
        x = self.pivot.x + self.length_px * math.sin(self.angle)
        y = self.pivot.y + self.length_px * math.cos(self.angle)
        return pygame.Vector2(x, y)

    def get_period(self):
        # Период малых колебаний (приближенная формула)
        return 2 * math.pi * math.sqrt(self.length / self.gravity)

    def get_energy(self):
        # Полная механическая энергия
        height = self.length_px * (1 - math.cos(self.angle))
        kinetic = 0.5 * self.mass * (self.length * self.angle_vel) ** 2
        potential = self.mass * self.gravity * height / self.scale
        return kinetic + potential

    def start_drag(self, pos):
        ball_pos = self.get_ball_pos()
        if math.dist(pos, ball_pos) <= self.ball_radius + 10:
            self.dragging = True
            self.angle_vel = 0
            self.paused = True

    def drag(self, pos):
        if self.dragging:
            dx = pos[0] - self.pivot.x
            dy = pos[1] - self.pivot.y
            new_length_px = math.sqrt(dx * dx + dy * dy)
            self.length = max(0.1, new_length_px / self.scale)
            self.length_px = self.length * self.scale
            self.angle = math.atan2(dx, dy)

    def stop_drag(self):
        if self.dragging:
            self.dragging = False
            self.paused = False
            self.angle_vel = 0

    def draw(self, surface):
        ball_pos = self.get_ball_pos()

        # Рисуем след
        if self.show_trail and len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = int(150 * i / len(self.trail))
                color = (100, 200, 255)
                pygame.draw.circle(surface, color, self.trail[i], 3, 1)

        # Рисуем нить и шар
        pygame.draw.line(surface, (230, 230, 230), self.pivot, ball_pos, 4)
        pygame.draw.circle(surface, self.color, (int(ball_pos.x), int(ball_pos.y)), self.ball_radius)
        pygame.draw.circle(surface, (255, 255, 255), (int(ball_pos.x), int(ball_pos.y)), self.ball_radius, 3)

        # Рисуем точку крепления
        pygame.draw.circle(surface, (80, 80, 100), (int(self.pivot.x), int(self.pivot.y)), 12)
        pygame.draw.circle(surface, (120, 120, 140), (int(self.pivot.x), int(self.pivot.y)), 12, 3)


class Slider:
    def __init__(self, x, y, label, min_val, max_val, initial_val, unit="", description=""):
        self.rect = pygame.Rect(x, y, 300, 25)
        self.label = label
        self.min = min_val
        self.max = max_val
        self.value = initial_val
        self.unit = unit
        self.description = description
        self.dragging = False

    def draw(self, surface, font):
        # Фон слайдера
        pygame.draw.rect(surface, (60, 60, 70), self.rect, 0, 5)
        pygame.draw.rect(surface, (100, 100, 110), self.rect, 2, 5)

        # Заполненная часть
        fill_width = (self.value - self.min) / (self.max - self.min) * self.rect.width
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        pygame.draw.rect(surface, (80, 160, 255), fill_rect, 0, 5)

        # Ползунок
        pos = self.rect.x + fill_width
        pygame.draw.circle(surface, (255, 200, 100), (int(pos), self.rect.centery), 12)
        pygame.draw.circle(surface, (255, 255, 200), (int(pos), self.rect.centery), 12, 2)

        # Текст
        value_text = f"{self.value:.2f}{self.unit}"
        text = font.render(f"{self.label}: {value_text}", True, (240, 240, 240))
        surface.blit(text, (self.rect.x, self.rect.y - 30))

        # Описание
        desc_font = pygame.font.SysFont('arial', 14)
        desc_text = desc_font.render(self.description, True, (180, 180, 180))
        surface.blit(desc_text, (self.rect.x, self.rect.y - 15))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = max(0, min(event.pos[0] - self.rect.x, self.rect.width))
            self.value = self.min + (rel_x / self.rect.width) * (self.max - self.min)
            return True
        return False


class Button:
    def __init__(self, x, y, width, height, text, color=(80, 160, 255), hover_color=(100, 180, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.hovered = False

    def draw(self, surface, font):
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, 0, 8)
        pygame.draw.rect(surface, (200, 200, 220), self.rect, 3, 8)

        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


def draw_info_panel(surface, pendulum, font, small_font, time_elapsed):
    # Панель информации справа
    panel_rect = pygame.Rect(WIDTH - 450, 50, 400, HEIGHT - 100)
    pygame.draw.rect(surface, (30, 30, 40), panel_rect, 0, 15)
    pygame.draw.rect(surface, (60, 60, 80), panel_rect, 3, 15)

    y_offset = 80

    # Заголовок
    title = font.render("ФИЗИЧЕСКИЕ ПАРАМЕТРЫ", True, (255, 220, 100))
    surface.blit(title, (WIDTH - 430, y_offset))
    y_offset += 50

    # Параметры в реальных единицах
    period = pendulum.get_period()
    if period > 0:
        frequency = 1 / period
    else:
        frequency = 0

    params = [
        f"Длина маятника: {pendulum.length:.3f} м",
        f"Масса груза: {pendulum.mass:.2f} кг",
        f"Гравитация: {pendulum.gravity:.2f} м/с² ({pendulum.gravity / GRAVITY_EARTH:.2f} g)",
        f"Угол отклонения: {math.degrees(pendulum.angle):.1f}°",
        f"Угловая скорость: {pendulum.angle_vel:.3f} рад/с",
        f"Период колебаний: {period:.3f} с",
        f"Частота: {frequency:.3f} Гц",
        f"Полная энергия: {pendulum.get_energy():.3f} Дж",
        f"Демпфирование: {pendulum.damping:.4f}",
        f"Время симуляции: {time_elapsed:.1f} с",
        f"Размер шара: {pendulum.ball_radius} px"
    ]

    for i, param in enumerate(params):
        text = small_font.render(param, True, (220, 220, 240))
        surface.blit(text, (WIDTH - 430, y_offset + i * 35))

    # Земля для сравнения
    y_offset += len(params) * 35 + 20
    earth_text = font.render("СРАВНЕНИЕ С ЗЕМЛЁЙ", True, (100, 220, 100))
    surface.blit(earth_text, (WIDTH - 430, y_offset))
    y_offset += 40

    # Шкала сравнения гравитации
    g_ratio = pendulum.gravity / GRAVITY_EARTH
    g_bar_width = 300
    g_fill = min(2.0, g_ratio) * g_bar_width / 2.0  # Показываем до 2g

    pygame.draw.rect(surface, (50, 50, 60), (WIDTH - 430, y_offset, g_bar_width, 20), 0, 10)
    pygame.draw.rect(surface, (100, 220, 100), (WIDTH - 430, y_offset, g_fill, 20), 0, 10)
    pygame.draw.rect(surface, (150, 150, 170), (WIDTH - 430, y_offset, g_bar_width, 20), 2, 10)

    # Отметка для 1g
    mark_pos = WIDTH - 430 + (g_bar_width / 2.0)
    pygame.draw.line(surface, (255, 255, 200), (mark_pos, y_offset), (mark_pos, y_offset + 20), 2)

    g_text = small_font.render(f"Земля = 1.00g (9.81 м/с²) | Текущая: {g_ratio:.2f}g", True, (200, 240, 200))
    surface.blit(g_text, (WIDTH - 430, y_offset + 30))

    # Примеры гравитации на других небесных телах
    y_offset += 70
    examples_title = font.render("ГРАВИТАЦИЯ В СОЛНЕЧНОЙ СИСТЕМЕ", True, (100, 180, 255))
    surface.blit(examples_title, (WIDTH - 430, y_offset))
    y_offset += 40

    celestial_bodies = [
        ("Солнце", 274.0, 27.93),
        ("Меркурий", 3.70, 0.377),
        ("Венера", 8.87, 0.904),
        ("Земля", 9.81, 1.000),
        ("Луна", 1.62, 0.165),
        ("Марс", 3.72, 0.379),
        ("Юпитер", 24.79, 2.527),
        ("Сатурн", 10.44, 1.064),
        ("Уран", 8.87, 0.904),
        ("Нептун", 11.15, 1.137),
        ("Плутон", 0.62, 0.063),
    ]
    for i, (name, g_value, ratio) in enumerate(celestial_bodies):
        body_text = small_font.render(f"{name}: {g_value:.2f} м/с² ({ratio:.3f}g)", True, (180, 200, 255))
        surface.blit(body_text, (WIDTH - 430, y_offset + i * 25))

        # Кнопка для быстрой установки
        btn_rect = pygame.Rect(WIDTH - 120, y_offset + i * 25 - 5, 70, 20)
        btn_color = (60, 100, 160) if not btn_rect.collidepoint(pygame.mouse.get_pos()) else (80, 120, 180)
        pygame.draw.rect(surface, btn_color, btn_rect, 0, 5)
        btn_text = small_font.render("Установить", True, (220, 220, 255))
        surface.blit(btn_text, (WIDTH - 115, y_offset + i * 25 - 5))

        if pygame.mouse.get_pressed()[0] and btn_rect.collidepoint(pygame.mouse.get_pos()):
            # Ограничиваем значение гравитации максимумом слайдера (25)
            return min(g_value, 274.0)

    return None


def main():
    pendulum = Pendulum(
        pivot_x=WIDTH * 0.4,
        pivot_y=200,
        length=1.5,  # 1.5 метра
        angle=math.radians(60),
        mass=5.0,  # 5 кг
        damping=0.998,
        gravity=9.81,  # Земная гравитация
        scale=150  # 150 пикселей на метр
    )

    font_large = pygame.font.SysFont('arial', 28, bold=True)
    font_medium = pygame.font.SysFont('arial', 22)
    font_small = pygame.font.SysFont('arial', 18)

    # Слайдеры с физическими пояснениями
    sliders = [
        Slider(50, HEIGHT - 400, "Гравитация", 1.0, 25.0, pendulum.gravity, " м/с²", "Ускорение свободного падения"),
        Slider(50, HEIGHT - 350, "Демпфирование", 0.990, 1.000, pendulum.damping, "", "Потеря энергии за колебание"),
        Slider(50, HEIGHT - 300, "Масса груза", 0.1, 20.0, pendulum.mass, " кг", "Масса груза (не влияет на период)"),
        Slider(50, HEIGHT - 250, "Длина нити", 0.5, 3.0, pendulum.length, " м",
               "Длина от точки крепления до центра массы"),
        Slider(50, HEIGHT - 200, "Начальный угол", 5.0, 175.0, math.degrees(pendulum.angle), "°",
               "Начальный угол отклонения"),
        Slider(50, HEIGHT - 150, "Масштаб", 50, 300, pendulum.scale, " px/м", "Пикселей на метр для визуализации"),
    ]

    # Кнопки управления
    buttons = [
        Button(50, HEIGHT - 100, 180, 40, "СБРОС", (255, 100, 100)),
        Button(250, HEIGHT - 100, 180, 40, "ПАУЗА/ПУСК", (100, 200, 100)),
        Button(450, HEIGHT - 100, 180, 40, "СЛЕД ВКЛ/ВЫКЛ", (100, 150, 255)),
        Button(650, HEIGHT - 100, 180, 40, "ОЧИСТИТЬ СЛЕД", (255, 180, 100)),
    ]

    running = True
    time_elapsed = 0.0

    while running:
        dt = clock.tick(120) / 1000.0  # 120 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    pendulum.paused = not pendulum.paused
                elif event.key == pygame.K_r:
                    pendulum = Pendulum(
                        pivot_x=WIDTH * 0.4,
                        pivot_y=200,
                        length=sliders[3].value,
                        angle=math.radians(sliders[4].value),
                        mass=sliders[2].value,
                        damping=sliders[1].value,
                        gravity=sliders[0].value,
                        scale=sliders[5].value
                    )
                    time_elapsed = 0.0
                elif event.key == pygame.K_c:
                    pendulum.trail.clear()
                elif event.key == pygame.K_UP:
                    pendulum.gravity = min(25.0, pendulum.gravity + 1.0)
                    sliders[0].value = pendulum.gravity
                elif event.key == pygame.K_DOWN:
                    pendulum.gravity = max(1.0, pendulum.gravity - 1.0)
                    sliders[0].value = pendulum.gravity

            # Обработка слайдеров
            for slider in sliders:
                if slider.handle_event(event):
                    # Обновляем маятник при изменении параметров
                    pendulum.length = sliders[3].value
                    pendulum.length_px = pendulum.length * pendulum.scale
                    pendulum.gravity = sliders[0].value
                    pendulum.damping = sliders[1].value
                    pendulum.mass = sliders[2].value
                    pendulum.angle = math.radians(sliders[4].value)
                    pendulum.scale = sliders[5].value
                    # Увеличиваем влияние массы на размер шара
                    pendulum.ball_radius = max(15, int(pendulum.mass * 2.5))

            # Обработка кнопок
            mouse_pos = pygame.mouse.get_pos()
            for button in buttons:
                button.check_hover(mouse_pos)
                if button.handle_event(event):
                    if button.text == "СБРОС":
                        pendulum = Pendulum(
                            pivot_x=WIDTH * 0.4,
                            pivot_y=200,
                            length=sliders[3].value,
                            angle=math.radians(sliders[4].value),
                            mass=sliders[2].value,
                            damping=sliders[1].value,
                            gravity=sliders[0].value,
                            scale=sliders[5].value
                        )
                        time_elapsed = 0.0
                    elif button.text == "ПАУЗА/ПУСК":
                        pendulum.paused = not pendulum.paused
                    elif button.text == "СЛЕД ВКЛ/ВЫКЛ":
                        pendulum.show_trail = not pendulum.show_trail
                    elif button.text == "ОЧИСТИТЬ СЛЕД":
                        pendulum.trail.clear()

            # Перетаскивание маятника
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pendulum.start_drag(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    pendulum.stop_drag()
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    pendulum.drag(event.pos)

        # Обновление времени
        if not pendulum.paused and not pendulum.dragging:
            time_elapsed += dt

        # Обновление маятника
        pendulum.update(dt)

        # Отрисовка
        screen.fill((15, 15, 25))  # Темный космический фон

        # Сетка для масштаба - исправлено: используем целые числа
        grid_size = int(pendulum.scale)  # 1 метр в пикселях
        if grid_size > 10:  # Не рисуем слишком мелкую сетку
            for x in range(0, WIDTH, grid_size):
                pygame.draw.line(screen, (40, 40, 50), (x, 0), (x, HEIGHT), 1)
            for y in range(0, HEIGHT, grid_size):
                pygame.draw.line(screen, (40, 40, 50), (0, y), (WIDTH, y), 1)

        # Маятник
        pendulum.draw(screen)

        # Информационная панель
        celestial_g = draw_info_panel(screen, pendulum, font_medium, font_small, time_elapsed)
        if celestial_g is not None:
            pendulum.gravity = celestial_g
            sliders[0].value = celestial_g

        # Панель управления слева
        control_panel = pygame.Rect(30, 50, 400, HEIGHT - 200)
        pygame.draw.rect(screen, (25, 25, 35), control_panel, 0, 15)
        pygame.draw.rect(screen, (50, 50, 70), control_panel, 3, 15)

        # Заголовок панели управления
        control_title = font_large.render("УПРАВЛЕНИЕ МАЯТНИКОМ", True, (255, 200, 100))
        screen.blit(control_title, (50, 70))

        # Подсказки
        hints_y = HEIGHT - 50
        hints = [
            "Управление: ЛКМ - тащить маятник, ПРОБЕЛ - пауза, R - сброс, C - очистить след",
            "↑/↓ - быстро менять гравитацию, ESC - выход",
            "Нажмите на кнопку с названием планеты, чтобы установить её гравитацию"
        ]

        for i, hint in enumerate(hints):
            hint_text = pygame.font.SysFont('arial', 14).render(hint, True, (180, 180, 220))
            screen.blit(hint_text, (50, hints_y + i * 20))

        # Слайдеры
        for slider in sliders:
            slider.draw(screen, font_small)

        # Кнопки
        for button in buttons:
            button.draw(screen, font_small)

        # Статус паузы
        if pendulum.paused:
            pause_text = font_large.render("ПАУЗА", True, (255, 100, 100))
            screen.blit(pause_text, (WIDTH // 2 - 50, 30))

        # FPS
        fps_text = font_small.render(f"FPS: {int(clock.get_fps())}", True, (200, 200, 200))
        screen.blit(fps_text, (WIDTH - 150, 20))

        # Статус перетаскивания
        if pendulum.dragging:
            drag_text = font_small.render("ПЕРЕТАСКИВАНИЕ", True, (100, 200, 255))
            screen.blit(drag_text, (WIDTH // 2 - 80, 70))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()