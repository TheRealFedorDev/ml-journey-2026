import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import urllib.request
import gzip
import os

print("=== СОЗДАЁМ МИНИ-MNIST ВРУЧНУЮ (если не грузится) ===")


def create_mini_mnist():
    """Создаём маленький датасет для демонстрации, если MNIST не грузится"""
    np.random.seed(42)

    # Создаём 1000 "цифр" 28x28
    n_samples = 1000
    images = []
    labels = []

    for i in range(n_samples):
        # Создаём простые паттерны, похожие на цифры
        img = np.zeros((28, 28))
        label = i % 10  # 10 классов

        # Рисуем разные паттерны для разных цифр
        if label == 0:  # Круг
            img[8:20, 8:20] = 0.8  # Заполненный круг
            img[10:18, 10:18] = 0.2  # Дырка
        elif label == 1:  # Вертикальная линия
            img[5:23, 13:15] = 0.9
        elif label == 2:  # Две горизонтальные линии
            img[8:10, 5:23] = 0.9
            img[18:20, 5:23] = 0.9
        elif label == 7:  # Наклонная линия
            for j in range(28):
                if 5 <= j <= 22:
                    img[j, 20 - j // 2] = 0.9

        # Добавляем немного шума
        img += np.random.normal(0, 0.1, (28, 28))
        img = np.clip(img, 0, 1)

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


# Создаём или загружаем данные
try:
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    print("✅ MNIST успешно загружен!")
except:
    print("⚠️  Не удалось загрузить MNIST. Создаём мини-версию...")
    images, labels = create_mini_mnist()
    # Преобразуем в тензоры
    images_tensor = torch.FloatTensor(images).unsqueeze(1)  # [1000, 1, 28, 28]
    labels_tensor = torch.LongTensor(labels)  # [1000]
    trainset = TensorDataset(images_tensor, labels_tensor)
    print(f"✅ Создан мини-датасет: {len(trainset)} изображений")


# 1. Простая CNN (как в твоём примере) - ИСПРАВЛЕННАЯ ВЕРСИЯ!
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Первый свёрточный слой: 1 входной канал → 4 выходных канала
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
                               kernel_size=3, padding=1)

        # Пулинг (уменьшение в 2 раза)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Второй свёрточный слой: 4 входных канала → 8 выходных
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8,
                               kernel_size=3)  # НЕТ padding!

        # ВАЖНО: Посчитаем размер после всех преобразований
        # 28x28 → conv1(+padding) → 28x28 → pool → 14x14
        # → conv2(без padding) → 12x12 → pool → 6x6
        # Итого: 8 каналов * 6 * 6 = 288 признаков
        self.fc = nn.Linear(8 * 6 * 6, 10)  # 10 классов цифр

    def forward(self, x):
        # Размеры будем отслеживать в комментариях
        x = torch.relu(self.conv1(x))  # [batch, 4, 28, 28]
        x = self.pool(x)  # [batch, 4, 14, 14]
        x = torch.relu(self.conv2(x))  # [batch, 8, 12, 12]
        x = self.pool(x)  # [batch, 8, 6, 6]

        # Выпрямляем для полносвязного слоя
        x = x.view(-1, 8 * 6 * 6)  # [batch, 288]
        x = self.fc(x)  # [batch, 10]
        return x


# 2. Функция для визуализации фильтров - УЛУЧШЕННАЯ!
def visualize_filters(conv_layer, epoch, title, save=False):
    """
    Визуализирует фильтры свёрточного слоя
    """
    # Получаем веса фильтров
    filters = conv_layer.weight.data.cpu().numpy()
    print(f"\n{title}: {filters.shape}")  # Например: (4, 1, 3, 3)

    # filters.shape = [out_channels, in_channels, kernel_height, kernel_width]
    n_filters = filters.shape[0]  # сколько фильтров
    n_input_channels = filters.shape[1]  # обычно 1 для ч/б

    fig, axes = plt.subplots(1, n_filters, figsize=(3 * n_filters, 3))
    if n_filters == 1:
        axes = [axes]

    fig.suptitle(f'{title} - Эпоха {epoch}', fontsize=14)

    for i in range(n_filters):
        ax = axes[i]
        # Берем первый (и единственный) входной канал
        filter_img = filters[i, 0]  # размер [3, 3]

        # Визуализируем
        im = ax.imshow(filter_img, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f'Фильтр {i}')
        ax.axis('off')

        # Добавляем значения весов
        for row in range(3):
            for col in range(3):
                text = ax.text(col, row, f'{filter_img[row, col]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()

    if save:
        plt.savefig(f'filters_epoch_{epoch}.png', dpi=100)
    plt.show()

    # Выводим статистику
    print(f"  Min weight: {filters.min():.3f}, Max weight: {filters.max():.3f}")
    print(f"  Mean: {filters.mean():.3f}, Std: {filters.std():.3f}")


# 3. Обучение с визуализацией - УПРОЩЕННОЕ!
def train_with_visualization():
    # Создаём DataLoader
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    # Инициализируем модель
    model = SimpleCNN()

    # Проверяем архитектуру
    print("\n=== АРХИТЕКТУРА МОДЕЛИ ===")
    print(f"conv1 weights shape: {model.conv1.weight.shape}")  # [4, 1, 3, 3]
    print(f"conv2 weights shape: {model.conv2.weight.shape}")  # [8, 4, 3, 3]

    # Создаём оптимизатор и функцию потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("\n=== ФИЛЬТРЫ ДО ОБУЧЕНИЯ (СЛУЧАЙНЫЕ) ===")
    visualize_filters(model.conv1, epoch=0, title="Conv1 - Начало")

    # ОБУЧЕНИЕ на 3 эпохах
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Статистика
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Выводим статистику эпохи
        accuracy = 100. * correct / total
        print(f"\nЭпоха {epoch + 1}:")
        print(f"  Loss: {total_loss / len(trainloader):.4f}")
        print(f"  Accuracy: {accuracy:.1f}%")

        # Визуализируем фильтры после эпохи
        visualize_filters(model.conv1, epoch=epoch + 1,
                          title=f"Conv1 - После эпохи {epoch + 1}")

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)

    # Финальная визуализация
    print("\n=== ЧТО ПРОИСХОДИЛО С ФИЛЬТРАМИ? ===")
    print("1. Фильтры были случайными (близко к 0)")
    print("2. После обучения веса стали больше по модулю")
    print("3. Каждый фильтр 'специализировался' на своём паттерне")
    print("4. Некоторые стали 'активаторами' (положительные веса)")
    print("5. Другие стали 'ингибиторами' (отрицательные веса)")


# 4. ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ: Что делают фильтры?
def demonstrate_filter_effect():
    """Показывает, как фильтр преобразует изображение"""
    print("\n=== КАК ФИЛЬТР РАБОТАЕТ С ИЗОБРАЖЕНИЕМ? ===")

    # Создаём тестовое изображение (вертикальная линия)
    test_image = torch.zeros(1, 1, 7, 7)  # [1, 1, 7, 7]
    test_image[0, 0, :, 3] = 1.0  # Вертикальная линия в центре

    # Создаём фильтр для вертикальных линий
    vertical_filter = torch.tensor([
        [[[1.0, 0.0, -1.0],
          [1.0, 0.0, -1.0],
          [1.0, 0.0, -1.0]]]
    ])  # [1, 1, 3, 3]

    # Применяем свёртку
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    conv.weight.data = vertical_filter

    with torch.no_grad():
        result = conv(test_image)

    # Визуализируем
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(test_image[0, 0].numpy(), cmap='gray')
    axes[0].set_title("Исходное изображение\n(вертикальная линия)")
    axes[0].axis('off')

    axes[1].imshow(vertical_filter[0, 0].numpy(), cmap='coolwarm')
    axes[1].set_title("Фильтр для\nвертикальных линий")
    axes[1].axis('off')

    axes[2].imshow(result[0, 0].numpy(), cmap='gray')
    axes[2].set_title("Результат свёртки\n(яркая линия в центре)")
    axes[2].axis('off')

    plt.suptitle("Демонстрация работы фильтра", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nНаблюдение:")
    print("Фильтр для вертикальных линий даёт СИЛЬНЫЙ отклик")
    print("на вертикальных линиях и СЛАБЫЙ на горизонтальных")


# ЗАПУСКАЕМ ВСЁ
if __name__ == "__main__":
    print("ЗАПУСК ОБУЧЕНИЯ CNN С ВИЗУАЛИЗАЦИЕЙ")
    print("=" * 50)

    # Демонстрация работы фильтра
    demonstrate_filter_effect()

    # Основное обучение
    train_with_visualization()
