import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Шаг 2: Определение архитектуры
class SimpleCNN(nn.Module):
    def __init__(self):
        # ЧТО ЭТО? Создаём слой с 10 фильтрами размером 3x3
        # Зачем 10 фильтров? Чтобы найти 10 разных типов признаков
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1) # Только 4 фильтра для наглядности
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.fc = nn.Linear(8 * 6 * 6, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 6 * 6)
        x = self.fc(x)
        return x

# 2. Функция для визуализации фильтров
def visualize_filters(conv_layer, epoch, title):
    filters = conv_layer.weight.data.cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(8,8))
    fig.suptitle(f'{title} - Epoch {epoch}')

    for i, ax in enumerate(axes.flat):
        if i < len(filters):
            # Визуализируем один фильтр (3x3)
            im = ax.imshow(filters[i, 0], cmap='viridis')
            ax.set_title(f'Filter {i}')
            ax.axis('off')

    plt.show()


# 3. Обучение с визуализацией

def train_with_visualization():
    # Загрузка данных
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterition = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    print("=== ФИЛЬТРЫ ДО ОБУЧЕНИЯ ===")
    visualize_filters(model.conv1, 0, "Conv1 Filters")

    for epoch in range(3):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterition(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"\n=== ПОСЛЕ ЭПОХИ {epoch + 1} ===")
        visualize_filters(model.conv1, epoch+1, "Conv1 Filters")

    print("Обрати внимание, как фильтры меняются!")
    print("Они начинают напоминать паттерны (линии, углы)")

train_with_visualization()