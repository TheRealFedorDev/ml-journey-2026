import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models

# 1. АУГМЕНТАЦИИ ДАННЫХ (чтобы модель обобщала лучше)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Зеркальное отражение
    transforms.RandomRotation(10), # Поворот на ±10 градусов
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Нормализация RGB
    transforms.RandomHorizontalFlip(p=0.5),  # 50% зеркалим
    transforms.RandomRotation(degrees=15),   # Поворачиваем
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Меняем цвета
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # "Приближаем"
])

# 2. АРХИТЕКТУРА ДЛЯ CIFAR-10
class CIFARCNN(nn.Module):
    def __init__(self):
        super(CIFARCNN, self).__init__()
        # Свёрточные слои

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 3 канала (RGB) → 32
        self.bn1 = nn.BatchNorm2d(32) # Батч-нормализация (ускоряет обучение)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Пулинг
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25) # Регуляризация

        # Полносвязные слои
        # Размер после 3 пулингов: 32 → 16 → 8 → 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10) # 10 классов
        print(self.fc1, "-fc1", self.fc2, "-fc2")

    def forward(self,x):
        # [batch, 3, 32, 32]
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # → [batch, 32, 16, 16]
        x = self.dropout(x)

        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # → [batch, 64, 8, 8]
        x = self.dropout(x)

        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # → [batch, 128, 4, 4]
        x = self.dropout(x)

        # Выпрямляем
        x = x.view(-1, 128 * 4 * 4)  # → [batch, 2048]

        x = torch.relu(self.fc1(x))  # → [batch, 256]
        x = self.dropout(x)
        x = self.fc2(x)  # → [batch, 10]
        return x

# 2. ======= Transfer Learning ========

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Загрузка модели с современным синтаксисом (без warning):
weights = EfficientNet_B0_Weights.DEFAULT # или IMAGENET1K_V1
model = efficientnet_b0(weights=weights)


# Меняем классификатор (последний слой):
# EfficientNet имеет model.classifier[1] вместо model.fc

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 10) # 10 классов для CIFAR-10

# Замораживаем все слои кроме классификатора:
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True