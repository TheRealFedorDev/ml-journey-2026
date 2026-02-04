import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

if __name__ == '__main__':
    torch.set_num_threads(4)  # Ограничиваем потоки для CPU
    print("Загрузка CIFAR-10...")
    IMG_SIZE = 64
    # Аугментации для тренировочных данных
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 статы!
                             (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Центральный кроп для теста
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Загрузка датасетов
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform_test)

    # Создание DataLoader'ов
    trainloader = DataLoader(trainset, batch_size=64,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64,
                            shuffle=False, num_workers=2)

    # Названия классов CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"   Обучающих изображений: {len(trainset)}")
    print(f"   Тестовых изображений: {len(testset)}")


    # 2. АРХИТЕКТУРА ДЛЯ CIFAR-10
    class CIFARCNN(nn.Module):
        def __init__(self):
            super(CIFARCNN, self).__init__()
            self.features = nn.Sequential(
                # Первый блок
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),

                # Второй блок
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),

                # Третий блок
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
            )

            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )

        def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # 2. ======= Transfer Learning ========

    from torchvision.models import resnet18, ResNet18_Weights
    from tqdm import tqdm

    # Загрузка модели с современным синтаксисом
    weights = ResNet18_Weights.DEFAULT # или IMAGENET1K_V1
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    ## Меняем классификатор (последний слой):
    ## EfficientNet имеет model.classifier[1] вместо model.fc

    model.fc = nn.Linear(model.fc.in_features, 10)

    nn.init.xavier_uniform_(model.fc.weight)
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias)


    # Стратегия разморозки
    def unfreeze_layers(model, num_layers=3):
        """Размораживает последние num_layers слоёв"""
        # Получаем все параметры
        params = list(model.named_parameters())

        # Размораживаем только последние num_layers
        for i, (name, param) in enumerate(reversed(params)):
            if i < num_layers:
                param.requires_grad = True
                print(f"✓ Разморожен: {name}")
            else:
                param.requires_grad = False


    unfreeze_layers(model, num_layers=3)

    # Проверка forward pass
    test_img, _ = next(iter(trainloader))
    print(f"Input shape: {test_img.shape}")  # Должно быть [64, 3, 224, 224]
    with torch.no_grad():
        output = model(test_img[:2])
        print(f"Output shape: {output.shape}")  # Должно быть [1, 10]
        print(f"Model output sample: {output[0][:3]}")  # Первые 3 вероятности
    def train_model(model1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                     lr=0.1,           # БОЛЬШОЙ learning rate для CIFAR
                     momentum=0.9,
                     weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        epochs = 50
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            model.train()
            with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}") as pbar:
                for batch_idx, (images, labels) in enumerate(trainloader):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    batch_total = labels.size(0)
                    batch_correct = predicted.eq(labels).sum().item()

                    correct += batch_correct
                    total += batch_total
                    current_acc = 100. * correct / total
                    avg_loss = total_loss / (batch_idx + 1)

                    # Обновляем прогресс-бар
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.1f}%',
                        'batch_acc': f'{100. * batch_correct / batch_total:.1f}%'
                    })

            # Итог эпохи
            epoch_acc = 100. * correct / total
            print(f"\n✅ Эпоха {epoch + 1} завершена: "
                  f"Loss: {total_loss / len(trainloader):.4f}, "
                  f"Accuracy: {epoch_acc:.1f}%")




    train_model(model)

