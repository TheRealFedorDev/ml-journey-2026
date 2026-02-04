import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

if __name__ == '__main__':
    torch.set_num_threads(4)
    print("=== CIFAR-10 С ADAPTED RESNET18 ===")

    # 1. ТРАНСФОРМЫ (одинаковые для train и test)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 статы
                             (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # ТАКИЕ ЖЕ как train!
                             (0.2470, 0.2435, 0.2616))
    ])

    # 2. ДАННЫЕ
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=128,  # ↑ batch_size для скорости
                             shuffle=True, num_workers=0)  # 0 для Windows
    testloader = DataLoader(testset, batch_size=128,
                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Обучающих: {len(trainset)}, Тестовых: {len(testset)}")

    # 3. МОДЕЛЬ С ПРЕДОБУЧЕНИЕМ!
    print("\n=== ЗАГРУЗКА ПРЕДОБУЧЕННОЙ RESNET18 ===")
    weights = ResNet18_Weights.IMAGENET1K_V1  # ИЛИ .DEFAULT
    model = resnet18(weights=weights)  # ← С ПРЕДОБУЧЕНИЕМ!

    # 4. АДАПТАЦИЯ ДЛЯ CIFAR-10 (32×32)
    print("Адаптация для CIFAR-10 (32×32)...")

    # Изменяем первый слой для 32×32
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Копируем веса из оригинального conv1 (адаптация)
    with torch.no_grad():
        # Берем центральную часть 7×7 ядра как 3×3
        model.conv1.weight.data = original_conv1.weight.data[:, :, 2:5, 2:5]

    # Убираем первый maxpool (слишком агрессивно для 32×32)
    model.maxpool = nn.Identity()

    # Изменяем последний слой
    model.fc = nn.Linear(512, 10)

    # Инициализация последнего слоя
    nn.init.xavier_uniform_(model.fc.weight)
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias)

    print(f"conv1 размер: {model.conv1.weight.shape}")  # Должно быть [64, 3, 3, 3]


    # 5. РАЗМОРОЗКА СЛОЁВ
    def unfreeze_layers(model, num_layers=5):
        """Размораживает последние num_layers"""
        all_params = list(model.named_parameters())

        print(f"\n=== РАЗМОРОЗКА ({num_layers} слоёв) ===")
        frozen = 0
        unfrozen = 0

        for i, (name, param) in enumerate(reversed(all_params)):
            if i < num_layers:
                param.requires_grad = True
                unfrozen += 1
                print(f"✓ {name}")
            else:
                param.requires_grad = False
                frozen += 1

        print(f"Разморожено: {unfrozen}, Заморожено: {frozen}")


    unfreeze_layers(model, num_layers=5)

    # 6. ОПТИМИЗАТОР И SCHEDULER
    criterion = nn.CrossEntropyLoss()

    # Оптимизатор ТОЛЬКО для размороженных параметров
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params,
                          lr=0.01,  # Начни с 0.01, не 0.1!
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 7. ПРОВЕРКА ПЕРЕД ОБУЧЕНИЕМ
    print("\n=== ПРОВЕРКА ===")
    test_batch, _ = next(iter(trainloader))
    print(f"Batch shape: {test_batch.shape}")  # Должно быть [128, 3, 32, 32]

    with torch.no_grad():
        model.eval()
        output = model(test_batch[:4])
        print(f"Output shape: {output.shape}")
        print(f"Sample outputs:\n{output[:2]}")

        # Проверяем, что выходы разные
        if torch.allclose(output[0], output[1], rtol=0.1):
            print("⚠Внимание: выходы слишком похожи!")
        else:
            print("Выходы разные — модель работает!")


    # 8. ФУНКЦИЯ ВАЛИДАЦИИ
    def evaluate(model, testloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        return accuracy


    # 9. ОБУЧЕНИЕ
    print("\n" + "=" * 50)
    print("НАЧИНАЕМ ОБУЧЕНИЕ")
    print("=" * 50)

    best_acc = 0
    for epoch in range(50):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        with tqdm(trainloader, desc=f"Epoch {epoch + 1:2d}", unit="batch") as pbar:
            for images, labels in pbar:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Статистика
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Обновляем прогресс-бар
                current_acc = 100. * correct / total
                avg_loss = total_loss / (pbar.n + 1)

                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'train_acc': f'{current_acc:.1f}%'
                })

        # Валидация
        val_acc = evaluate(model, testloader)

        # Обновляем scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Эпоха {epoch + 1:2d}: "
              f"Train Loss: {total_loss / len(trainloader):.4f}, "
              f"Train Acc: {100. * correct / total:.1f}%, "
              f"Val Acc: {val_acc:.1f}%, "
              f"LR: {current_lr:.5f}")

        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_cifar10_resnet.pth')
            print(f"Сохранена лучшая модель (Val Acc: {val_acc:.1f}%)")

        # Ранняя остановка если accuracy > 90%
        if val_acc > 90:
            print(f"Достигнута цель >90%! Останавливаем обучение.")
            break

    print("\n" + "=" * 50)
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучшая Val Accuracy: {best_acc:.1f}%")
    print("=" * 50)

    # 10. ТЕСТ НА ВСЕХ ТЕСТОВЫХ ДАННЫХ
    print("\n=== ФИНАЛЬНЫЙ ТЕСТ ===")
    final_acc = evaluate(model, testloader)
    print(f"Финальная точность на тесте: {final_acc:.1f}%")

    if final_acc < 80:
        print("\n РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
        print("1. Увеличь num_layers в unfreeze_layers() до 10")
        print("2. Увеличь batch_size до 256 (если памяти хватает)")
        print("3. Добавь CutMix/MixUp аугментации")
        print("4. Используй Label Smoothing (smoothing=0.1)")
        print("5. Обучи больше эпох (100 вместо 50)")