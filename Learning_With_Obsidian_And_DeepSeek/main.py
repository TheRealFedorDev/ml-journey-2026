import os

import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from src.dataset import CovidDataset
from src.model import create_model
from src.train import train_one_epoch, validate
from src.dataset import get_file_paths_and_labels
from Learning_With_Obsidian_And_DeepSeek.config import TRAIN_DIR, VAL_DIR, TEST_DIR, BASE_DIR


def main():
    ROOT_DIR = BASE_DIR
    print("Текущая рабочая директория:", os.getcwd())
    print("Существует ли путь?", os.path.exists(ROOT_DIR))
    print("Содержимое папки train:")

    for item in os.listdir(TRAIN_DIR):
        item_path = os.path.join(TRAIN_DIR, item)
        if os.path.isdir(item_path):
            print(f"  Папка: {item}")
            # посмотрим первые 3 файла внутри каждой папки
            files = os.listdir(item_path)[:3]
            for f in files:
                print(f"    - {f}")
        else:
            print(f"  Файл: {item}")

    def split_data(file_paths, labels):
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            file_paths, labels, test_size=0.15, random_state=42, stratify=labels
        )

        val_size = 0.15 / 0.85
        if len(train_paths) == 0:
            raise ValueError("train_paths пуст после первого split!")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels
        )
        if len(train_paths) == 0:
            raise ValueError("После первого split train_paths пуст!")
        print(f"После второго split: train={len(train_paths)}, val={len(val_paths)}")
        print(f"Всего файлов: {len(file_paths)}")
        print(f"Train: {len(train_paths)} файлов")
        print(f"Val:   {len(val_paths)} файлов")
        print(f"Test:  {len(test_paths)} файлов")
        print(f"Train labels: {set(train_labels)}")
        print(f"Val labels: {set(val_labels)}")
        print(f"Test labels: {set(test_labels)}")
        return (train_paths, val_paths, test_paths,
                    train_labels, val_labels, test_labels)


    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=5),
        transforms.CenterCrop(224),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    (file_paths, labels) = get_file_paths_and_labels(TRAIN_DIR)
    le = LabelEncoder()

    labels = le.fit_transform(labels)
    print("Уникальные метки после кодирования:", set(labels))
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_data(file_paths, labels)

    print(f"Всего файлов: {len(file_paths)}")
    print(f"Train: {len(train_paths)} файлов")
    print(f"Val:   {len(val_paths)} файлов")
    print(f"Test:  {len(test_paths)} файлов")

    train_dataset = CovidDataset(train_paths, train_labels, train_transform)
    val_dataset = CovidDataset(val_paths, val_labels, val_transform)
    test_dataset = CovidDataset(test_paths, test_labels, val_transform)  # для теста используем val_transform (без аугментаций)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


    model = create_model(num_classes=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(device)

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    print(outputs.shape)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.
    num_epochs = 50

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", unit="epoch")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_pbar.set_postfix({
            'train_acc': f'{train_acc:.1f}%',
            'val_acc': f'{val_acc:.1f}%'
        })

        print(
            f"Epoch {epoch + 1:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  → Saved best model with val_acc {val_acc:.2f}%")

if __name__ == "__main__":
    main()