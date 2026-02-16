from .model import create_model

import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(total=len(trainloader), desc="Training", unit="batch", leave=False)

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Статистика
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / (batch_idx + 1)
        current_acc = 100. * correct / total

        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'acc': f'{current_acc:.1f}%'
        })

    pbar.close()

    avg_loss = total_loss / len(trainloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, valloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_total = labels.size(0)
            batch_correct = predicted.eq(labels).sum().item()

    if total == 0:
        print("Внимание: валидационный загрузчик пуст!")
        return 0.0, 0.0
    avg_loss = total_loss / len(valloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy