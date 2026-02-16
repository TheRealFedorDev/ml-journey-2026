from Learning_With_Obsidian_And_DeepSeek.config import TRAIN_DIR, VAL_DIR, TEST_DIR
import os
import torch.utils.data
from PIL import Image


print("Проверяем путь:", TRAIN_DIR)
print("Существует?", os.path.exists(TRAIN_DIR))
if not os.path.exists(TRAIN_DIR):
    print("Папка НЕ найдена!")
def get_file_paths_and_labels(root_dir):
    file_paths = []
    labels = []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            images_dir = os.path.join(class_path, 'images')
            if os.path.isdir(images_dir):
                for file_name in os.listdir(images_dir):
                    file_path = os.path.join(images_dir, file_name)
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_paths.append(file_path)
                        labels.append(class_name)
    return file_paths, labels
class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform):
        self.paths = file_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open(self.paths[item]).convert('RGB')
        image = self.transform(image)
        return (image, self.labels[item])
    def __len__(self):
        return len(self.paths)

train_paths, train_labels = get_file_paths_and_labels(TRAIN_DIR)