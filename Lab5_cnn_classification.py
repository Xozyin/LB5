import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Решение конфликта с OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Проверка изображений на корректность
def validate_images(image_dir):
    valid_paths = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                img = Image.open(path)
                img.verify()
                img.close()
                valid_paths.append(path)
            except Exception as e:
                print(f"Пропущено: {path} ({e})")
    return valid_paths

# Путь к данным
image_root = './Animals'

print("Проверка изображений...")
image_paths = validate_images(image_root)

# Преобразования изображений
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Класс датасета
class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self.class_names = sorted(set(os.path.basename(os.path.dirname(p)) for p in paths))
        self.class_map = {cls: idx for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            img = Image.open(path).convert("RGB")
            label_name = os.path.basename(os.path.dirname(path))
            label = self.class_map[label_name]
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return self.__getitem__((index + 1) % len(self.paths))

# Загрузка датасета
dataset = AnimalDataset(image_paths, transform=image_transforms)

# Делим на обучающую и тестовую части
train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
train_set, test_set = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print("Обнаружены классы:", dataset.class_names)

# Модель ResNet18
from torchvision.models import ResNet18_Weights
net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
net.fc = nn.Linear(net.fc.in_features, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Оптимизация
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_history = []
epochs = 10

# Обучение
for ep in range(epochs):
    net.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = net(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(train_loader)
    loss_history.append(avg)
    print(f"Эпоха {ep+1}/{epochs}, Потери: {avg:.4f}")

# График потерь
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.title("График функции потерь")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.grid(True)
plt.show()

# Оценка точности
net.eval()
total = 0
correct = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        y_pred = net(x_test)
        _, preds = torch.max(y_pred, 1)
        total += y_test.size(0)
        correct += (preds == y_test).sum().item()

accuracy = 100 * correct / total
print(f"Точность на тестовой выборке: {accuracy:.2f}%")

# Матрица ошибок
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

conf_matrix = np.zeros((len(dataset.class_names), len(dataset.class_names)), dtype=int)
for t, p in zip(all_targets, all_preds):
    conf_matrix[t][p] += 1

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d",
            xticklabels=dataset.class_names,
            yticklabels=dataset.class_names,
            cmap="Blues")
plt.title("Матрица ошибок классификации")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.show()