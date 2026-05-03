# 1. 环境与依赖
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 数据集路径与统计
data_root = "./chest_xray"
train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")

# 统计原始train集的类别分布
def count_samples(directory):
    normal = len(os.listdir(os.path.join(directory, "NORMAL")))
    pneumonia = len(os.listdir(os.path.join(directory, "PNEUMONIA")))
    return normal, pneumonia

train_normal, train_pneu = count_samples(train_dir)
test_normal, test_pneu = count_samples(test_dir)

print(f"Train: NORMAL={train_normal}, PNEUMONIA={train_pneu}")
print(f"Test: NORMAL={test_normal}, PNEUMONIA={test_pneu}")

# 3. 自定义数据集类
class XRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 加载train集所有路径
def load_data_paths(directory):
    paths = []
    labels = []
    # 正常类：标签0
    normal_dir = os.path.join(directory, "NORMAL")
    for img_name in os.listdir(normal_dir):
        paths.append(os.path.join(normal_dir, img_name))
        labels.append(0)
    # 肺炎类：标签1
    pneu_dir = os.path.join(directory, "PNEUMONIA")
    for img_name in os.listdir(pneu_dir):
        paths.append(os.path.join(pneu_dir, img_name))
        labels.append(1)
    return paths, labels

# 加载并划分train/val
train_paths, train_labels = load_data_paths(train_dir)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# 加载test集
test_paths, test_labels = load_data_paths(test_dir)

# 4. 数据预处理与增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集与DataLoader
train_dataset = XRayDataset(train_paths, train_labels, transform=train_transform)
val_dataset = XRayDataset(val_paths, val_labels, transform=val_test_transform)
test_dataset = XRayDataset(test_paths, test_labels, transform=val_test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. 模型构建（以ResNet50迁移学习为例）
def build_model():
    model = models.resnet50(pretrained=True)
    # 冻结底层参数
    for param in model.parameters():
        param.requires_grad = False
    # 替换顶层全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类：正常/肺炎
    return model.to(device)

model = build_model()

# 6. 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 7. 训练循环
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc.cpu().numpy())
        val_acc_history.append(val_acc.cpu().numpy())

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

model, train_loss, val_loss, train_acc, val_acc = train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=10
)

# 8. 绘制训练曲线
def plot_curves(train_loss, val_loss, train_acc, val_acc, save_path="figures/training_curves.png"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

plot_curves(train_loss, val_loss, train_acc, val_acc)

# 9. 测试集评估
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, cm

acc, precision, recall, f1, cm = evaluate_model(model, test_loader)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# 绘制混淆矩阵
def plot_confusion_matrix(cm, save_path="figures/confusion_matrix.png"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

plot_confusion_matrix(cm)