import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from config import DATASET_DIR, MODEL_DIR, NUM_EPOCHS

def train_model():
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir   = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"❌ 未找到 train/ 或 val/ 文件夹，请检查路径: {DATASET_DIR}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 数据增强
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"📊 训练样本数: {len(train_dataset)} | 验证样本数: {len(val_dataset)}")
    print(f"📊 类别数: {len(train_dataset.classes)}")

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("💻 使用设备:", device)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # 保存 pth
    pth_path = os.path.join(MODEL_DIR, "mobilenet_classifier.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"✅ 模型已保存到 {pth_path}")

    return model, len(train_dataset.classes)
