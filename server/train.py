import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from config import DATASET_DIR, MODEL_DIR, NUM_EPOCHS, PATIENCE

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

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    stopped_early = False

    for epoch in range(NUM_EPOCHS):
        # ====== 训练 ======
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
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # ====== 验证 ======
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ====== 保存 & 早停 ======
        if val_loss < best_val_loss:  # ✅ 用 Val Loss 判断
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            pth_path = os.path.join(MODEL_DIR, "mobilenet_classifier.pth")
            torch.save(model.state_dict(), pth_path)
            print(f"💾 新最佳模型已保存 (Val Loss={best_val_loss:.4f} @ Epoch {best_epoch})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ 验证集 Loss 连续 {PATIENCE} 次未下降，提前停止训练。")
                stopped_early = True
                break

    # ====== 训练结束提示 ======
    if stopped_early:
        print(f"🏆 最佳模型出现在 Epoch {best_epoch} (Val Loss={best_val_loss:.4f})")
    else:
        print(f"⚠️ 已达到最大轮数 {NUM_EPOCHS}，最佳模型在 Epoch {best_epoch} (Val Loss={best_val_loss:.4f})")
        print("👉 可以考虑增大 NUM_EPOCHS 或调整 PATIENCE 以获得更高精度。")

    return model, len(train_dataset.classes)
