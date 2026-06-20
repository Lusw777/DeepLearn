import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 配置 ====================
DATA_PATH = "./raw-img"  # ⬅️ 改成你的实际路径
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 检查数据集是否存在
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ 找不到数据集 {DATA_PATH}，请检查路径！")

# ==================== 数据预处理 ====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 加载数据集 ====================
full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=train_transform)
class_names = full_dataset.classes  # 自动读取：['cane', 'cavallo', ...]
NUM_CLASSES = len(class_names)
print(f"📂 发现 {NUM_CLASSES} 个类别: {class_names}")

# 划分数据集
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# 验证集和测试集用不同的transform（无数据增强）
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

# ==================== 模型 ====================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==================== 训练函数 ====================
def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="训练"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        outputs = model(images)
        loss = crit(outputs, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def validate(model, loader, crit):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="验证"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = crit(outputs, labels)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

# ==================== 训练主循环 ====================
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0
os.makedirs("models", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    print(f"\n📊 Epoch {epoch}/{EPOCHS}")
    tl, ta = train_epoch(model, train_loader, optimizer, criterion)
    vl, va = validate(model, val_loader, criterion)
    scheduler.step()
    
    train_losses.append(tl); val_losses.append(vl)
    train_accs.append(ta); val_accs.append(va)
    print(f"训练 Loss: {tl:.4f} Acc: {ta:.2f}% | 验证 Loss: {vl:.4f} Acc: {va:.2f}%")
    
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), "models/resnet18_best.pth")
        print(f"✅ 保存最佳模型 (准确率: {va:.2f}%)")

print(f"\n🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")

# ==================== 绘制曲线 ====================
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.legend()
plt.title('损失曲线')
plt.subplot(1,2,2)
plt.plot(train_accs, label='训练准确率')
plt.plot(val_accs, label='验证准确率')
plt.legend()
plt.title('准确率曲线')
plt.savefig('training_curves.png', dpi=300)
plt.show()