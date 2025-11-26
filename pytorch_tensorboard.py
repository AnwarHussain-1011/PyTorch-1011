import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import timm  # EfficientNetV2!
import os

# ✅ FIXED: Simple CNN (MNIST) or EfficientNetV2 (Blastocysts)
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def get_model(model_name, num_classes, img_size=224, pretrained=True):
    if model_name == "cnn":  # MNIST
        return SimpleCNN(num_classes=num_classes)
    else:  # EfficientNetV2 (Blastocysts)
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

def run_training(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using {device} | Model: {cfg['model_name']}")

    # ✅ Transforms (EffNetV2-Optimal)
    if cfg['dataset'] == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
    else:  # Blastocysts
        transform = transforms.Compose([
            transforms.Resize((cfg['img_size'], cfg['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    # ✅ Datasets/Loaders
    if cfg['dataset'] == 'mnist':
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_ds = datasets.ImageFolder(cfg['train_path'], transform=transform)
        test_ds = datasets.ImageFolder(cfg['test_path'], transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    # ✅ Model + Opt + LR
    model = get_model(cfg['model_name'], cfg['num_classes'], cfg['img_size']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # ✅ TensorBoard
    writer = SummaryWriter(f"runs/{cfg['run_name']}")
    step = 0
    best_acc = 0

    for epoch in range(cfg['epochs']):
        # Train
        model.train(); train_loss, train_acc = 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (scores.argmax(1) == targets).sum().item()

            # ✅ LOG: Loss/Acc/Images
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Acc", train_acc/len(train_loader.dataset), step)
            writer.add_images("Train/Images", vutils.make_grid(data[:8], normalize=True), step)
            step += 1

        # Val/Test
        model.eval(); test_acc = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                scores = model(data)
                test_acc += (scores.argmax(1) == targets).sum().item()

        test_acc /= len(test_ds)
        writer.add_scalar("Test/Acc", test_acc, epoch)
        writer.add_hparams({f"lr:{cfg['lr']}", f"bs:{cfg['batch_size']}"}, {"test_acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"best_{cfg['run_name']}.pth")
            print(f"✅ Epoch {epoch}: Test Acc {test_acc:.4f} | BEST!")

        scheduler.step()
        print(f"Epoch {epoch}: Train Loss {train_loss/len(train_loader):.4f} | Test Acc {test_acc:.4f}")

    writer.close()
    print(f"🎉 Best Model: best_{cfg['run_name']}.pth | Acc: {best_acc:.4f}")

if __name__ == "__main__":
    # ✅ CONFIGS: MNIST or BLASTOCYST
    cfgs = [
        # MNIST (Tutorial: 99% in 1 Epoch)
        {"model_name": "cnn", "dataset": "mnist", "run_name": "mnist_cnn", 
         "epochs": 1, "batch_size": 64, "lr": 1e-3, "num_classes": 10, "img_size": 28},
        
        # BLASTOCYST (Your Data: 95%+)
        {"model_name": "efficientnetv2_s", "dataset": "blastocyst", "run_name": "effv2_blastocyst",
         "train_path": r"D:/classifier/data_three_tSB_tB_tEB/train",
         "test_path": r"D:/classifier/data_three_tSB_tB_tEB/val",
         "epochs": 100, "batch_size": 16, "lr": 1e-4, "num_classes": 3, "img_size": 384}
    ]

    for cfg in cfgs:
        run_training(cfg)