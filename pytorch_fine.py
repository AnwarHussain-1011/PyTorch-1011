import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import sys 

# --- Set device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
in_channel = 3          # CIFAR-10 is RGB
num_classes = 10        # CIFAR-10 has 10 classes
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

# --- Custom Layers for Model Modification ---

# 1. Identity Layer: Used to bypass VGG's fixed 7x7 AvgPool requirement for smaller images.
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# --- Load Pre-trained Model & Modify ---

# Load VGG16 model with pre-trained ImageNet weights
# Using VGG16_Weights.IMAGENET1K_V1 is the modern way to load weights
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Freeze all VGG base parameters (Transfer Learning: Feature Extractor)
for param in model.parameters():
    param.requires_grad = False 

# Modify the VGG layers for CIFAR-10:

# 1. Replace the AvgPool layer
# This is necessary because CIFAR-10's 32x32 image size won't produce the 7x7 feature map 
# VGG's original AvgPool expects. We bypass it with Identity().
model.avgpool = Identity()

# 2. Replace the Classifier Head
# The input to the new classifier is 512 (number of feature maps) * 1 * 1 (size of feature map).
# We use a simple 2-layer classifier as shown in your images.
model.classifier = nn.Sequential(
    nn.Linear(512 * 1 * 1, 512), # Input size = 512 * 1 * 1 = 512
    nn.ReLU(inplace=True),
    nn.Linear(512, num_classes)
)

model.to(device)
print("VGG16 Model loaded and modified successfully.")

# --- Load Data ---
# Standard normalization for ImageNet pre-trained models is used.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- Loss and Optimizer ---
# NOTE: Only the UN-FROZEN parameters (the new classifier) will be passed to the optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Check Accuracy Function ---
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}%')
    model.train()
    
# --- Training Loop ---
print("Starting Training...")
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()
        
    print(f"Cost at epoch {epoch} is {loss.item():.4f}")

# --- Final Evaluation ---
print("\nChecking accuracy on training data")
check_accuracy(train_loader, model, device)
print("Checking accuracy on test data")
check_accuracy(test_loader, model, device)