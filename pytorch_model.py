import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
import torch.nn.functional as F

# --- 1. Model Definition ---
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # Initial image size: 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Size: 14x14
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # After second pool: 7x7
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) 
        x = self.fc1(x)
        return x

# --- 2. Checkpointing Functions ---
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, device):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Ensure optimizer state (like momentum) is moved to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

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

    print(f'Accuracy: {float(num_correct) / float(num_samples)*100:.2f}%')
    model.train()
    
# --- 3. Setup and Hyperparameters ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 1e-4
batch_size = 1024
num_epochs = 10
load_model = True # <--- NEW PARAMETER

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 4. Load Checkpoint Logic ---
if load_model:
    # Use map_location=device to correctly handle loading from GPU to CPU, or vice versa
    # Wrap in a try-except block in case the file doesn't exist
    try:
        load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=device), model, optimizer, device)
    except FileNotFoundError:
        print("Checkpoint file not found. Starting training from scratch.")


# --- 5. Train Network ---
print("Starting Training...")
for epoch in range(num_epochs):
    
    # Save a checkpoint every 2 epochs (as seen in your screenshot)
    if (epoch + 1) % 2 == 0:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")


# --- 6. Final Evaluation ---
print("\nFinal Evaluation:")
print("Training Data:")
check_accuracy(train_loader, model, device)
print("Test Data:")
check_accuracy(test_loader, model, device)