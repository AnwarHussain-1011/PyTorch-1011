import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils # For random_split
from torch.utils.data import DataLoader, Dataset # Import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image # For opening images
import pandas as pd  # For reading the CSV file
import os # For path joining
from customDataset import CatsAndDogsDataset

# --- Custom Dataset Class Definition ---
# This class handles loading your custom image data (referenced by CSV)
class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations (image names and labels).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # NOTE: Assumes your CSV has columns like 'image_name' and 'label'
        img_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_name).convert("RGB") # Ensure 3 channels for GoogLeNet

        # Assuming label is in the second column (adjust if necessary)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

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
            # GoogLeNet outputs may be tuples if using auxiliary losses, handle scores correctly
            if isinstance(scores, tuple):
                 scores = scores[0] 
                 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
    # NOTE: Output formatting matches your image
    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}")

    model.train()


# --- Main Script Execution ---

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 10 
learning_rate = 1e-3
batch_size = 32
num_epochs = 1 

# Load Data
# NOTE: GoogLeNet requires input size of at least 224x224. 
# Added transforms.Resize(224) to ensure compatibility.
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet Normals
])

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', 
                             root_dir='cats_dogs_resized', 
                             transform=transform)

# Assume your dataset has 25000 images total based on your split
# NOTE: Adjusted split size to use the data_utils module as imported
train_set, test_set = data_utils.random_split(dataset, [20000, 5000])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model (GoogLeNet Transfer Learning)
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

# Freeze the feature extraction layers
for param in model.parameters():
    param.requires_grad = False

# Modify the classification head
# The original GoogLeNet classifier is model.fc. Replace it with a new Linear layer
# NOTE: GoogLeNet's final layer has input features = 1024
model.fc = nn.Linear(1024, num_classes) 

# Move model to device
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# NOTE: Optimizer only updates the un-frozen parameters (model.fc)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
print("Starting Training...")
for epoch in range(num_epochs):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        
        # GoogLeNet's forward pass returns a tuple if auxiliary loss is used (during train)
        # We must grab the main output (scores[0])
        if isinstance(scores, tuple):
             scores = scores[0] 
             
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.4f}")


# Check accuracy on training and test sets
print("\nChecking accuracy on Training Set")
check_accuracy(train_loader, model, device)

print("\nChecking accuracy on Test Set")
check_accuracy(test_loader, model, device)