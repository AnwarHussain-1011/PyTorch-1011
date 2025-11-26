# Imports 
import torch
import torch.nn as nn # All NN models, nn.linear, nn.conv2d, Batchnorm, Loss functions
import torch.optim as optim # for all Optimization algorithms, SGD, Adam etc
from torch.utils.data import DataLoader # gives easier dataset management and creates mini batches
import torchvision.datasets as datasets # has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset 

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (For MNIST)
input_size = 28         # The size of one input vector (image row)
sequence_length = 28    # The number of time steps (image columns/rows)
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1 

# Create the LSTM Model
# Replaced nn.RNN with nn.LSTM for better performance
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Use nn.LSTM with batch_first=True
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # FIX: nn.Linear input size must be 'hidden_size', NOT sequence_length * hidden_size,
        # because the forward pass only uses the output of the *last* time step.
        self.fc = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        # Initialize hidden state (h0) and cell state (c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward prop
        # out shape: (batch_size, sequence_length, hidden_size)
        # hn, cn are the states of the LAST layer at the LAST time step
        out, (hn, cn) = self.rnn(x, (h0, c0))
        
        # FIX: Select only the output from the LAST time step: out[:,-1,:]
        # This converts (batch_size, sequence_length, hidden_size) to (batch_size, hidden_size)
        out = self.fc(out[:, -1, :]) 
        
        return out
        
# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to device
        # Squeeze(1) converts (Batch, 1, 28, 28) to (Batch, 28, 28) -> (Batch, Seq Length, Input Size)
        data = data.to(device=device).squeeze(1) 
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

# Check accuracy function (essential for seeing output)
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): 
        for x, y in loader:
            x = x.to(device=device).squeeze(1) 
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}%')

    model.train()

# Execution of accuracy check after training
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)