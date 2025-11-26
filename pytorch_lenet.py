import torch
import torch.nn as nn

# LeNet architecture 1x32x32

class LeNet(nn.Module): # 1. Corrected parent class from nn.Model
    
    # 2. Corrected __init__ definition and indentation
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        # 3. Corrected AvgPool2d: kernel_size and stride must be integers
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) 
        
        # All nn.Conv2d instances must be correctly capitalized
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        
        # Input size (120) is correct for 32x32 input after the conv blocks
        self.linear1 = nn.Linear(120, 84) 
        self.linear2 = nn.Linear(84, 10) # Output is 10 classes
        

    def forward(self, x):
        # Layer 1: Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv1(x))) # Output shape: [N, 6, 14, 14]
        
        # Layer 2: Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x))) # Output shape: [N, 16, 5, 5]
        
        # Layer 3: Conv -> ReLU
        x = self.relu(self.conv3(x)) # Output shape: [N, 120, 1, 1]
        
        # Flatten (reshape) for Linear layers
        x = x.reshape(x.shape[0], -1) # Output shape: [N, 120]
        
        # Layer 4: Linear -> ReLU
        x = self.relu(self.linear1(x)) # Output shape: [N, 84]
        
        # Layer 5: Linear (Output layer)
        x = self.linear2(x) 
        
        return x

# Test the model and print the output shape
x = torch.rand(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)