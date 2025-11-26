import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data
from torch.utils.data import DataLoader # Corrected import statement
import torchvision.transforms as transforms 

# VGG16-like architecture configuration (VGG-D)
# Note: The original VGG used 3x3 convolutions with same padding (padding=1)
VGG_A_LIST = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# Your list was missing a 'M' at the start and had a typo (156 should be 256)
VGG_D_LIST = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# Using the list you provided (renamed for clarity)
VGG_ARCHITECTURE = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_net(nn.Module):
    # Corrected Indentation Level 1 for the method definition
    def __init__(self, in_channels=3, num_classes=1000): 
        # Corrected Indentation Level 2 for the method body
        super(VGG_net, self).__init__() 
        self.in_channels = in_channels
        
        # 1. Corrected variable name from VGG16 to VGG_ARCHITECTURE
        self.conv_layers = self.create_conv_layers(VGG_ARCHITECTURE)

        # 2. Corrected input size for nn.Linear for 224x224 input
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            # 3. Corrected typo: nn.Droupout -> nn.Dropout
            nn.Dropout(p=0.5), 
            nn.Linear(4096, 4096), # VGG typically has two 4096 layers
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        # 4. Corrected function call: x.shape(...) -> x.reshape(...)
        # and we must flatten the tensor from (N, C, H, W) to (N, C*H*W)
        x = x.reshape(x.shape[0], -1) 
        
        x = self.fcs(x)
        return x
    
    # Corrected Indentation Level 1 for the method definition
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(out_channels), # Added nn.BatchNorm2d
                    nn.ReLU()
                ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        
        # 5. Corrected function call: nn.sequential -> nn.Sequential
        # 6. Moved this return statement OUT of the loop
        return nn.Sequential(*layers) 

# 7. Moved test code OUT of the class definition 
# and fixed the model call (mode -> model)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = VGG_net(in_channels=3, num_classes=1000).to(device)
x = torch.rand(1, 3, 224, 224).to(device) # VGG typically uses 224x224
print(model(x).shape)