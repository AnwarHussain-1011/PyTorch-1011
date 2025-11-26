import torch
import torch.nn as nn 
import torch.optim as optim
from PIL import Image 
import torchvision.transforms as transforms 
import torch.models as models 
from torchvision.utlis import save_image 

model = models.vgg19(pretrained=True).features

print(model)