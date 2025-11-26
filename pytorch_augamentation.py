import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import CatsAndDogsDataset

# Define transformations
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # This won't change values but is syntactically fine
])

# Load dataset
dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=my_transforms)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Save transformed images
img_num = 0
for i, (img, label) in enumerate(train_loader):
    if img_num >= 10:
        break
    save_image(img, f'img{img_num}.png')
    img_num += 1