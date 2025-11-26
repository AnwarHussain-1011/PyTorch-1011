import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms

# Removed unused imports: pandas, spacy, pad_sequence, Image, nn (nn is not used directly)

# Methods for dealing with imbalanced datasets:
# 1. Oversampling (implemented here)
# 2. Class weighting


def get_loader(root_dir, batch_size):
    """
    Creates a DataLoader with WeightedRandomSampler for oversampling underrepresented classes.
    
    The weight for each class is calculated as 1 / (number of samples in that class).
    """
    my_transforms = transforms.Compose( # Corrected: Capital 'C'
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Loads data assuming a structure like: root_dir/class_name_A/img.jpg
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    
    # Get the list of class names (subdirectories)
    subdirectories = dataset.classes 
    class_weights = []

    # Calculate the inverse frequency (weight) for each class
    for subdir in subdirectories:
        # Get the number of files in the class subdirectory
        files = os.listdir(os.path.join(root_dir, subdir))
        # Weight is 1 / count
        class_weights.append(1 / len(files))

    # Assign the calculated class weight to every sample in the dataset
    sample_weights = [0] * len(dataset)

    # Loop through the dataset's targets (labels)
    # NOTE: (data, label) is used for backward compatibility; ImageFolder uses (img, target) 
    for idx, (_, label) in enumerate(dataset.samples):
        # label here is the class index (0, 1, 2, ...)
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    # Create the sampler: selects samples with replacement based on weights
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights), # Total number of samples to draw in one epoch
        replacement=True # Must be True for oversampling
    )

    # Create DataLoader using the sampler
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return loader


def main():
    # Set the path to the dataset folder.
    # The structure should be: D:/.../dataset/Golden retriever/ and D:/.../dataset/Swedish elkhound/
    # NOTE: Using a raw string (r"...") is essential for Windows paths with backslashes.
    path_to_data = r"D:\PyTorch_1011\Machine-Learning-Collection-master\ML\Pytorch\Basics\Imbalanced_classes\dataset"
    
    # Check for the exact path match (optional check)
    if not os.path.isdir(path_to_data):
        print(f"ERROR: Dataset path not found: {path_to_data}")
        print("Please check the path or ensure the folder exists.")
        return

    loader = get_loader(root_dir=path_to_data, batch_size=8)
    
    # Assuming two classes: 0 (Golden retriever) and 1 (Swedish elkhound)
    num_retrievers = 0
    num_elkhounds = 0
    num_epochs = 10
    
    print(f"Starting weighted sampling test for {num_epochs} epochs...")
    
    # Run through the DataLoader for 10 epochs to see the effect of sampling
    for epoch in range(num_epochs):
        for data, labels in loader:
            # Count the number of samples for each class in the retrieved data
            num_retrievers += torch.sum(labels == 0).item()
            num_elkhounds += torch.sum(labels == 1).item()
    
    print("\n--- Weighted Sampling Results (10 Epochs) ---")
    print(f"Total Retrieved Samples (Class 0: Retrievers): {num_retrievers}")
    print(f"Total Retrieved Samples (Class 1: Elkhounds): {num_elkhounds}")


if __name__ == "__main__":
    main()