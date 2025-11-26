#hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1000

#load Data 
train_dataset = datasets.MIIST(
    root"dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), sownload=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch-size, shuffle=True)
