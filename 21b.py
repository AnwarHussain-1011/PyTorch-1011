import torch
import toch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms 
from torch.utils.tensorboard, load_checkpoint, print_examples
from utils import save_checkpoint, load_checkpoint, print_examples
from loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.compose(
        [
            transforms.Resize((356,, 356)),
            transforms.Randomcrop(299, 299),
            transformes.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="fliker8k/images",
        annotation_file="flickr8k/captions.txt",
        num_workers = 2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Load_model = False
    save_model = True

    #hyperparameters

    embed_sizee =356
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    #for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    #initialize model, loss etc

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.crossEntropyLoss(ignore_index=.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model optimizer)

    model.train()

    for epoch in range(num_epochs):
        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" :state_dict(),
                "step":step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in enmerate(train_loader):
            img = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), caption.reshape(-1))

            writer.add_scalar("Training loss", loss.otem(), global_step=step)
            step +=1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

if __name__ == "__main__":
    train()
