import sys

import numpy as np
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch import tensor, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import cuda

import matplotlib.pyplot as plt


INPUT_POINTS = 64


def get_device():
    if cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


class CNN_Autoencoder(nn.Module):

    def __init__(self):
        super(CNN_Autoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=(1, 3))
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(1, 16, kernel_size=(3, 4), stride=2)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)

    def forward(self, x):
        #print("Input shape: ", x.shape)

        # Encoding
        x = F.relu(self.conv1(x))
        #print("Shape after Conv 1: ", x.shape)
        x = self.pool(x)
        #print("Shape after Pool 1: ", x.shape)
        x = F.relu(self.conv2(x))
        #print("Shape after Conv 2: ", x.shape)
        x = self.pool(x)
        #print("Shape after Pool 2:", x.shape)

        # Decoding
        x = F.relu(self.deconv1(x))
        #print("Shape after Deconv 1:", x.shape)
        x = F.sigmoid(self.deconv2(x))
        #print("Shape after Deconv 2:", x.shape)
        #print()

        return x


# returns: loss
def train_epoch(model, loader, optimizer, criterion):
    total_loss = 0.0

    for i, batch in enumerate(loader):
        #print("Training on batch {}".format(i))
        batch = batch.to(get_device())
        #print(batch.shape)
        batch = batch.reshape((len(batch), 1, INPUT_POINTS, 6))  # from 2D to 4D
        batch = batch.permute(0, 1, 3, 2)
        #print(batch.shape)

        """for j in range(4):
            image = batch[j, 0, :, :]
            plt.imshow(image.cpu())
            plt.title("Curve {} - From Batch Loader".format(j))
            plt.show()"""

        optimizer.zero_grad()

        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


if __name__ == '__main__':
    data = np.loadtxt(sys.argv[1], delimiter=",")
    labels = np.loadtxt(sys.argv[2], delimiter=",")

    data_tensor = tensor(data).float()
    labels_tensor = tensor(labels).float()

    train_data, eval_data, train_labels, eval_labels = train_test_split(data_tensor, labels_tensor, test_size=0.2)

    print(train_data.shape, train_labels.shape)
    print(eval_data.shape, eval_labels.shape)

    model = CNN_Autoencoder().float()
    print(model)

    #criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # show a few images and run a forward pass on the model to make sure it works
    """for i in range(4):
        image = train_data[i].reshape(INPUT_POINTS, 6).T

        plt.imshow(image)
        plt.title("Curve {}".format(i))
        plt.show()

        image = image.unsqueeze(0)
        image = image.unsqueeze(1)

        model(image)

    assert False"""

    model.to(get_device())

    # create data loaders
    train_loader = DataLoader(train_data, batch_size=32)

    # sample images to compare reconstructions to
    for e in range(3):
        image = eval_data[e].reshape(INPUT_POINTS, 6).T
        plt.imshow(image)
        plt.title("Sample Curve {}".format(e))
        plt.show()

    for epoch in range(20):
        epoch_loss = train_epoch(model, train_loader, optimizer, criterion)
        print("Epoch {}: {}".format(epoch, epoch_loss))

        if epoch % 4 == 0:
            for e in range(3):
                image = eval_data[e].reshape(INPUT_POINTS, 6).T
                reconstructed = model(image.reshape((1, 1, 6, INPUT_POINTS)).to(get_device()))

                plt.imshow(reconstructed.detach().cpu()[0, 0, :, :])
                plt.title("Reconstructed Curve {} - Epoch {}".format(e, epoch))
                plt.show()
