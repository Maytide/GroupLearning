import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

from utils import *
from network import NeuralNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, num_epochs, learning_rate=0.001, name='model.ckpt'):
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('-- Beginning training --')
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(train_loader):
            # Move tensors to the configured device
            data = data.to(device)
            targets = targets.to(device)
            # print(type(data), type(targets), max(data), max(targets))
            # print(targets); print(targets)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), name)


def test(model, test_loader, resultfile='results_goodLabels.txt'):
    acc = 0
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            # targets = oneHot_to_labels(targets)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            _, targets_label = torch.max(targets, 1)
            total += targets.size(0)
            # print(predicted.shape, targets.shape)
            correct += (predicted == targets_label).sum().item()

        acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: {} %'.format(acc))

    with open(resultfile, 'w') as f:
        f.write(str(acc))


if __name__ == '__main__':
    print(f'Using device: {device}')
    if device.__str__() == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    batch_size = 128
    data_train, targets_train, data_test, targets_test = load_mnist(60000)
    # corrupt_targets_train = corrupt_labels(targets_train, num_corrupt=2)
    train_dataset = MNISTDataset(data_train,
                                 targets_train,
                                 transform=transforms.Compose([
                                                   Reshape(),
                                                   Scale(),
                                               ]))
    test_dataset = MNISTDataset(data_test,
                                targets_test,
                                transform=transforms.Compose([
                                                  Reshape(),
                                                  Scale(),
                                              ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    num_epochs = 1
    model = NeuralNet(784, 392, 10).to(device)

    train(model, train_loader, num_epochs, name='models/model_goodLabels.ckpt')
    test(model, test_loader, resultfile='results_goodLabels.txt')

