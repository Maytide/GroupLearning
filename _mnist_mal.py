import os
import csv

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

from utils import *
from network import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, num_epochs, learning_rate=0.001, name='models/model.ckpt'):
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('Beginning training:')
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(train_loader):
            # Move tensors to the configured device
            data = data.to(device)
            targets = targets.to(device)
            # print(type(data), type(targets), max(data), max(targets))

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


def test(model, test_loader):
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

    return acc


def experiment(reshaper, num_malignant, model, num_epochs):
    print(f'-- Beginning experiment with {num_malignant} malignant labels --')
    batch_size = 128
    data_train, targets_train, data_test, targets_test = load_mnist(60000)
    corrupt_targets_train = corrupt_labels(targets_train, num_corrupt=num_malignant)
    train_dataset = MNISTDataset(data_train,
                                 corrupt_targets_train,
                                 transform=transforms.Compose([
                                     reshaper,
                                     Scale(),
                                 ]))
    test_dataset = MNISTDataset(data_test,
                                targets_test,
                                transform=transforms.Compose([
                                    reshaper,
                                    Scale(),
                                ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    train(model, train_loader, num_epochs,
          name=f'models/model_malignantLabels_{num_malignant}.ckpt')
    test_acc = test(model, test_loader)
    print(f'Accuracy of the network on the 10000 test images: {test_acc} %\n')

    return test_acc


def experiment_1d(num_malignant, model, num_epochs):
    return experiment(Reshape(), num_malignant, model, num_epochs)


def experiment_2d(num_malignant, model, num_epochs):
    return experiment(lambda x: x, num_malignant, model, num_epochs)


if __name__ == '__main__':
    print(f'Using device: {device}')
    if device.__str__() == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = NeuralNet(784, 392, 10).to(device)
    # model = ConvNet().to(device)
    savedir = f'graphs/{str(model)}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    csvpath = os.path.join(savedir, 'results_malignantLabels.csv')

    test_accs = []
    for i in range(0, 10):  # 0-9 malignancies
        model = NeuralNet(784, 392, 10).to(device)
        # model = ConvNet().to(device)
        test_acc = experiment_1d(i, model, 1)
        test_accs.append(test_acc)

    with open(csvpath, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        wr.writerow(test_accs)


