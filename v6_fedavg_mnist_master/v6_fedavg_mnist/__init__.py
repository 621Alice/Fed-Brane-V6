import time

from vantage6.tools.util import info

import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

import torch.optim as optim
from opacus import PrivacyEngine
from vantage6.tools.util import info, warn
from torchvision import transforms
import argparse
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import numpy as np

def master(client, data):
    """Combine partials to global model
    """

    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get("id") for organization in organizations]

    # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize model and send parameters of server to all workers
    model = Net().to(device)

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'model': model,
                'parameters': model.parameters(),
                'device': device,
                'log_interval': 10,
                'local_dp': False,  # throws error if true, maybe confilction with opacus version
                'return_params': True,
                'epoch': 1,
                'if_test': False
            }
        }, organization_ids=ids
    )

    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)


    info("Obtaining results")

    results = client.get_results(task_id=task.get("id"))

    global_sum = 0
    global_count = 0


    for output in results:
        # print(len(output))
        global_sum += output["params"]
        global_count += len(global_sum)


    averaged_parameters = global_sum / global_count

    # in order to not have the optimizer see the new parameters as a non-leaf tensor, .clone().detach() needs
    # to be applied in order to turn turn "grad_fn=<DivBackward0>" into "grad_fn=True"
    averaged_parameters = [averaged_parameters.clone().detach()]

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'model': output['model'],
                'parameters': averaged_parameters,
                'device': device,
                'log_interval': 10,
                'local_dp': False,
                'return_params': True,
                'epoch': 1,
                'if_test': True
            }
        },
        organization_ids=ids
    )

    results = client.get_results(task_id=task.get("id"))
    for output in results:
        acc = output["test_accuracy"]
    return acc



def RPC_train_test(data, model, parameters, device, log_interval, local_dp, return_params, epoch, if_test):
    """Compute the average partial
    """
    train = data
    train_batch_size = 64
    test_batch_size = 64

    X = (train.iloc[:,1: ].values).astype('float32')
    Y = train.iloc[:,0].values
    print(X.shape)
    features_train, features_test, targets_train, targets_test = train_test_split(X, Y, test_size=0.2,
                                                                                  random_state=42)
    X_train = torch.from_numpy(features_train/255.0)
    X_test = torch.from_numpy(features_test/255.0)

    Y_train = torch.from_numpy(targets_train).type(torch.LongTensor)
    Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)

    train = torch.utils.data.TensorDataset(X_train, Y_train)
    test = torch.utils.data.TensorDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False)

    #if input is train.pt
    # train_loader = data

    learning_rate = 0.01

    # if local_dp == True:
    # initializing optimizer and scheduler
    optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.5)

    if local_dp:
        privacy_engine = PrivacyEngine(model, batch_size=64,
                                       sample_size=60000, alphas=range(2, 32), noise_multiplier=1.3,
                                       max_grad_norm=1.0, )
        privacy_engine.attach(optimizer)


    test_accuracy=0
    if if_test:
        model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # Send the local and target to the device (cpu/gpu) the model is at
                data, target = data.to(device), target.to(device)
                # Run the model on the local
                batch_size = data.shape[0]
                # print(batch_size)
                data = data.reshape(batch_size, 28, 28)
                data = data.unsqueeze(1)
                output = model(data)
                # Calculate the loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # Check whether prediction was correct
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))
            test_accuracy = 100. * correct / len(test_loader.dataset)

    else:
        model.train()
        for epoch in range(1, epoch + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                batch_size = data.shape[0]
                # print(batch_size)
                data = data.reshape(batch_size, 28, 28)
                data = data.unsqueeze(1)
                # print(data.shape)
                # print(data.type())
                # print(target.type())
                output = model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
    if return_params:
        for parameters in model.parameters():
            return {'params': parameters,
                    'model': model,
                    'test_accuracy': test_accuracy}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
