import os
import time
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import sys
import logging
import pathlib
import copy


class TrainClass:
    def __init__(self, params_to_train, learning_rate, momentum, step_size, gamma, weight_decay,
                 logger=None,
                 models_save_path='output'):

        self.num_epochs = 20
        self.save_path = models_save_path
        self.logger = logger if logger is not None else logging.StreamHandler(sys.stdout)
        self.model = None
        self.eval_test_during_train = True
        self.eval_test_in_end = True
        self.save_name = 'cifar10_trained_model.pt'
        self.print_during_train = True

        # Optimizer
        self.optimizer = optim.SGD(params_to_train,
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=step_size,
                                                        gamma=gamma)

        # Create save path if not exists
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def train_model(self, model, dataloaders, num_epochs=10):
        # self.model = copy.deepcopy(model)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.num_epochs = num_epochs
        train_loss, train_acc = torch.tensor([-1.]), torch.tensor([-1.])
        test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])

        # Loop on epochs
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()
            train_loss, train_acc = self.train(dataloaders['train'])
            if self.eval_test_during_train is True:
                test_loss, test_acc = self.test(dataloaders['test'])
            else:
                test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])
            epoch_time = time.time() - epoch_start_time

            self.logger.info('[%d/%d] [train test] loss =[%f %f], acc=[%f %f] epoch_time=%f' %
                             (epoch, self.num_epochs-1,
                              train_loss, test_loss, train_acc, test_acc,
                              epoch_time))
        test_loss, test_acc = self.test(dataloaders['test'])

        # Print and save
        self.logger.info('[%d/%d] [train test] loss =[%f %f], acc=[%f %f] epoch_time=%f' %
                         (epoch, self.num_epochs,
                          train_loss, test_loss, train_acc, test_acc,
                          epoch_time))
        train_loss_output = float(train_loss.cpu().detach().numpy().round(16))
        test_loss_output = float(test_loss.cpu().detach().numpy().round(16))
        return self.model, train_loss_output, test_loss_output

    def train(self, train_loader):
        self.model.train()

        # Turn off batch normalization update
        self.model = self.model.apply(set_bn_eval)

        train_loss = 0
        correct = 0
        # Iterate over dataloaders
        for iter_num, (images, labels) in enumerate(train_loader):

            # Adjust to CUDA
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)  # Negetive log-loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_loss += loss

            # Back-propagation
            loss.backward()
            self.optimizer.step()

        self.optimizer.step()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        return train_loss, train_acc

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.cuda() if torch.cuda.is_available() else data
                labels = labels.cuda() if torch.cuda.is_available() else labels

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                test_loss += loss
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_acc = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        return test_loss, test_acc


def eval_single_sample(model, test_sample_data):
    # test_sample = (data, label)

    # Test the sample
    model.eval()
    sample_data = test_sample_data.cuda() if torch.cuda.is_available() else test_sample_data
    output = model(sample_data.unsqueeze(0))

    # Prediction
    pred = output.max(1, keepdim=True)[1]
    pred = pred.cpu().detach().numpy().round(16)[0][0]

    # Extract prob
    prob = F.softmax(output, dim=-1)
    prob = prob.cpu().detach().numpy().round(16).tolist()[0]
    return prob, pred


def set_bn_eval(model):
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        model.eval()

