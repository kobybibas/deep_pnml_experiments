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
from torch.autograd import Variable

def progress_bar(value, end_value, iter_time_sec, prefix_str='', bar_length=20):
    """
    Prints progress bar on the console in the same line.

    :param value: (int/float) current value of the progress
    :param end_value: (int/float)  end value for which the progress will be ended
    :param iter_time_sec: (float) time per iteration
    :param prefix_str: (str) prefix string which will be printed before the progress bar
    :param bar_length: (int) total bar length in the consloe
    :return: None
    """

    percent = float(value) / end_value
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r%s [%s] %d%%  %4.2f[sec/iter]" % (prefix_str, arrow + spaces,
                                                           int(round(percent * 100)), iter_time_sec))
    sys.stdout.flush()

    # # new line
    if value == end_value:
        print('')


class TrainClass:
    def __init__(self, params_to_train, learning_rate, momentum, step_size, gamma,
                 logger=None,
                 models_save_path='models'):

        self.num_epochs = 20
        self.save_path = models_save_path
        self.logger = logger if logger is not None else logging.StreamHandler(sys.stdout)
        self.model = None
        self.eval_test_during_train = True
        self.save_name = 'cifar10_model.pt'

        # Optimizer
        self.optimizer = optim.SGD(params_to_train,
                                   lr=learning_rate,
                                   momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)

        # create save path if not exsits
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)


    def train_model(self, model, dataloaders, num_epochs=10):

        self.model = model.cuda() if torch.cuda.is_available() else model
        self.num_epochs = num_epochs
        #self.logger.info('Start training for %d epochs' % num_epochs)
        print('Start training for %d epochs' % num_epochs)
        # Loop on epochs
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            dataset_size = 0

            epoch_start_time = time.time()
            train_loss, train_acc = self.train(dataloaders['train'])

            if self.eval_test_during_train is True:
                test_loss, test_acc = self.test(dataloaders['test'])
            else:
                test_loss, test_acc = 0, 0
            epoch_time = time.time() - epoch_start_time

            self.logger.info('[%d/%d] [train test] loss =[%f %f], acc=[%f %f] epoch_time=%f' %
                             (epoch, self.num_epochs,
                              train_loss, test_loss, train_acc, test_acc,
                              epoch_time))

        test_loss, test_acc = self.test(dataloaders['test'])
        self.logger.info('[%d/%d] [train test] loss =[%f %f], acc=[%f %f] epoch_time=%f' %
                         (epoch, self.num_epochs,
                          train_loss, test_loss, train_acc, test_acc,
                          epoch_time))

        torch.save(self.model.state_dict(),
                   os.path.join(self.save_path, self.save_name % test_loss))
        return self.model, test_loss

    def train(self, train_loader):
        self.model.train()
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
                test_loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        acc = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        return test_loss, acc

    def train_model_with_test_sample(self, model, train_loader, test_sample, num_epochs=10):
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.num_epochs = num_epochs

        # Loop on epochs
        for epoch in range(self.num_epochs):

            # Train
            train_loss = self.train_with_test_sample(train_loader, test_sample)

            # Test the sample
            self.model.eval()
            sample_data = test_sample[0].cuda() if torch.cuda.is_available() else test_sample[0]
            sample_label = test_sample[1].cuda() if torch.cuda.is_available() else test_sample[1]
            output = self.model(sample_data.unsqueeze(0))
            test_loss = F.nll_loss(output, sample_label.unsqueeze(0), size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            prob = F.softmax(output, dim=-1)

        return self.model, prob, train_loss, test_loss

    def train_with_test_sample(self, train_loader, test_sample):

        self.model.train()
        train_loss = 0
        # Iterate over dataloaders
        for iter_num, (images, labels) in enumerate(train_loader):

            # Forceing train on test sample
            # if iter_num == 0:
            #     images[0] = test_sample[0]
            #     labels[0] = test_sample[1]

            # Adjust to CUDA
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.nll_loss(outputs, labels)  # Negetive log-loss
            train_loss += loss

            # Back-propagation
            loss.backward()
            self.optimizer.step()

        train_loss /= len(train_loader.dataset)
        return train_loss

    def eval_single_sample(self, test_sample):
        # test_sample = (data, label)

        # Test the sample
        sample_data = test_sample[0].cuda() if torch.cuda.is_available() else test_sample[0]
        sample_label = test_sample[1].cuda() if torch.cuda.is_available() else test_sample[1]
        output = self.model(sample_data.unsqueeze(0))
        test_loss = self.criterion(output, sample_label.unsqueeze(0)).item()

        # Prediction
        pred = output.max(1, keepdim=True)[1]
        pred = pred.cpu().detach().numpy()[0][0]

        # Extract prob
        prob = F.softmax(output, dim=-1)
        prob = prob.cpu().detach().numpy().round(16).tolist()[0]
        return prob, pred