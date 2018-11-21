import logging
import sys
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from dataset_utilities import insert_sample_to_dataset


class TrainClass:
    def __init__(self, params_to_train, learning_rate, momentum, step_size, gamma, weight_decay,
                 logger=None):

        self.num_epochs = 20
        self.logger = logger if logger is not None else logging.StreamHandler(sys.stdout)
        self.model = None
        self.eval_test_during_train = True
        self.eval_test_in_end = True
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
        self.freeze_batch_norm = True

    def train_model(self, model, dataloaders, num_epochs=10, loss_goal=None):
        # self.model = copy.deepcopy(model)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.num_epochs = num_epochs
        train_loss, train_acc = torch.tensor([-1.]), torch.tensor([-1.])
        test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])
        epoch = 0
        epoch_time = 0

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
                             (epoch, self.num_epochs - 1,
                              train_loss, test_loss, train_acc, test_acc,
                              epoch_time))

            # Stop training if desired goal is achieved
            if loss_goal is not None and train_loss < loss_goal:
                break
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
        if self.freeze_batch_norm is True:
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
            loss = self.criterion(outputs, labels)  # Negative log-loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_loss += loss * len(images)  # loss sum for all the batch

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
                test_loss += loss * len(data)  # loss sum for all the batch
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


def execute_nml_training(train_params, dataloaders_input, sample_test_data, sample_test_true_label, idx,
                         model_base_input, logger):
    # Check train_params contains all required keys
    required_keys = ['lr', 'momentum', 'step_size', 'gamma', 'weight_decay', 'epochs']
    for key in required_keys:
        if key not in train_params:
            logger.logger.error('The key: %s is not in train_params' % key)
            raise ValueError('The key: %s is not in train_params' % key)

    classes_trained = dataloaders_input['classes']
    if 'classes_cifar100' in dataloaders_input:
        classes_true = dataloaders_input['classes_cifar100']
    elif 'classes_svhn' in dataloaders_input:
        classes_true = dataloaders_input['classes_svhn']
    elif 'classes_noise' in dataloaders_input:
        classes_true = dataloaders_input['classes_noise']
    else:
        classes_true = classes_trained

    # Iteration of all labels
    for trained_label in range(len(classes_trained)):
        time_trained_label_start = time.time()

        # Insert test sample to train dataset
        dataloaders = deepcopy(dataloaders_input)
        trainloader_with_sample = insert_sample_to_dataset(dataloaders['train'], sample_test_data, trained_label)
        dataloaders['train'] = trainloader_with_sample

        # Train model
        model = deepcopy(model_base_input)
        model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        train_class = TrainClass(filter(lambda p: p.requires_grad, model.parameters()),
                                 train_params['lr'], train_params['momentum'], train_params['step_size'],
                                 train_params['gamma'], train_params['weight_decay'],
                                 logger.logger)
        train_class.eval_test_during_train = False
        train_class.freeze_batch_norm = True
        model, train_loss, test_loss = train_class.train_model(model, dataloaders, train_params['epochs'])
        model = model.module if torch.cuda.device_count() > 1 else model
        time_trained_label = time.time() - time_trained_label_start

        # Evaluate trained model
        if sample_test_data.shape == (28, 28):
            # Mnist case
            sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data.
                                                                           unsqueeze(2).
                                                                           numpy())
        else:
            # CIFA10, NOISE, SVHN, CIFAR100 case
            sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data)
        prob, pred = eval_single_sample(model, sample_test_data_trans)

        # Save to file
        logger.add_entry_to_results_dict(idx, str(trained_label), prob, train_loss, test_loss)
        logger.info(
            'idx=%d trained_label=[%d,%s], true_label=[%d,%s], loss [train, test]=[%f %f], time=%4.2f[sec]' %
            (idx, trained_label, classes_trained[trained_label],
             sample_test_true_label, classes_true[sample_test_true_label],
             train_loss, test_loss,
             time_trained_label))


def freeze_resnet_layers(model, max_freeze_layer, logger):
    # todo: currently max_freeze_layer 0 and 1 are the same. move ct+=1 to the end
    ct = 0
    for child in model.children():
        ct += 1
        if ct < max_freeze_layer:
            logger.info('Freeze Layer: idx={}, name={}'.format(ct, child))
            for param in child.parameters():
                param.requires_grad = False
            continue
        logger.info('UnFreeze Layer: idx={}, name={}'.format(ct, child))
    return model
