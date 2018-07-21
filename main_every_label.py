from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torchvision
from net_architecture import Net
from train_utilities import TrainClass
import time
from logger_utilities import Logger
import numpy as np
import copy

params = {}
params['batch_size'] = 64
params['epochs'] = 10
params['lr'] = 0.001
params['momentum'] = 0.9
params['seed'] = 1
params['test_start_idx'] = 0
params['test_end_idx'] = 0


# Create logger
logger = Logger()
logger.define_log_file('log_idxes_%d_%d.log' %
                       (params['test_start_idx'], params['test_end_idx']))
logger.define_json_output('results_idxes_%d_%d.json' %
                          (params['test_start_idx'], params['test_end_idx']))

################
# Load datasets
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

############################
# Iterate over test dataset
logger.logger.info('Iterate over test dataset')
for idx in range(params['test_start_idx'], params['test_end_idx']+1):
    idx = 2
    time_start_idx = time.time()

    # Extract a sample from test dataset
    sample_test_data, sample_test_true_label = dataset_test[idx]

    ############################
    # Iteration of all labels
    for trained_label in range(10):
        trained_label = 1
        time_trained_label_start = time.time()

        sample_test_to_train = (sample_test_data, torch.tensor(trained_label))

        # Add the test sample to the train dataset
        print(dataset_train.__len__())
        dataset_train_with_test_sample = copy.deepcopy(dataset_train)
        dataset_train_with_test_sample.train_data = torch.cat((dataset_train.train_data, dataset_test.test_data[idx].unsqueeze(0)), 0)
        dataset_train_with_test_sample.train_labels = torch.cat((dataset_train.train_labels, dataset_test.test_labels[idx].unsqueeze(0)), 0)
        train_loader_with_test_sample = torch.utils.data.DataLoader(dataset_train_with_test_sample,
                                                   batch_size=params['batch_size'],
                                                   shuffle=True)

        print(dataset_train.__len__())
        print(dataset_train_with_test_sample.__len__())

        model = Net()
        train_class = TrainClass(model.parameters(),
                                 params['lr'],
                                 params['momentum'],
                                 logger.logger)
        model, prob, train_loss, test_loss = \
            train_class.train_model_with_test_sample(model,
                                                     train_loader_with_test_sample,
                                                     sample_test_to_train,
                                                     params['epochs'])
        # Add to dict and print
        time_trained_label = time.time() - time_trained_label_start
        logger.add_entry_to_results_dict(idx, sample_test_true_label, trained_label, prob)
        logger.save_json_file()
        prob_str = " ".join(str(x.cpu().detach().numpy()) for x in prob)
        logger.logger.info('idx=%d, [true trained] =[%d %d], loss [train, test]=[%f %f], time=%4.2f[sec]' %
                           (idx, sample_test_true_label, trained_label,
                            train_loss, test_loss,
                            time_trained_label))
        logger.logger.info(
            '    ' + np.array2string(prob.cpu().detach().numpy(), precision=16, separator=', ').replace('\n', ''))

        z = 1
    # Finish idx
    time_idx = time.time()-time_start_idx
    logger.logger.info('------- Finish idx = %d, time=%f[sec] --------' % (idx, time_idx))


