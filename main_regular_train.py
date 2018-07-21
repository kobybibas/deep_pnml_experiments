from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torchvision
from net_architecture import Net
from train_utilities import TrainClass
import time
from logger_utilities import Logger
import numpy as np

# Training settings
params = {}
params['batch_size'] = 64
params['epochs'] = 10
params['lr'] = 0.01
params['momentum'] = 0.9
params['step_size'] = 20  # step_size > epochs -> not doing anything
params['gamma'] = 0.5

# Create logger
logger = Logger()
unique_time = time.strftime("%Y%m%d%H%M")
logger.define_log_file('results/log_regular_train_%s.log' % unique_time)
logger.define_json_output('results/results_regular_train_%s.json' % unique_time)

################
# Load datasets
logger.logger.info('Load datasets')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=params['batch_size'],
                                          shuffle=True,
                                          num_workers=4)

testset = torchvision.datasets.CIFAR10(root='../data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=params['batch_size'],
                                         shuffle=False,
                                         num_workers=4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataloaders = {'train': trainloader, 'test': testloader}

################
# Train
time_start = time.time()
model = Net()
train_class = TrainClass(model.parameters(),
                         params['lr'],
                         params['momentum'],
                         params['step_size'],
                         params['gamma'],
                         logger=logger.logger)
train_class.save_name = 'cifar10_regular_trian_%s.pt' % unique_time
train_class.eval_test_during_train = False
model, test_loss = train_class.train_model(model, dataloaders, params['epochs'])
print('Finish train: %f[sec]' % (time.time() - time_start))

model.eval()
for idx in range(100):

    test_data, test_label = testset[idx]
    prob, pred = train_class.eval_single_sample((test_data, torch.tensor(test_label)))

    # Save to file
    logger.add_entry_to_results_dict(idx, test_label, 'prob', prob)
    logger.save_json_file()

