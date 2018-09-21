from __future__ import print_function

import json
import os
import time

import torch

from dataset_utilities import create_cifar10_dataloaders_with_training_subset
from logger_utilities import Logger
from resnet import resnet20
from train_utilities import TrainClass, eval_single_sample
from train_utilities import execute_nml_training

# Load training params
with open(os.path.join('src', 'params.json')) as f:
    params = json.load(f)
params = params['trainset_subset']

# Create logger and save params to output folder
logger = Logger(experiment_type='NML_trainset_subset', output_root='output')
# logger = Logger(experiment_type='TMP', output_root='output')
logger.logger.info('OutputDirectory: %s' % logger.output_folder)
with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    outfile.write(json.dumps(params, indent=4, sort_keys=True))

################
# Load datasets
trainloader, testloader, classes = create_cifar10_dataloaders_with_training_subset(
    '../data', params['batch_size'], params['num_workers'], params['trainset_size'])
dataloaders = {'train': trainloader, 'test': testloader, 'classes': classes}

################
# Run basic training
logger.logger.info('Execute basic training')
train_params = params['initial_training']
model_base = resnet20()
model_base = torch.nn.DataParallel(model_base) if torch.cuda.device_count() > 1 else model_base
train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                         train_params['lr'], train_params['momentum'], train_params['step_size'],
                         train_params['gamma'], train_params['weight_decay'],
                         logger.logger)
train_class.eval_test_during_train = False
train_class.freeze_batch_norm = False
model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders, train_params['epochs'])
model_base = model_base.module if torch.cuda.device_count() > 1 else model_base
torch.save(model_base.state_dict(), os.path.join(logger.output_folder, 'base_model_%f.pt' % train_loss))

############################
# Iterate over test dataset
logger.logger.info('Iterate over test dataset')
for idx in range(params['fit_to_sample']['test_start_idx'], params['fit_to_sample']['test_end_idx'] + 1):
    time_start_idx = time.time()

    # Extract a sample from test dataset and check output of base model
    sample_test_data = dataloaders['test'].dataset.test_data[idx]
    sample_test_true_label = dataloaders['test'].dataset.test_labels[idx]
    prob_org, _ = eval_single_sample(model_base, testloader.dataset.transform(sample_test_data))
    logger.add_org_prob_to_results_dict(idx, prob_org, sample_test_true_label)

    # NML training- train the model with test sample
    execute_nml_training(params['fit_to_sample'], dataloaders, idx, model_base, logger)

    # Log and save
    logger.save_json_file()
    time_idx = time.time() - time_start_idx
    logger.logger.info('------------ Finish NML idx = %d, time=%f[sec] ------------' % (idx, time_idx))
logger.logger.info('Finish All!')
