from __future__ import print_function

import json
import os
import time

import torch

from dataset_utilities import create_cifar10_random_label_dataloaders
from logger_utilities import Logger
from resnet import resnet20
from wide_resnet import WideResNet
from train_utilities import TrainClass, eval_single_sample
from train_utilities import execute_nml_training

"""
Example of running:
CUDA_VISIBLE_DEVICES=0,1 python src/main_NML_random_labels.py
"""

# Load training params
with open(os.path.join('src', 'params.json')) as f:
    params = json.load(f)
params = params['random_labels']

# Create logger and save params to output folder
logger = Logger(experiment_type='Random_Labels', output_root='output')
# logger = Logger(experiment_type='TMP', output_root='output')
logger.logger.info('OutputDirectory: %s' % logger.output_folder)
with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    outfile.write(json.dumps(params, indent=4, sort_keys=True))

################
# Load datasets
trainloader, testloader, classes = create_cifar10_random_label_dataloaders(
    './data', params['batch_size'], params['num_workers'],
    label_corrupt_prob=params['label_corrupt_prob'])
dataloaders = {'train': trainloader, 'test': testloader, 'classes': classes}

################
# Run basic training- so the base model will be in the same conditions as NML model
params_initial_training = params['initial_training']
# model_base = resnet20()
model_base = WideResNet()
if params_initial_training['do_initial_training'] is True:
    logger.logger.info('Execute basic training')
    model_base = torch.nn.DataParallel(model_base) if torch.cuda.device_count() > 1 else model_base
    train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                             params_initial_training['lr'],
                             params_initial_training['momentum'],
                             params_initial_training['step_size'],
                             params_initial_training['gamma'],
                             params_initial_training['weight_decay'],
                             logger.logger)
    train_class.eval_test_during_train = False
    train_class.freeze_batch_norm = False
    model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders,
                                                                params_initial_training['epochs'],
                                                                params_initial_training['loss_goal'])
    model_base = model_base.module if torch.cuda.device_count() > 1 else model_base
    torch.save(model_base.state_dict(), os.path.join(logger.output_folder, 'random_labels_model_%f.pt' % train_loss))
else:
    logger.logger.info('Load pretrained model')
    model_base.load_state_dict(torch.load(params_initial_training['pretrained_model_path']))
    model_base = model_base.cuda() if torch.cuda.is_available() else model_base

############################
# Iterate over test dataset
logger.logger.info('Iterate over test dataset')
params_fit_to_sample = params['fit_to_sample']
for idx in range(params_fit_to_sample['test_start_idx'], params_fit_to_sample['test_end_idx'] + 1):
    time_start_idx = time.time()

    # Extract a sample from test dataset and check output of base model
    sample_test_data = dataloaders['test'].dataset.test_data[idx]
    sample_test_true_label = dataloaders['test'].dataset.test_labels[idx]
    prob_org, _ = eval_single_sample(model_base, testloader.dataset.transform(sample_test_data))
    logger.add_org_prob_to_results_dict(idx, prob_org, sample_test_true_label)

    # NML training- train the model with test sample
    execute_nml_training(params_fit_to_sample, dataloaders, idx, model_base, logger)

    # Log and save
    logger.save_json_file()
    time_idx = time.time() - time_start_idx
    logger.logger.info('----- Finish NML random labels idx = %d, time=%f[sec] ----' % (idx, time_idx))
logger.logger.info('Finish All!')
