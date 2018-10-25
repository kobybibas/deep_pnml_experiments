from __future__ import print_function

import json
import os
import time

import torch

from dataset_utilities import create_cifar10_dataloaders, generate_noise_sample
from dataset_utilities import insert_sample_to_dataset
from logger_utilities import Logger
from resnet import resnet20, load_pretrained_resnet20_cifar10_model
from train_utilities import TrainClass, eval_single_sample

# Load
with open(os.path.join('src', 'params.json')) as f:
    params = json.load(f)

# Create logger and save params to output folder
logger = Logger(experiment_type='OutOfDist_Noise', output_root='output')
logger.info('OutputDirectory: %s' % logger.output_folder)
with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    outfile.write(json.dumps(params, indent=4, sort_keys=True))

################
# Load base model and datasets
data_folder = './data'
trainloader, testloader, classes = create_cifar10_dataloaders(data_folder, params['batch_size'], params['num_workers'])
dataloaders = {'train': trainloader, 'test': testloader}

################
# Run basic training- so the base model will be in the same conditions as NML model
logger.info('Execute basic training')
model_base = load_pretrained_resnet20_cifar10_model(resnet20())
model_base = torch.nn.DataParallel(model_base) if torch.cuda.device_count() > 1 else model_base
train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                         params['lr'], params['momentum'], params['step_size'],
                         params['gamma'], params['weight_decay'],
                         logger.logger,
                         models_save_path=logger.output_folder)
# todo: use the same template as nml
train_class.eval_test_during_train = False
model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders, params['epochs'])
model_base = model_base.module if torch.cuda.device_count() > 1 else model_base

################
# Iterate over test dataset
logger.info('Iterate over test dataset')
for idx in range(params['test_start_idx'], params['test_end_idx'] + 1):
    time_start_idx = time.time()

    # Extract a sample from test dataset and check output of base model
    sample_test_data, sample_test_true_label = generate_noise_sample()
    prob_org, _ = eval_single_sample(model_base, testloader.dataset.transform(sample_test_data))

    ############################
    # Iteration of all labels
    for trained_label in range(len(classes)):
        time_trained_label_start = time.time()

        # Insert test sample to train dataset
        trainloader_with_sample = insert_sample_to_dataset(trainloader, sample_test_data, trained_label)
        dataloaders = {'train': trainloader_with_sample, 'test': testloader}

        # Train model
        model = load_pretrained_resnet20_cifar10_model(resnet20())
        model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        train_class = TrainClass(filter(lambda p: p.requires_grad, model.parameters()),
                                 params['lr'], params['momentum'], params['step_size'],
                                 params['gamma'], params['weight_decay'],
                                 logger.logger,
                                 models_save_path=logger.output_folder)
        train_class.eval_test_during_train = False
        model, train_loss, test_loss = train_class.train_model(model, dataloaders, params['epochs'])
        model = model.module if torch.cuda.device_count() > 1 else model
        time_trained_label = time.time() - time_trained_label_start

        # Add to dict and print
        prob, pred = eval_single_sample(model, testloader.dataset.transform(sample_test_data))

        # Save to file
        logger.add_entry_to_results_dict(idx, sample_test_true_label, str(trained_label), prob,
                                         train_loss, test_loss, prob_org)
        logger.save_json_file()
        logger.info(
            'idx=%d trained_label=[%d,%s], true_label=[%d,%s], loss [train, test]=[%f %f], time=%4.2f[sec]'
            % (idx, trained_label, classes[trained_label],
               sample_test_true_label, 'Noise',
               train_loss, test_loss,
               time_trained_label))
        prob_str = " ".join(str(x) for x in prob)
        logger.info('    Prob: %s' % prob_str)

    time_idx = time.time() - time_start_idx
    logger.info(
        '--- Finish OutOfDist_Noise idx = %d, time=%f[sec], outputs in %s' % (idx, time_idx, logger.output_folder))
logger.info('Finish All!')
