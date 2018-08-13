from __future__ import print_function
from train_utilities import TrainClass, eval_single_sample
from logger_utilities import Logger
from dataset_utilities import insert_sample_to_dataset
from resnet import resnet20, load_pretrained_resnet20_cifar10_model
from dataset_utilities import create_cifar10_dataloaders
import torch
import time
import json
import os

# Load training params
with open(os.path.join('src', 'params.json')) as f:
    params = json.load(f)

# Create logger and save params to output folder
logger = Logger(experiment_type='NML', output_root='output')
with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    outfile.write(json.dumps(params, indent=4, sort_keys=True))

################
# Load datasets
trainloader, testloader, classes = create_cifar10_dataloaders('../data', params['batch_size'], params['num_workers'])
dataloaders = {'train': trainloader, 'test': testloader}

################
# Run basic training- so the base model will be in the same conditions as NML model
logger.logger.info('Execute basic training')
model_base = load_pretrained_resnet20_cifar10_model(resnet20())
model_base = torch.nn.DataParallel(model_base) if torch.cuda.device_count() > 1 else model_base
train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                         params['lr'], params['momentum'], params['step_size'],
                         params['gamma'], params['weight_decay'],
                         logger.logger,
                         models_save_path=logger.output_folder)
train_class.eval_test_during_train = False
model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders, params['epochs'])
model_base = model_base.module if torch.cuda.device_count() > 1 else model_base

############################
# Iterate over test dataset
logger.logger.info('Iterate over test dataset')
for idx in range(params['test_start_idx'], params['test_end_idx']+1):
    time_start_idx = time.time()

    # Extract a sample from test dataset and check output of base model
    sample_test_data, sample_test_true_label = testloader.dataset.test_data[idx], testloader.dataset.test_labels[idx]
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
        logger.logger.info('idx=%d trained_label=[%d,%s], true_label=[%d,%s], loss [train, test]=[%f %f], time=%4.2f[sec]' %
                           (idx, trained_label, classes[trained_label],
                            sample_test_true_label,  classes[sample_test_true_label],
                            train_loss, test_loss,
                            time_trained_label))
        logger.logger.info('    Prob: %s' % " ".join(str(x) for x in prob))

    time_idx = time.time()-time_start_idx
    logger.logger.info('------------ Finish NML idx = %d, time=%f[sec] ------------' % (idx, time_idx))
logger.logger.info('Finish All!')

