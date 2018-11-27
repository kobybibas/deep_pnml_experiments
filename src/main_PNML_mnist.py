import json
import os
import time

from mpl import Net
from dataset_utilities import create_mnist_dataloaders
from logger_utilities import Logger
from train_utilities import TrainClass, eval_single_sample, execute_nml_training
from train_utilities import freeze_resnet_layers

# Load training params
with open(os.path.join('src', 'params.json')) as f:
    params = json.load(f)
params = params['nml_mnist']

# Create logger and save params to output folder
logger = Logger(experiment_type='NML_Mnist', output_root='output')
# logger = Logger(experiment_type='TMP', output_root='output')
logger.info('OutputDirectory: %s' % logger.output_folder)
with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    outfile.write(json.dumps(params, indent=4, sort_keys=True))

################
# Load datasets
trainloader, testloader, classes = create_mnist_dataloaders('./data', params['batch_size'], params['num_workers'])
dataloaders = {'train': trainloader, 'test': testloader, 'classes': classes}

################
# Run basic training
logger.info('Execute basic training')
params_initial_training = params['initial_training']
model_base = Net()
train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                         params_initial_training['lr'],
                         params_initial_training['momentum'],
                         params_initial_training['step_size'],
                         params_initial_training['gamma'],
                         params_initial_training['weight_decay'],
                         logger.logger)
train_class.eval_test_during_train = True
model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders,
                                                            params_initial_training['epochs'])

################
# Freeze layers
logger.info('Freeze layer: %d' % params['freeze_layer'])
model_base = freeze_resnet_layers(model_base, params['freeze_layer'] + 1, logger)

############################
# Iterate over test dataset
logger.info('Iterate over test dataset')
params_fit_to_sample = params['fit_to_sample']
for idx in range(params_fit_to_sample['test_start_idx'], params_fit_to_sample['test_end_idx'] + 1):
    time_start_idx = time.time()

    # Extract a sample from test dataset and check output of base model
    sample_test_data = dataloaders['test'].dataset.test_data[idx]
    sample_test_true_label = dataloaders['test'].dataset.test_labels[idx]

    # Evaluate Base model
    if sample_test_data.shape == (28, 28):
        # Mnist case
        sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data.
                                                                       unsqueeze(2).
                                                                       numpy())
    else:
        # CIFA10, NOISE, SVHN, CIFAR100 case
        sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data)
    prob_org, _ = eval_single_sample(model_base, sample_test_data_trans)
    logger.add_org_prob_to_results_dict(idx, prob_org, sample_test_true_label)

    # NML training- train the model with test sample
    execute_nml_training(params_fit_to_sample, dataloaders, sample_test_data, sample_test_true_label, idx,
                         model_base, logger)

    # Log and save
    logger.save_json_file()
    time_idx = time.time() - time_start_idx
    logger.info('----- Finish NML MNIST idx = %d, time=%f[sec] ----' % (idx, time_idx))
logger.info('Finish All!')
