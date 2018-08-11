from __future__ import print_function
from train_utilities import TrainClass, eval_single_sample
from logger_utilities import Logger
from resnet import resnet20, load_pretrained_resnet20_cifar10_model
from dataset_utilities import create_cifar10_dataloaders
import json
import time
import os


# Training params
with open(os.path.join('src', 'params.json')) as f:
    params = json.load(f)

# Create logger
logger = Logger(experiment_type='ERM', output_root='output')

# Save params to output folder
with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    outfile.write(json.dumps(params, indent=4, sort_keys=True))

################
# Load datasets
logger.logger.info('Load datasets')
trainloader, testloader, classes = create_cifar10_dataloaders('../data', params['batch_size'], params['num_workers'])
dataloaders = {'train': trainloader, 'test': testloader}

################
# Load model
logger.logger.info('Load model')
model = resnet20()  # ResNet18
model = load_pretrained_resnet20_cifar10_model(model)

################
# Train
time_start = time.time()
train_class = TrainClass(model.parameters(),
                         params['lr'], params['momentum'], params['step_size'],
                         params['gamma'], params['weight_decay'],
                         logger=logger.logger,
                         models_save_path=logger.output_folder)
train_class.eval_test_during_train = True
model, train_loss, test_loss = train_class.train_model(model, dataloaders, params['epochs'])
print('Finish train: %f[sec]' % (time.time() - time_start))

model.eval()
for idx in range(100):

    test_data, test_label = trainloader.testset[idx]
    prob, pred = eval_single_sample(model, testloader.testset.transform(test_data))

    # Save to file
    logger.add_entry_to_results_dict(idx, test_label, 'prob', prob, train_loss, test_loss)
    logger.save_json_file()

