import json
import os

import argparse

from experimnet_utilities import Experiment
from logger_utilities import Logger
from pnml_utilities import execute_pnml_on_testset
from train_utilities import execute_basic_training
from train_utilities import freeze_model_layers

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python src/main.py -t pnml_cifar10
"""


def run_experiment(experiment_type: str):
    ################
    # Load training params
    with open(os.path.join('src', 'params.json')) as f:
        params = json.load(f)

    ################
    # Class that depends ins the experiment type
    experiment_h = Experiment(experiment_type, params)
    params = experiment_h.get_params()

    ################
    # Create logger and save params to output folder
    logger = Logger(experiment_type=experiment_h.get_exp_name(), output_root='output')
    # logger = Logger(experiment_type='TMP', output_root='output')
    logger.info('OutputDirectory: %s' % logger.output_folder)
    with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
    logger.info(params)

    ################
    # Load datasets
    data_folder = './data'
    logger.info('Load datasets: %s' % data_folder)
    dataloaders = experiment_h.get_dataloaders(data_folder)

    ################
    # Run basic training- so the base model will be in the same conditions as NML model
    model_base = experiment_h.get_model()
    params_init_training = params['initial_training']
    params_init_training['debug_flags'] = params['debug_flags']
    model_erm = execute_basic_training(model_base, dataloaders, params_init_training, experiment_h, logger)

    ################
    # Freeze layers
    logger.info('Freeze layer: %d' % params['freeze_layer'])
    model_erm = freeze_model_layers(model_erm, params['freeze_layer'], logger)

    ############################
    # Iterate over test dataset
    logger.info('Execute pNML')
    params_fit_to_sample = params['fit_to_sample']
    params_fit_to_sample['debug_flags'] = params['debug_flags']
    execute_pnml_on_testset(model_erm, experiment_h, params_fit_to_sample, dataloaders, logger)
    logger.info('Finish All!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-t', '--experiment_type', default='pnml_cifar10',
                        help='Type of experiment to execute',
                        type=str)
    args = vars(parser.parse_args())

    # Available experiment_type:
    #   'pnml_cifar10'
    #   'random_labels'
    #   'out_of_dist_svhn'
    #   'out_of_dist_noise'
    #   'pnml_mnist'
    #   'pnml_cifar10_lenet'

    run_experiment(args['experiment_type'])
    print('Finish experiment')
