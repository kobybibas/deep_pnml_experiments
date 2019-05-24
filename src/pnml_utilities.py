import time
from copy import deepcopy

import numpy as np

from dataset_utilities import insert_sample_to_dataset
from experimnet_utilities import Experiment
from logger_utilities import Logger
from train_utilities import TrainClass
from train_utilities import eval_single_sample


def execute_pnml_on_testset(model_base, experiment_h: Experiment, params_fit_to_sample: dict, dataloaders: dict,
                            logger: Logger):
    for idx in range(params_fit_to_sample['test_start_idx'], params_fit_to_sample['test_end_idx'] + 1):
        time_start_idx = time.time()

        # Extract a sample from test dataset and check output of base model
        sample_test_data = dataloaders['test'].dataset.test_data[idx]
        sample_test_true_label = dataloaders['test'].dataset.test_labels[idx]

        # Make sure the data is HxWxC:
        if len(sample_test_data.shape) == 3 and sample_test_data.shape[2] > sample_test_data.shape[0]:
            sample_test_data = sample_test_data.transpose([1, 2, 0])

        # Execute transformation
        sample_test_data_for_trans = deepcopy(sample_test_data)
        if len(sample_test_data.shape) == 2:
            sample_test_data_for_trans = sample_test_data_for_trans.unsqueeze(2).numpy()
        sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data_for_trans)

        # Evaluate with base model
        prob_org, _ = eval_single_sample(model_base, sample_test_data_trans)
        logger.add_org_prob_to_results_dict(idx, prob_org, sample_test_true_label)

        # NML training- train the model with test sample
        execute_pnml_training(params_fit_to_sample, dataloaders, sample_test_data, sample_test_true_label, idx,
                              model_base, logger)

        # Log and save
        logger.save_json_file()
        time_idx = time.time() - time_start_idx
        logger.info('----- Finish %s idx = %d, time=%f[sec] ----' % (experiment_h.get_exp_name(), idx, time_idx))


def execute_pnml_training(train_params: dict, dataloaders_input: dict,
                          sample_test_data, sample_test_true_label, idx: int,
                          model_base_input, logger):
    """
    Execute the PNML procedure: for each label train the model and save the prediction afterword.
    :param train_params: parameters of training the model for each label
    :param dataloaders_input: dataloaders which contains the trainset
    :param sample_test_data: the data of the test sample that will be evaluated
    :param sample_test_true_label: the true label of the test sample
    :param idx: the index in the testset dataset of the test sample
    :param model_base_input: the base model from which the train will start
    :param logger: logger class to print logs and save results to file
    :return: None
    """

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
        train_class = TrainClass(filter(lambda p: p.requires_grad, model.parameters()),
                                 train_params['lr'], train_params['momentum'], train_params['step_size'],
                                 train_params['gamma'], train_params['weight_decay'],
                                 logger.logger)
        train_class.eval_test_during_train = train_params['debug_flags']['eval_test_during_train']
        train_class.freeze_batch_norm = True
        model, train_loss, test_loss = train_class.train_model(model, dataloaders, train_params['epochs'])
        time_trained_label = time.time() - time_trained_label_start

        # Execute transformation
        sample_test_data_for_trans = deepcopy(sample_test_data)
        if len(sample_test_data.shape) == 2:
            sample_test_data_for_trans = sample_test_data_for_trans.unsqueeze(2).numpy()
        sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data_for_trans)

        # Evaluate with base model
        prob, pred = eval_single_sample(model, sample_test_data_trans)

        # Save to file
        logger.add_entry_to_results_dict(idx, str(trained_label), prob, train_loss, test_loss)
        logger.info(
            'idx=%d trained_label=[%d,%s], true_label=[%d,%s] predict=[%d], loss [train, test]=[%f %f], time=%4.2f[s]'
            % (idx, trained_label, classes_trained[trained_label],
               sample_test_true_label, classes_true[sample_test_true_label],
               np.argmax(prob),
               train_loss, test_loss,
               time_trained_label))
