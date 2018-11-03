import json
import os

import numpy as np
import pandas as pd

from dataset_utilities import create_cifar10_dataloaders


def extract_probabilities_list(evaluation_dict):
    # if the sample was trained with label 2, extract the prob to be 2 ...
    # return list of probabilities
    prob_all = []

    true_label = evaluation_dict['true_label'] if 'true_label' in evaluation_dict else None
    prob_org = evaluation_dict['original']['prob']
    for trained_label in evaluation_dict:

        # One of the key is a string, ignore it
        if trained_label.isdigit():
            prob_on_trained = evaluation_dict[trained_label]['prob'][int(trained_label)]
            prob_all.append(prob_on_trained)
    predicted_label = np.argmax(prob_all) if len(prob_all) > 0 else None

    return prob_all, true_label, predicted_label, prob_org


def extract_jinni_probabilities_list(evaluation_dict):
    # Extract to probabilities of the model which was trained with the true label
    # return list of probabilities
    true_label = evaluation_dict['true_label']
    prob_jinni = evaluation_dict[str(true_label)]['prob']
    predicted_jinni_label = np.argmax(prob_jinni)

    return prob_jinni, true_label, predicted_jinni_label


def execute_normalize_prob(prob_list):
    # Normalize the probabilities to be valid distribution
    # Return list of probabilities along with the normalization factor which was used.
    normalization_factor = np.sum(prob_list)
    normalized_prob = np.array(prob_list) / (normalization_factor)  # + np.finfo(float).eps)
    return normalized_prob, normalization_factor


def compute_log_loss(normalized_prob, true_label):
    # Compute the log loss
    return -np.log10(normalized_prob[true_label] + np.finfo(float).eps)


def get_ERM_log_loss_from_dict(results_dict, is_random_labels=False):
    if is_random_labels == True:
        _, testloader, _ = create_cifar10_dataloaders('../data/', 1, 1)

    loss_ERM_list = []
    is_correct_ERM_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        _, true_label, _, prob_org = extract_probabilities_list(sample_dict)

        # If was trained with random- extract the real label
        if is_random_labels == True:
            true_label = testloader.dataset.test_labels[int(keys)]

        loss_ERM = compute_log_loss(prob_org, true_label)
        loss_ERM_list.append(loss_ERM)
        is_correct_ERM_list.append(np.argmax(prob_org) == true_label)

    acc_ERM = np.sum(is_correct_ERM_list) / len(is_correct_ERM_list)
    return loss_ERM_list, acc_ERM, is_correct_ERM_list


def get_NML_log_loss_from_dict(results_dict, is_random_labels=False):
    if is_random_labels == True:
        _, testloader, _ = create_cifar10_dataloaders('../data/', 1, 1)

    loss_NML_list = []
    is_correct_list_NML_list = []
    normalization_factor_list = []
    test_sample_idx_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, _ = extract_probabilities_list(sample_dict)
        normalized_prob, normalization_factor = execute_normalize_prob(prob)

        # If was trained with random - extract the real label
        if is_random_labels == True:
            true_label = testloader.dataset.test_labels[int(keys)]

        # Protection against out of distribution
        if true_label in range(0, len(normalized_prob)):
            loss_NML = compute_log_loss(normalized_prob, true_label)
            loss_NML_list.append(loss_NML)
        normalization_factor_list.append(normalization_factor)
        is_correct_list_NML_list.append(predicted_label == true_label)
        test_sample_idx_list.append(keys)
    acc_NML = np.sum(is_correct_list_NML_list) / len(is_correct_list_NML_list)
    return loss_NML_list, normalization_factor_list, acc_NML, is_correct_list_NML_list, test_sample_idx_list


def extract_probabilities_form_train_loss_list(evaluation_dict, trainset_size=50000):
    # if the sample was trained with label 2, extract the prob to be 2 ...
    # return list of probabilities

    prob_all = []
    true_label = evaluation_dict['true_label']
    prob_org = evaluation_dict['original']['prob']
    for trained_label in evaluation_dict:

        # One of the key is a string, ignore it
        if trained_label.isdigit():
            train_loss = evaluation_dict[trained_label]['train_loss']
            prob_on_trained = np.exp(-(trainset_size + 1) * train_loss)  # adding 1 for the test sample
            prob_all.append(prob_on_trained)
    predicted_label = np.argmax(prob_all) if len(prob_all) > 0 else None

    return prob_all, true_label, predicted_label, prob_org


def get_NML_log_loss_of_the_series_from_dict(results_dict, trainset_size=50000):
    loss_NML_list = []
    is_correct_list = []
    normalization_factor_list = []
    test_sample_idx_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, _ = \
            extract_probabilities_form_train_loss_list(sample_dict, trainset_size)
        normalized_prob, normalization_factor = execute_normalize_prob(prob)

        # Protection against out of distribution
        if true_label in range(0, len(normalized_prob)):
            loss_NML = compute_log_loss(normalized_prob, true_label)
            loss_NML_list.append(loss_NML)
        normalization_factor_list.append(normalization_factor)
        is_correct_list.append(predicted_label == true_label)
        test_sample_idx_list.append(keys)
    acc_NML = np.sum(is_correct_list) / len(is_correct_list)
    return loss_NML_list, normalization_factor_list, acc_NML, is_correct_list, test_sample_idx_list


def calculate_top_k_acc(results_dict, top_k):
    is_correct_nml_list = []
    is_correct_erm_list = []
    test_sample_idx_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, prob_org = extract_probabilities_list(sample_dict)

        top_k_labels = np.argsort(prob)[-top_k:][::-1]
        if true_label in top_k_labels:
            is_correct_nml_list.append(True)
        else:
            is_correct_nml_list.append(False)

        top_k_labels = np.argsort(prob_org)[-top_k:][::-1]
        if true_label in top_k_labels:
            is_correct_erm_list.append(True)
        else:
            is_correct_erm_list.append(False)

        test_sample_idx_list.append(keys)
    acc_top_k_nml = np.sum(is_correct_nml_list) / len(is_correct_nml_list)
    acc_top_k_erm = np.sum(is_correct_erm_list) / len(is_correct_erm_list)
    return acc_top_k_nml, acc_top_k_erm


def get_jinni_log_loss_from_dict(results_dict, is_random_labels=False):
    if is_random_labels == True:
        _, testloader, _ = create_cifar10_dataloaders('../data/', 1, 1)

    loss_jinni_list = []
    is_correct_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]

        # If was trained with random- extract the real label
        if is_random_labels == True:
            true_label = testloader.dataset.test_labels[int(keys)]
            sample_dict['true_label'] = true_label
        prob_jinni, true_label, predicted_jinni_label = extract_jinni_probabilities_list(sample_dict)
        loss_jinni = compute_log_loss(prob_jinni, true_label)
        loss_jinni_list.append(loss_jinni)
        is_correct_list.append(predicted_jinni_label == true_label)
    acc = np.sum(is_correct_list) / len(is_correct_list)
    return loss_jinni_list, acc, is_correct_list


def get_training_loss_list_from_dict(results_dict):
    training_loss_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]

        # Get training loss of the model which was trained with the true label
        if 'true_label' in sample_dict:
            true_label = sample_dict['true_label']
            train_loss = sample_dict[str(true_label)]['train_loss']
            training_loss_list.append(train_loss)

        else:
            # Iterate on trained labels per sample
            for trained_label in sample_dict:
                if trained_label.isdigit():
                    train_loss = sample_dict[trained_label]['train_loss']
                    training_loss_list.append(train_loss)

    return training_loss_list


def load_dict_from_file_list(files):
    result_dict = {}
    for file in files:
        with open(file) as data_file:
            result_dict.update(json.load(data_file))
    return result_dict


def load_results_to_df(files):
    results_dict = load_dict_from_file_list(files)

    log_nml_list, normalization_factor_list, acc, is_correct_nml_list, test_sample_idx = get_NML_log_loss_from_dict(
        results_dict)
    loss_erm_list, acc_erm, is_correct_erm_list = get_ERM_log_loss_from_dict(results_dict)
    loss_jinni_list, acc_Jinni, is_correct_jinni_list = get_jinni_log_loss_from_dict(results_dict)

    test_sample_idx = np.array(test_sample_idx).astype(int)

    dict_for_df = {'nml_loss': pd.Series(log_nml_list, index=test_sample_idx),
                   'erm_loss': pd.Series(loss_erm_list, index=test_sample_idx),
                   'jinni_loss': pd.Series(loss_jinni_list, index=test_sample_idx),
                   'log10_norm_factor': pd.Series(np.log10(normalization_factor_list), index=test_sample_idx),
                   'nml_is_correct': pd.Series(is_correct_nml_list, index=test_sample_idx),
                   'erm_is_correct': pd.Series(is_correct_erm_list, index=test_sample_idx),
                   'jinni_is_correct': pd.Series(is_correct_jinni_list, index=test_sample_idx)}
    result_df = pd.DataFrame(dict_for_df)

    # calc mean and std
    mean_loss_jinni, std_loss_jinni = result_df['jinni_loss'].mean(), result_df['jinni_loss'].std()
    mean_loss_nml, std_loss_nml = result_df['nml_loss'].mean(), result_df['nml_loss'].std()
    mean_loss_erm, std_loss_erm = result_df['erm_loss'].mean(), result_df['erm_loss'].std()

    # calc acc
    count = result_df.shape[0]
    acc_jinni = result_df['jinni_is_correct'][result_df['jinni_is_correct'] == True].count() / count
    acc_erm = result_df['erm_is_correct'][result_df['erm_is_correct'] == True].count() / count
    acc_nml = result_df['nml_is_correct'][result_df['nml_is_correct'] == True].count() / count

    statistics = {'jinni': pd.Series([acc_jinni, mean_loss_jinni, std_loss_jinni],
                                     index=['acc', 'mean loss', 'std loss']),
                  'nml': pd.Series([acc_nml, mean_loss_nml, std_loss_nml],
                                   index=['acc', 'mean loss', 'std loss']),
                  'erm': pd.Series([acc_erm, mean_loss_erm, std_loss_erm],
                                   index=['acc', 'mean loss', 'std loss'])}

    statistics_df = pd.DataFrame(statistics)

    return result_df, statistics_df


if __name__ == "__main__":
    # Example
    json_file_name = os.path.join('output', 'NML_results_20180815_135021', 'results_NML_20180815_135021.json')
    with open(json_file_name) as data_file:
        results_dict = json.load(data_file)
    # loss_jinni_list, acc = get_jinni_log_loss_from_dict(results_dict)
    loss_NML_list, normalization_factor_list, acc_NML, is_correct_list_NML_list, test_sample_idx_list = \
        get_NML_log_loss_of_the_series_from_dict(results_dict)
