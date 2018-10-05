import json
import os

import numpy as np


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


def get_ERM_log_loss_from_dict(results_dict):
    loss_ERM_list = []
    is_correct_ERM_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        _, true_label, _, prob_org = extract_probabilities_list(sample_dict)
        loss_ERM = compute_log_loss(prob_org, true_label)
        loss_ERM_list.append(loss_ERM)
        is_correct_ERM_list.append(np.argmax(prob_org) == true_label)

    acc_ERM = np.sum(is_correct_ERM_list) / len(is_correct_ERM_list)
    return loss_ERM_list, acc_ERM, is_correct_ERM_list


def get_NML_log_loss_from_dict(results_dict):
    loss_NML_list = []
    is_correct_list_NML_list = []
    normalization_factor_list = []
    test_sample_idx_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, _ = extract_probabilities_list(sample_dict)
        normalized_prob, normalization_factor = execute_normalize_prob(prob)

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


def get_jinni_log_loss_from_dict(results_dict):
    loss_jinni_list = []
    is_correct_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob_jinni, true_label, predicted_jinni_label = extract_jinni_probabilities_list(sample_dict)
        loss_jinni = compute_log_loss(prob_jinni, true_label)
        loss_jinni_list.append(loss_jinni)
        is_correct_list.append(predicted_jinni_label == true_label)
    acc = np.sum(is_correct_list) / len(is_correct_list)
    return loss_jinni_list, acc


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


if __name__ == "__main__":
    # Example
    json_file_name = os.path.join('output', 'NML_results_20180815_135021', 'results_NML_20180815_135021.json')
    with open(json_file_name) as data_file:
        results_dict = json.load(data_file)
    # loss_jinni_list, acc = get_jinni_log_loss_from_dict(results_dict)
    loss_NML_list, normalization_factor_list, acc_NML, is_correct_list_NML_list, test_sample_idx_list = \
        get_NML_log_loss_of_the_series_from_dict(results_dict)
