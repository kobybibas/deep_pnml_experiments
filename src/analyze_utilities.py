import matplotlib.pyplot as plt
import numpy as np
import json
import os


def extract_probabilities_list(evaluation_dict):
    # if the sample was trained with label 2, extract the prob to be 2 ...
    # return list of probabilities
    prob_all = []
    true_label = evaluation_dict['true_label']
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
    # Normalize the probabilities to be valid disterbution
    # Reuturn list of probabilities along with the normalization factor which was used.
    np.sum(prob_list)
    normalization_factor = np.sum(prob_list)
    normalized_prob = np.array(prob_list)/normalization_factor
    return normalized_prob, normalization_factor


def compute_log_loss(normalized_prob, true_label):
    # Compute the log loss
    return -np.log10(normalized_prob[true_label])


def get_normalization_factor_from_dict(results_dict):
    normalization_factor_list = []
    is_correct_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, _ = extract_probabilities_list(sample_dict)
        normalized_prob, normalization_factor = execute_normalize_prob(prob)

        normalization_factor_list.append(normalization_factor)
        is_correct_list.append(np.argmax(normalized_prob) == true_label)
    return normalization_factor_list, is_correct_list


def get_ERM_log_loss_from_dict(results_dict):
    loss_ERM_list = []
    is_correct_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        _, true_label, _, prob_org = extract_probabilities_list(sample_dict)
        loss_ERM = compute_log_loss(prob_org, true_label)
        loss_ERM_list.append(loss_ERM)
        is_correct_list.append(np.argmax(prob_org) == true_label)

    acc = np.sum(is_correct_list)/len(is_correct_list)
    return loss_ERM_list, acc


def get_NML_log_loss_from_dict(results_dict):
    loss_NML_list = []
    is_correct_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, _ = extract_probabilities_list(sample_dict)
        normalized_prob, normalization_factor = execute_normalize_prob(prob)
        loss_NML = compute_log_loss(normalized_prob, true_label)
        loss_NML_list.append(loss_NML)
        is_correct_list.append(predicted_label == true_label)
    acc = np.sum(is_correct_list) / len(is_correct_list)
    return loss_NML_list, acc


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


if __name__ == "__main__":

    # Example
    json_file_name = os.path.join('output', 'NML_results_20180810_140920', 'results_NML_20180810_140920.json')
    with open(json_file_name) as data_file:
        results_dict = json.load(data_file)
    loss_jinni_list, acc = get_jinni_log_loss_from_dict(results_dict)


