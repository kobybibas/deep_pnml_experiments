import json
import os
import time

import numpy as np
import pandas as pd

from dataset_utilities import create_cifar10_dataloaders

# Import tesloader for random label experiment
_, testloader, _ = create_cifar10_dataloaders('../data/', 1, 1)


def extract_probabilities_list(evaluation_dict):
    # if the sample was trained with label 2, extract the prob to be 2 ...
    # return list of probabilities
    prob_all = []

    true_label = evaluation_dict['true_label'] if 'true_label' in evaluation_dict else None
    prob_org = np.array(evaluation_dict['original']['prob'])
    for trained_label in evaluation_dict:

        # One of the key is a string, ignore it
        if trained_label.isdigit():
            prob_on_trained = evaluation_dict[trained_label]['prob'][int(trained_label)]
            prob_all.append(prob_on_trained)
    predicted_label = np.argmax(prob_all) if len(prob_all) > 0 else None

    return np.array(prob_all), true_label, predicted_label, prob_org


def extract_jinni_probabilities_list(evaluation_dict):
    # Extract to probabilities of the model which was trained with the true label
    # return list of probabilities
    true_label = evaluation_dict['true_label']
    prob_jinni = np.array(evaluation_dict[str(true_label)]['prob'])
    predicted_jinni_label = np.argmax(prob_jinni)

    return prob_jinni, true_label, predicted_jinni_label


def execute_normalize_prob(prob_list):
    # Normalize the probabilities to be valid distribution
    # Return list of probabilities along with the normalization factor which was used.
    normalization_factor = np.sum(prob_list)
    normalized_prob = np.array(prob_list) / normalization_factor
    return normalized_prob, normalization_factor


def compute_log_loss(normalized_prob, true_label):
    # Compute the log loss
    return -np.log10(normalized_prob[true_label] + np.finfo(float).eps)


def calculate_top_k_acc(results_dict, top_k, prob_thresh=0.0):
    is_correct_nml_list = []
    is_correct_erm_list = []
    test_sample_idx_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, prob_org = extract_probabilities_list(sample_dict)
        normalized_prob, _ = execute_normalize_prob(prob)

        top_k_labels = np.argsort(normalized_prob)[-top_k:][::-1].astype(int)
        if true_label in top_k_labels and normalized_prob[true_label] > prob_thresh:
            is_correct_nml_list.append(True)
        else:
            is_correct_nml_list.append(False)

        top_k_labels = np.argsort(prob_org)[-top_k:][::-1].astype(int)
        if true_label in top_k_labels and prob_org[true_label] > prob_thresh:
            is_correct_erm_list.append(True)
        else:
            is_correct_erm_list.append(False)

        test_sample_idx_list.append(keys)
    acc_top_k_nml = np.sum(is_correct_nml_list) / len(is_correct_nml_list)
    acc_top_k_erm = np.sum(is_correct_erm_list) / len(is_correct_erm_list)

    return acc_top_k_nml, acc_top_k_erm


def load_dict_from_file_list(files):
    result_dict = {}
    for file in files:
        with open(file) as f:
            result_dict.update(json.load(f))
    return result_dict


def load_results_to_df(files, is_random_labels=False):
    results_dict = load_dict_from_file_list(files)

    nml_df = result_dict_to_nml_df(results_dict, is_random_labels=is_random_labels)
    erm_df = result_dict_to_erm_df(results_dict, is_random_labels=is_random_labels)
    jinni_df = result_dict_to_jinni_df(results_dict, is_random_labels=is_random_labels)

    statisic_nml_df = calc_statistic_from_df_single(nml_df).rename(columns={'statistics': 'nml'})
    statisic_erm_df = calc_statistic_from_df_single(erm_df).rename(columns={'statistics': 'erm'})
    statisic_jinni_df = calc_statistic_from_df_single(jinni_df).rename(columns={'jinni': 'erm'})

    # Add prefix before merge
    nml_df = nml_df.add_prefix('nml_')
    nml_df = nml_df.rename(columns={'nml_log10_norm_factor': 'log10_norm_factor'})
    erm_df = erm_df.add_prefix('erm_')
    jinni_df = jinni_df.add_prefix('jinni_')

    # Merge and return
    result_df = pd.concat([nml_df, erm_df, jinni_df], axis=1)
    statisic_df = pd.concat([statisic_nml_df, statisic_erm_df, statisic_jinni_df], axis=1)
    return result_df, statisic_df


def calc_statistic_from_df_single(result_df):
    mean_loss, std_loss = result_df['loss'].mean(), result_df['loss'].std()
    acc_jinni = result_df['is_correct'].sum() / result_df.shape[0]
    statistics_df = pd.DataFrame(
        {'statistics': pd.Series([acc_jinni, mean_loss, std_loss], index=['acc', 'mean loss', 'std loss'])})
    return statistics_df


def result_dict_to_nml_df(results_dict, is_random_labels=False):
    # Initialize col of df
    df_col = [str(x) for x in range(10)] + ['true_label', 'loss', 'log10_norm_factor']
    nml_dict = {}
    for col in df_col:
        nml_dict[col] = []
    loc = []

    # Iterate on test samples
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob_all, true_label, predicted_label, _ = extract_probabilities_list(sample_dict)
        prob_nml, norm_factor = execute_normalize_prob(prob_all)
        true_label = true_label if is_random_labels is False else testloader.dataset.test_labels[int(keys)]
        nml_dict['true_label'].append(true_label)
        nml_dict['loss'].append(compute_log_loss(prob_nml, true_label))
        for prob_label, prob_single in enumerate(prob_nml):
            nml_dict[str(prob_label)].append(prob_single)
        nml_dict['log10_norm_factor'].append(np.log10(norm_factor))
        loc.append(int(keys))

    # Create df
    nml_df = pd.DataFrame(nml_dict, index=loc)

    # Add more columns
    is_correct = np.array(nml_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        nml_df['true_label']).astype(int)
    nml_df['is_correct'] = is_correct
    return nml_df


def result_dict_to_erm_df(results_dict, is_random_labels=False):
    # Initialize columns to df
    df_col = [str(x) for x in range(10)] + ['true_label', 'loss']
    erm_dict = {}
    for col in df_col:
        erm_dict[col] = []
    loc = []

    # Iterate on keys
    for keys in results_dict:
        # extract probability of test sample
        sample_dict = results_dict[keys]
        _, true_label, _, prob_org = extract_probabilities_list(sample_dict)
        true_label = true_label if is_random_labels is False else testloader.dataset.test_labels[int(keys)]
        erm_dict['true_label'].append(true_label)
        erm_dict['loss'].append(compute_log_loss(prob_org, true_label))
        for prob_label, prob_single in enumerate(prob_org):
            erm_dict[str(prob_label)].append(prob_single)
        loc.append(int(keys))

    # Create df
    erm_df = pd.DataFrame(erm_dict, index=loc)

    # Add more columns
    is_correct = np.array(erm_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        erm_df['true_label']).astype(int)
    erm_df['is_correct'] = is_correct
    return erm_df


def result_dict_to_jinni_df(results_dict, is_random_labels=False):
    # Initialize columns to df
    df_col = [str(x) for x in range(10)] + ['true_label', 'loss']
    jinni_dict = {}
    for col in df_col:
        jinni_dict[col] = []
    loc = []

    # Iterate on keys
    for keys in results_dict:
        # extract probability of test sample
        sample_dict = results_dict[keys]
        prob_jinni, true_label, predicted_jinni_label = extract_jinni_probabilities_list(sample_dict)
        true_label = true_label if is_random_labels is False else testloader.dataset.test_labels[int(keys)]
        jinni_dict['true_label'].append(true_label)
        jinni_dict['loss'].append(compute_log_loss(prob_jinni, true_label))
        for prob_label, prob_single in enumerate(prob_jinni):
            jinni_dict[str(prob_label)].append(prob_single)
        loc.append(int(keys))

    # Create df
    jinni_df = pd.DataFrame(jinni_dict, index=loc)

    # Add more columns
    is_correct = np.array(jinni_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        jinni_df['true_label']).astype(int)
    jinni_df['is_correct'] = is_correct

    return jinni_df


if __name__ == "__main__":
    # Example
    json_file_name = os.path.join('.', 'results_example.json')
    with open(json_file_name) as data_file:
        results_dict_sample = json.load(data_file)
    nml_df_sample = result_dict_to_nml_df(results_dict_sample)

    tic = time.time()
    result_df_sample, statistics_df_sample = load_results_to_df([json_file_name])
    # print('load_results_to_df: {0:.2f} [s]'.format(time.time() - tic))
    tic = time.time()
    nml_df = result_dict_to_nml_df(results_dict_sample)
    print('result_dict_to_nml_df: {0:.2f} [s]'.format(time.time() - tic))
    tic = time.time()
    statisic = calc_statistic_from_df_single(nml_df)
    print('calc_statistic_from_df_single: {0:.2f} [s]'.format(time.time() - tic))
    a = 1
