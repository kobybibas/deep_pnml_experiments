import logging
import json
import os
import sys

class Logger:
    def __init__(self):
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.logger = logger
        self.json_file_name = None
        self.results_dict = {}

    def define_log_file(self, log_file_name):
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def define_json_output(self, json_file_name):
        self.json_file_name = json_file_name

    def save_json_file(self):
        with open(self.json_file_name, 'w') as outfile:
            json.dump(self.results_dict,
                      outfile,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)

    def add_entry_to_results_dict(self, test_idx_sample, true_label, prob_key_str, prob):

        if str(test_idx_sample) not in self.results_dict:
            self.results_dict[str(test_idx_sample)] = {}
            self.results_dict[str(test_idx_sample)]['true_label'] = true_label

        self.results_dict[str(test_idx_sample)][prob_key_str] = prob



