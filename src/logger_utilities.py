import logging
import json
import os
import sys
import time
import pathlib


class Logger:
    def __init__(self, experiment_type, output_root):
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

        self.unique_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_root, '%s_results_%s' %
                                          (experiment_type, self.unique_time))
        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.define_log_file(os.path.join(self.output_folder, 'log_%s_%s.log' %
                                          (experiment_type, self.unique_time)))
        self.define_json_output(os.path.join(self.output_folder, 'results_%s_%s.json' %
                                             (experiment_type, self.unique_time)))

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

    def add_entry_to_results_dict(self, test_idx_sample, true_label, prob_key_str, prob,
                                  train_loss, test_loss, prob_org):

        if str(test_idx_sample) not in self.results_dict:
            self.results_dict[str(test_idx_sample)] = {}
            self.results_dict[str(test_idx_sample)]['true_label'] = true_label

        self.results_dict[str(test_idx_sample)][prob_key_str] = {}
        self.results_dict[str(test_idx_sample)][prob_key_str]['prob'] = prob
        self.results_dict[str(test_idx_sample)][prob_key_str]['train_loss'] = train_loss
        self.results_dict[str(test_idx_sample)][prob_key_str]['test_loss'] = test_loss

        self.results_dict[str(test_idx_sample)]['original'] = {}
        self.results_dict[str(test_idx_sample)]['original']['prob'] = prob_org



