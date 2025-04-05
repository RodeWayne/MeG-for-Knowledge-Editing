import evaluate_lr_zsre_self_and_tf_parallel
import evaluate_gr_self_and_tf_parallel
import evaluate_sr_self_and_tf_parallel
from getFilePath import *
# import test2
import yaml
from argparse import Namespace
from datetime import datetime
import sys
import numpy as np
import time
import os
import argparse

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()  # Linux 和 macOS 上生效11

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
def setup_logging(log_file):
    """
    Redirect print statements to both a log file and the console.
    """
    class Logger:
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)  # Print to console
            self.log.write(message)      # Write to file

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    sys.stdout = Logger(log_file)
def log_result(file_path, message):
    """
    Append a message to the result log file.
    """
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
def main():
    # Set up argument parser for config selection only
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_para_type", type=str, default='gptj')
    parser.add_argument("--data_type", type=str, default='zsre')
    args = parser.parse_args()
    # load config
    config = load_config(config_path=f"evaluate/hparams/run_all_evaluation_config_paradit_{args.model_para_type}_{args.data_type}.yaml")
    config = Namespace(**config)
    config.result_path=config.result_path+datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
    # init logging
    setup_logging(config.result_path + "log.txt")
    print("==================print setting start")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    result_log = config.result_path + "result.log"
    print("==================print setting end")

    log_result(result_log, f"model_state_dir：{config.model_state_dir}")

    try:
        accuracy_sr,mean_accuracy_sr_tf, std_accuracy_sr_tf = evaluate_sr_self_and_tf_parallel.main(config)
        log_result(result_log, f"evaluate_sr: {accuracy_sr:.4f}")
        log_result(result_log, f"evaluate_sr_tf: mean={mean_accuracy_sr_tf:.4f}, std={std_accuracy_sr_tf:.4f}")

        accuracy_gr,mean_accuracy_gr_tf, std_accuracy_gr_tf = evaluate_gr_self_and_tf_parallel.main(config)
        log_result(result_log, f"evaluate_gr: {accuracy_gr:.4f}")
        log_result(result_log, f"evaluate_gr_tf: mean={mean_accuracy_gr_tf:.4f}, std={std_accuracy_gr_tf:.4f}")


        accuracy_lr_zsre, mean_accuracy_lr_zsre_tf, std_accuracy_lr_zsre_tf = evaluate_lr_zsre_self_and_tf_parallel.main(config)
        log_result(result_log, f"evaluate_lr_zsre: {accuracy_lr_zsre:.4f}")
        log_result(result_log,f"evaluate_lr_zsre_tf: mean={mean_accuracy_lr_zsre_tf:.4f}, std={std_accuracy_lr_zsre_tf:.4f}")

        nums_para=config.data_range
        if config.type=="memit":
            train_epoch="memit"
        elif config.type=="paradit":
            train_epoch=config.model_state_dir.split('_')[-1].split('.')[0]
        # print title
        log_result(result_log,
                   f"nums_para,train_epoch,sr,gr,lr_zsre,\"sr_tf (mean,std)\",\"gr_tf (mean,std)\",\"lr_zsre_tf (mean,std)\"")
        # print result
        log_result(
            result_log,
            f"{nums_para},{train_epoch},{accuracy_sr:.4f},{accuracy_gr:.4f},{accuracy_lr_zsre:.4f},"
            f"\"({mean_accuracy_sr_tf:.4f},{std_accuracy_sr_tf:.4f})\",\"({mean_accuracy_gr_tf:.4f},{std_accuracy_gr_tf:.4f})\","
            f"\"({mean_accuracy_lr_zsre_tf:.4f},{std_accuracy_lr_zsre_tf:.4f})\""
        )


    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()