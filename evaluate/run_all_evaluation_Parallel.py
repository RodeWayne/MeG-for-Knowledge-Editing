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
import json
import setproctitle

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
def log_result_json(file_path, data_name, data):
    """
    Append the data to the JSON file. If data_name already exists, add the data to the corresponding list.
    """
    try:
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []  # if the file does not exist, create a new empty list
    except json.JSONDecodeError:
        existing_data = []
        print("not a JSON file")

    # chech if data_name exists
    name_exists = False
    for item in existing_data:
        if data_name in item:  # check if data_name exists as a key
            name_exists = True
            # add new data into existing list
            if isinstance(data, list):
                item[data_name].extend(data)  # if the new data is a list, extend the item list
            else:
                item[data_name].append(data)  # if the new data is a single vaule, append the item list
            break

    # create a new_item list for the data_name
    if not name_exists:
        new_item = {
            data_name: [data] if not isinstance(data, list) else data,
        }
        existing_data.append(new_item)

    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)
def main():
    # Set up argument parser for config selection only
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_para_type", type=str, default='gptj')
    parser.add_argument("--data_type", type=str, default='zsre')
    # Modification for parallelism
    parser.add_argument("--is_parallel", type=bool, default=False, help="Whether to use parallel evaluation.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of the data subset.")
    parser.add_argument("--end_index", type=int, default=0, help="End index of the data subset.")
    parser.add_argument("--data_range", type=int, default=10000, help="All data range")
    parser.add_argument("--time_stamp", type=str, default=None, help="Time stamp of the experiment.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id to use.")
    args = parser.parse_args()
    # load config
    config = load_config(config_path=f"evaluate/hparams/run_all_evaluation_config_paradit_{args.model_para_type}_{args.data_type}.yaml")
    if args.is_parallel:
        config["start_index"] = args.start_index
        config["end_index"] = args.end_index
        config["is_parallel"] = True
        config["gpu"] = args.gpu_id
        config["fi"] = args.data_range
        # set process name
        process_name = f"Validate_sr_gr_lr_{args.time_stamp}_{args.start_index}-{args.end_index}"
        setproctitle.setproctitle(process_name)
    config = Namespace(**config)
    if args.is_parallel:
        # Need a general path given by .sh script
        config.result_path = config.result_path + args.time_stamp + "/"
    else:
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
        if hasattr(args, 'is_parallel') and args.is_parallel:
            right_sr,step_sr,all_accuracys_sr_tf = evaluate_sr_self_and_tf_parallel.main(config)
            result_path_json = config.result_path + "evaluate_sr_parallel.json"
            log_result_json(result_path_json, "evaluate_sr_step", step_sr)
            log_result_json(result_path_json, "evaluate_sr_rights", right_sr)
            result_path_json = config.result_path + "evaluate_sr_tf_parallel.json"
            log_result_json(result_path_json, "evaluate_sr_tf_accuracys", all_accuracys_sr_tf)
            
            right_gr,step_gr,all_accuracys_gr_tf = evaluate_gr_self_and_tf_parallel.main(config)
            result_path_json = config.result_path + "evaluate_gr_parallel.json"
            log_result_json(result_path_json, "evaluate_gr_step", step_gr)
            log_result_json(result_path_json, "evaluate_gr_rights", right_gr)
            result_path_json = config.result_path + "evaluate_gr_tf_parallel.json"
            log_result_json(result_path_json, "evaluate_gr_tf_accuracys", all_accuracys_gr_tf)

            right_lr,step_lr,all_accuracys_lr_tf = evaluate_lr_zsre_self_and_tf_parallel.main(config)
            result_path_json = config.result_path + "evaluate_lr_parallel.json"
            log_result_json(result_path_json, "evaluate_lr_step", step_lr)
            log_result_json(result_path_json, "evaluate_lr_rights", right_lr)
            result_path_json = config.result_path + "evaluate_lr_tf_parallel.json"
            log_result_json(result_path_json, "evaluate_lr_tf_accuracys", all_accuracys_lr_tf)

        else:
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