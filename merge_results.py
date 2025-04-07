import argparse
import json
import os
import numpy as np


def load_json_data(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# 沿用了本来的函数
def log_result(file_path, message):
    with open(file_path, "a") as log_file:  # 打开结果日志文件，使用追加模式
        log_file.write(message + "\n")  # 写入消息并换行


def main(args):
    result_dir = "result_evaluation/" + args.time_stamp
    result_log = os.path.join(result_dir, "result.log")

    # --- load all data ---
    gr_data = load_json_data(os.path.join(result_dir, "evaluate_gr_parallel.json"))
    sr_data = load_json_data(os.path.join(result_dir, "evaluate_sr_parallel.json"))
    lr_data = load_json_data(os.path.join(result_dir, "evaluate_lr_parallel.json"))
    sr_tf_data = load_json_data(os.path.join(result_dir, "evaluate_sr_tf_parallel.json"))
    gr_tf_data = load_json_data(os.path.join(result_dir, "evaluate_gr_tf_parallel.json"))
    lr_tf_data = load_json_data(os.path.join(result_dir, "evaluate_lr_tf_parallel.json"))

    nums_para = args.data_range
    train_epoch = None
    model_type = None

    log_file_path = os.path.join(result_dir, "log.txt")  # config info is in the log

    # Extract epoch and model type from logs
    try:
        with open(log_file_path, "r") as file:
            for line in file:
                if "model_state_dir:" in line:
                    model_state_dir = line.split(":", 1)[1].strip()
                    if "memit" in model_state_dir.lower():
                        train_epoch = "memit"
                        model_type = "memit"
                    else:
                        train_epoch = model_state_dir.split("_")[-1].split(".")[0]
                        model_type = "paradit"
                    break 
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_file_path}")

    # --- sum up rights and steps ---
    def merge_rights_steps(data_list, rights_key, steps_key):
        total_rights = 0
        total_steps = 0
        for data in data_list:
            for entry in data:  
                if rights_key in entry:
                    total_rights += sum(data[entry])
                elif steps_key in entry:
                    total_steps += sum(data[entry])
        return total_rights, total_steps

    total_rights_gr, total_steps_gr = merge_rights_steps(gr_data, "evaluate_gr_rights", "evaluate_gr_step")
    total_rights_sr, total_steps_sr = merge_rights_steps(sr_data, "evaluate_sr_rights", "evaluate_sr_step")
    total_rights_lr, total_steps_lr = merge_rights_steps(
        lr_data, "evaluate_lr_rights", "evaluate_lr_step"
    )

    accuracy_gr = total_rights_gr / total_steps_gr
    accuracy_sr = total_rights_sr / total_steps_sr
    accuracy_lr = total_rights_lr / total_steps_lr

    def calculate_mean_std(data_list, accuracy_key):
        for data in data_list:
            for entry in data:
                if accuracy_key in entry:
                    return (np.mean(data[entry]), np.std(data[entry])) if data[entry] else (0.0, 0.0)

    mean_accuracy_sr_tf, std_accuracy_sr_tf = calculate_mean_std(sr_tf_data, "evaluate_sr_tf_accuracys")
    mean_accuracy_gr_tf, std_accuracy_gr_tf = calculate_mean_std(gr_tf_data, "evaluate_gr_tf_accuracys")
    mean_accuracy_lr_tf, std_accuracy_lr_tf = calculate_mean_std(
        lr_tf_data, "evaluate_lr_tf_accuracys"
    )

    log_result(result_log, f"evaluate_sr: {accuracy_sr:.4f}")
    log_result(result_log, f"evaluate_sr_tf: mean={mean_accuracy_sr_tf:.4f}, std={std_accuracy_sr_tf:.4f}")
    log_result(result_log, f"evaluate_gr: {accuracy_gr:.4f}")
    log_result(result_log, f"evaluate_gr_tf: mean={mean_accuracy_gr_tf:.4f}, std={std_accuracy_gr_tf:.4f}")
    log_result(result_log, f"evaluate_lr: {accuracy_lr:.4f}")
    log_result(
        result_log, f"evaluate_lr_tf: mean={mean_accuracy_lr_tf:.4f}, std={std_accuracy_lr_tf:.4f}"
    )
    log_result(
        result_log,
        f'nums_para,train_epoch,sr,gr,lr,"sr_tf (mean,std)","gr_tf (mean,std)","lr_tf (mean,std)"',
    ) 
    
    log_result(
        result_log,
        f"{nums_para},{train_epoch},{accuracy_sr:.4f},{accuracy_gr:.4f},{accuracy_lr:.4f},"
        f'"({mean_accuracy_sr_tf:.4f},{std_accuracy_sr_tf:.4f})","({mean_accuracy_gr_tf:.4f},{std_accuracy_gr_tf:.4f})",'
        f'"({mean_accuracy_lr_tf:.4f},{std_accuracy_lr_tf:.4f})"',
    )  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_stamp", type=str, required=True, help="uesd to identify the directory of results")
    parser.add_argument("--data_range", type=int, required=True)
    args = parser.parse_args()
    main(args)
