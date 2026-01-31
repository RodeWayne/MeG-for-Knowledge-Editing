import argparse
import time
import torch
from itertools import chain
import numpy as np
from util_para_generate import *
from datetime import datetime
from classifier_test import  test

def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  #
    start_time = time.time()
    device, edit_model, tokenizer, para_factory, zsre_datas_range,all_ids, xparas = initDeviceModelDataAndParadit(args)
    if args.type == "paradit":
        batch_size = args.batch_size
        num_batches = (len(zsre_datas_range) + batch_size - 1) // batch_size
    all_id_para_l2={}
    all_is_rel_kns = []  # store is_rel_kns for all batches
    all_true_count = 0
    all_false_count = 0
    for batch_idx in range(num_batches):
        batch_data = zsre_datas_range[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        if args.data_type == "cf":
            inputs = [SHORT_ANSWER_PROMPT[args.model_para_type].format(
                data["requested_rewrite"]["prompt"].format(data["requested_rewrite"]["subject"])) for data in
                      batch_data]
            querys = [
                data["requested_rewrite"]["prompt"].format(data["requested_rewrite"]["subject"]) for data in
                      batch_data]
        else:
            if args.temp == "newzsre":
                inputs = [SHORT_ANSWER_PROMPT[args.model_para_type].format(data["src"]) for data in batch_data]
                querys = [data["src"] for data in batch_data]
            else:
                inputs = [SHORT_ANSWER_PROMPT[args.model_para_type].format(data["query"]) for data in batch_data]
                querys = [data["query"] for data in batch_data]
        # Determine if the knowledge is relevant
        if args.is_rel :
            is_rel_kns = test(args.bmodel_train_state_path, args.data_type, args.model_para_type, args.fi, querys,
                              device)
        else:
            is_rel_kns = torch.ones(len(querys), dtype=torch.bool).to(device)
        true_count = sum(is_rel_kns)
        false_count = len(is_rel_kns) - true_count
        print(f"True: {true_count}, False: {false_count}")
        all_true_count += true_count
        all_false_count += false_count
        all_is_rel_kns.extend(is_rel_kns)
        # generate model para
        with torch.no_grad():
            paras = para_factory.generate_paral(is_rel_kns,inputs, args.gtype, srcmodel=edit_model, srctokenizer=tokenizer,
                                            layer=args.layer)
        for data, para in zip(batch_data, paras):
            try:
                index = all_ids.index(data["id"])
                x_orig = xparas[index]
                x_orig = x_orig.to(device)
                l2_norms = torch.norm(x_orig - para, p=2, dim=0)
                l2_norms_str = str(l2_norms.tolist())
            except ValueError:
                index = -1
                l2_norms_str="-1"
            all_id_para_l2[data["id"]] = [l2_norms_str,para]

    # ========================
    mid_time = time.time()
    step = 0
    accuracys_tf=[]

    for index, data in enumerate(zsre_datas_range):
        with torch.no_grad():
            step = step + 1
            data['l2_norms'] = all_id_para_l2[data["id"]][0]
            edit_model = get_edit_model(args.model_para_type, edit_model, all_id_para_l2[data["id"]][1], args.layer)
            if args.type == "paradit":
                if args.data_type == "cf":
                    input = data["requested_rewrite"]["prompt"].format(data["requested_rewrite"]["subject"])
                    input = SHORT_ANSWER_PROMPT[args.model_para_type].format(input)
                else:
                    if args.temp == "newzsre":
                        input = SHORT_ANSWER_PROMPT[args.model_para_type].format(data["src"])
                    else:
                        input = SHORT_ANSWER_PROMPT[args.model_para_type].format(data["query"])
            # prefix auto-regressive eval
            all_self_recursive = self_recursive(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model,
                                                all_false_count,
                                                index, input, mid_time, start_time, step, tokenizer,
                                                all_true_count)
            # teacher forcing eval
            all_teach_forcing = teach_forcing(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model,
                                              all_false_count,
                                              index, input, mid_time, start_time, step, tokenizer, all_true_count,
                                              accuracys_tf)
    # Modification for parallelism
    if hasattr(args, 'is_parallel') and args.is_parallel:
        return all_self_recursive["right"],all_self_recursive["step"], accuracys_tf
    else:
        return all_self_recursive["accuracy"],all_teach_forcing["accuracy_mean"],all_teach_forcing["accuracy_std"]
    # ========================
def teach_forcing(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model, all_false_count, index, input,
                   mid_time, start_time, step, tokenizer, all_true_count,accuracys_tf):
    prob_prompts = [[input]]
    inp_prompts_og = list(chain(*prob_prompts))

    label = getLabel(args, data)

    target_tok = tokenizer(label)["input_ids"]
    inp_prompts = [
        el + tokenizer.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tokenizer.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]
    stuff_probs = test_batch_prediction_acc(edit_model, tokenizer, inp_prompts, inp_targets, device)
    data['generate'] = stuff_probs
    data['isFTSuccess'] = data["id"] in all_ids
    data['is_rel_kn'] = all_is_rel_kns[index].item()
    accuracy = np.mean(stuff_probs)
    accuracys_tf.append(accuracy)
    if args.type == "memit":
        file_path = os.path.join(
            f"{args.result_path}/result_tf",
            f"validate_sr_{args.data_range}_{args.model_state_dir.split('/')[-2].split('_')[3]}_{current_time}.json")

    elif args.type == "paradit":
        file_path = os.path.join(
            f"{args.result_path}/result_tf_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
            f"validate_sr_{args.data_range}_{args.model_state_dir.split('_')[-1].split('.')[0]}_{current_time}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []
    existing_data.append(data)
    # insert updated record
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    all = {}
    all["modeldir"] = args.model_state_dir
    if args.type == "paradit":
        all["train_epoch"] = args.model_state_dir.split('_')[-1].split('.')[0]
    all["nums_para"] = args.data_range
    all["step"] = step
    all["accuracy_mean"] = np.mean(accuracys_tf)
    all["accuracy_std"] = np.std(accuracys_tf)
    all["eval_time"] = current_time
    all["true_case"] = all_true_count.item()
    all["false_case"] = all_false_count.item()
    elapsed_time_mid = format_elapsed_time(mid_time - start_time)
    all["mid_cost_time"] = elapsed_time_mid
    end_time = time.time()
    elapsed_time = format_elapsed_time(end_time - start_time)
    all["end_cost_time"] = elapsed_time
    if args.type == "memit":
        file_path = os.path.join(f"{args.result_path}/result_tf", f'all_sr.json')
    elif args.type == "paradit":
        file_path = os.path.join(
            f"{args.result_path}/result_tf_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}", f'all_sr.json')
    existing_data = []
    existing_data.append(all)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    return all

def self_recursive(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model, all_false_count, index, input,
                   mid_time, start_time, step, tokenizer, all_true_count):
    input = tokenizer(input, return_tensors="pt", return_attention_mask=False).to(device)
    if args.model_para_type == "phi2":
        outputs = edit_model.generate(**input, max_length=200)
    elif args.model_para_type == "gptj":
        outputs = edit_model.generate(**input, max_length=input['input_ids'].shape[1] + 20)
    elif args.model_para_type == "llama3":
        outputs = edit_model.generate(**input, max_length=input['input_ids'].shape[1] + 20)
    outputs = outputs[:, input['input_ids'].shape[1]:]
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    label = getLabel(args, data)
    if args.model_para_type == "gptj" or  args.model_para_type == "llama3":
        stop_sequence = "\\"
        prediction = prediction.split(stop_sequence)[0]
        stop_sequence = "\n"
        prediction = prediction.split(stop_sequence)[0]
    prediction = adjust_dots(prediction.rstrip())
    if args.type == "memit":
        if prediction.lower().startswith(label.lower()):
            data['generate'] = True
        else:
            data['generate'] = prediction
    elif args.type == "paradit":
        if label.lower() == prediction.lower():
            data['generate'] = True
        else:
            data['generate'] = prediction
    data['isFTSuccess'] = data["id"] in all_ids
    data['is_rel_kn'] = all_is_rel_kns[index].item()
    if args.type == "memit":
        file_path = os.path.join(f"{args.result_path}/result",
                                 f"validate_sr_{args.model_state_dir.split('/')[-2].split('_')[3]}_{current_time}.json")
    elif args.type == "paradit":
        file_path = os.path.join(
            f"{args.result_path}/result_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
            f"validate_sr_{args.data_range}_{args.model_state_dir.split('_')[-1].split('.')[0]}_{current_time}.json")

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []
    existing_data.append(data)
    # insert updated record
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    rights = 0
    all = {}
    for i, d in enumerate(existing_data):
        if d['generate'] == True:
            rights += 1
    all["modeldir"] = args.model_state_dir
    if args.type == "paradit":
        all["train_epoch"] = args.model_state_dir.split('_')[-1].split('.')[0]
    all["nums_para"] = args.data_range
    all["right"] = rights
    all["step"] = step
    all["accuracy"] = rights / step
    all["eval_time"] = current_time
    all["true_case"] = all_true_count.item()
    all["false_case"] = all_false_count.item()
    elapsed_time_mid = format_elapsed_time(mid_time - start_time)
    all["mid_cost_time"] = elapsed_time_mid
    end_time = time.time()
    elapsed_time = format_elapsed_time(end_time - start_time)
    all["end_cost_time"] = elapsed_time
    if args.type == "memit":
        file_path = os.path.join(f"{args.result_path}/result", f'all_sr.json')
    elif args.type == "paradit":
        file_path = os.path.join(
            f"{args.result_path}/result_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}", f'all_sr.json')
    existing_data = []
    existing_data.append(all)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    return all


