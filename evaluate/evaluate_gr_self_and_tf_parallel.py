import argparse
from util_para_generate import *
from datetime import datetime
import numpy as np
import time
from classifier_test import  test
from itertools import chain
import numpy as np
def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    device, edit_model, tokenizer, para_factory, zsre_datas_range, all_ids, xparas = initDeviceModelDataAndParadit(args)
    if args.type == "paradit":
        phrases=[]
        all_is_rel_kns = []  # store is_rel_kns for all batches
        for index, data in enumerate(zsre_datas_range):
            res_rephrase = {}
            if args.data_type == "cf":
                rephrases=data['paraphrase_prompts']
            else:
                rephrases=data['rephrase'] # 'paraphrase_prompts'
            for index, rephrase in enumerate(rephrases):
                if(index>0):continue
                phrases.append([data["id"],rephrase])
        # 2. Generate sentences and corresponding paras
        batch_size = args.batch_size
        num_batches = (len(phrases) + batch_size - 1) // batch_size
        all_phrase_para={}
        all_true_count = 0
        all_false_count = 0
        for batch_idx in range(num_batches):
            batch_data = phrases[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            inputs = [SHORT_ANSWER_PROMPT[args.model_para_type].format(data[1]) for data in batch_data]
            querys=[data[1] for data in batch_data]
            # Determine if the knowledge is relevant
            if args.is_rel:
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
            # Generate parameters
            with torch.no_grad():
                paras = para_factory.generate_paral(is_rel_kns, inputs, args.gtype, srcmodel=edit_model,
                                                    srctokenizer=tokenizer,
                                                    layer=args.layer)
            for data,input,para in zip(batch_data,inputs, paras):
                # Test L2 norm
                try:
                    index = all_ids.index(data[0])
                    x_orig = xparas[index]
                    x_orig = x_orig.to(device)
                    l2_norms = torch.norm(x_orig - para, p=2, dim=0)
                except ValueError:
                    index = -1
                    l2_norms = torch.tensor([float("-1")])
                all_phrase_para[input] = [l2_norms, para]

    mid_time = time.time()
    # prefix auto-regressive eval
    all_self_recursive = self_recursive(all_false_count, all_ids, all_is_rel_kns, all_phrase_para, all_true_count, args, current_time,
                         device, edit_model, input, mid_time, para, start_time, tokenizer, zsre_datas_range)
    # teacher forcing eval
    all_teach_forcing = teach_forcing(all_false_count, all_ids, all_is_rel_kns, all_phrase_para, all_true_count, args,
                                        current_time,
                                        device, edit_model, input, mid_time, para, start_time, tokenizer,
                                        zsre_datas_range)

    # Modification for parallelism
    if hasattr(args, 'is_parallel') and args.is_parallel:
        return all_self_recursive["right"],all_self_recursive["step"], all_teach_forcing
    else:
        return all_self_recursive["accuracy"],all_teach_forcing["accuracy_mean"],all_teach_forcing["accuracy_std"]

def teach_forcing(all_false_count, all_ids, all_is_rel_kns, all_phrase_para, all_true_count, args, current_time,
                   device, edit_model, input, mid_time, para, start_time, tokenizer, zsre_datas_range):
    step = 0
    accuracys = []
    for index0, data in enumerate(zsre_datas_range):
        res_rephrase = {}
        if args.data_type == "cf":
            rephrases = data['paraphrase_prompts']
        else:
            rephrases = data['rephrase']  # 'paraphrase_prompts'
        for index, rephrase in enumerate(rephrases):
            if (index > 0): continue
            with torch.no_grad():
                step = step + 1
                if args.type == "memit":
                    input = rephrase
                elif args.type == "paradit":
                    input = SHORT_ANSWER_PROMPT[args.model_para_type].format(rephrase)
                l2_norms = np.array([])
                if args.type == "paradit":
                    para = all_phrase_para[input][1]
                    l2_norms = all_phrase_para[input][0]
                    # edit model
                    edit_model = get_edit_model(args.model_para_type, edit_model, para, args.layer)
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
                res_rephrase[rephrase] = {"l2_norms": str(l2_norms.tolist()), "generate": stuff_probs}
                accuracy = np.mean(stuff_probs)
                accuracys.append(accuracy)

        data['res_rephrase'] = res_rephrase
        data['isFTSuccess'] = data["id"] in all_ids
        data['is_rel_kn'] = all_is_rel_kns[index0].item()

        if args.type == "memit":
            file_path = os.path.join(
                f"{args.result_path}/result_tf",
                f"validate_gr_{args.data_range}_{args.model_state_dir.split('/')[-2].split('_')[3]}_{current_time}.json")

        elif args.type == "paradit":
            file_path = os.path.join(
                f"{args.result_path}/result_tf_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
                f"validate_gr_{args.data_range}_{args.model_state_dir.split('_')[-1].split('.')[0]}_{current_time}.json")
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
        all["accuracy_mean"] = np.mean(accuracys)
        all["accuracy_std"] = np.std(accuracys)
        all["eval_time"] = current_time
        all["true_case"] = all_true_count.item()
        all["false_case"] = all_false_count.item()
        elapsed_time_mid = format_elapsed_time(mid_time - start_time)
        all["mid_cost_time"] = elapsed_time_mid
        end_time = time.time()
        elapsed_time = format_elapsed_time(end_time - start_time)
        all["end_cost_time"] = elapsed_time
        if args.type == "memit":
            file_path = os.path.join(f"{args.result_path}/result_tf", f'all_gr.json')
        elif args.type == "paradit":
            file_path = os.path.join(
                f"{args.result_path}/result_tf_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
                f'all_gr.json')
        existing_data = []
        existing_data.append(all)
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
    # Modification for parallelism
    if hasattr(args, 'is_parallel') and args.is_parallel:
        return accuracys
    else:
        return all

def self_recursive(all_false_count, all_ids, all_is_rel_kns, all_phrase_para, all_true_count, args, current_time,
                   device, edit_model, input, mid_time, para, start_time, tokenizer, zsre_datas_range):
    step = 0
    rights = 0
    for index0, data in enumerate(zsre_datas_range):
        res_rephrase = {}
        if args.data_type == "cf":
            rephrases = data['paraphrase_prompts']
        else:
            rephrases = data['rephrase']
        for index, rephrase in enumerate(rephrases):
            if (index > 0): continue
            l2_norms = np.array([])
            with torch.no_grad():
                step = step + 1
                if args.type == "memit":
                    input = rephrase
                elif args.type == "paradit":
                    input = SHORT_ANSWER_PROMPT[args.model_para_type].format(rephrase)
                if args.type == "paradit":
                    para = all_phrase_para[input][1]
                    l2_norms = all_phrase_para[input][0]
                # edit model
                edit_model = get_edit_model(args.model_para_type, edit_model, para, args.layer)

                input = tokenizer(input, return_tensors="pt", return_attention_mask=False).to(device)
                if args.model_para_type == "phi2":
                    outputs = edit_model.generate(**input, max_length=200)
                elif args.model_para_type == "gptj":
                    outputs = edit_model.generate(**input, max_length=input['input_ids'].shape[1] + 20)
                outputs = outputs[:, input['input_ids'].shape[1]:]
                prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                label = getLabel(args, data)
                if args.model_para_type == "gptj":
                    stop_sequence = "\\"
                    prediction = prediction.split(stop_sequence)[0]
                    stop_sequence = "\n"
                    prediction = prediction.split(stop_sequence)[0]
                prediction = adjust_dots(prediction.rstrip())
                if args.type == "memit":
                    if prediction.lower().startswith(label.lower()):
                        res_rephrase[rephrase] = str(l2_norms.tolist()) + str(True)
                        rights = rights + 1
                    else:
                        res_rephrase[rephrase] = str(l2_norms.tolist()) + str(prediction)
                elif args.type == "paradit":
                    if label.lower() == prediction.lower():
                        res_rephrase[rephrase] = str(l2_norms.tolist()) + str(True)
                        rights = rights + 1
                    else:
                        res_rephrase[rephrase] = str(l2_norms.tolist()) + str(prediction)
        data['res_rephrase'] = res_rephrase
        data['isFTSuccess'] = data["id"] in all_ids
        data['is_rel_kn'] = all_is_rel_kns[index0].item()
        if args.type == "memit":
            file_path = os.path.join(f"{args.result_path}/result",
                                     f"validate_gr_{args.model_state_dir.split('/')[-2].split('_')[3]}_{current_time}.json")
        elif args.type == "paradit":
            file_path = os.path.join(
                f"{args.result_path}/result_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
                f"validate_gr_{args.data_range}_{args.model_state_dir.split('_')[-1].split('.')[0]}_{current_time}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []
        existing_data.append(data)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
        all = {}
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
            file_path = os.path.join(f"{args.result_path}/result", f'all_gr.json')
        elif args.type == "paradit":
            file_path = os.path.join(
                f"{args.result_path}/result_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}", f'all_gr.json')
        existing_data = []
        existing_data.append(all)
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
    return all


