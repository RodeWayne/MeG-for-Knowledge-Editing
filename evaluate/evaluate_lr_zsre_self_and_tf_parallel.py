import argparse

import torch

from util_para_generate import *
from datetime import datetime
import time
from classifier_test import  test
from itertools import chain
import numpy as np
def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 当前时间格式化
    start_time = time.time()
    device, edit_model, tokenizer, para_factory, zsre_datas_range, all_ids, xparas = initDeviceModelDataAndParadit(args)
    # 生成参数
    # 1.组成所有的句子
    if args.type == "paradit":
        phrases=[]
        for index, data in enumerate(zsre_datas_range):
            with torch.no_grad():
                if args.data_type == "cf":
                    # query = data['neighborhood_prompts'][0]
                    query = next(iter(data['neighborhood_prompts']))  # 获取第一个键
                else:
                    if args.temp=="newcf":
                        query=next(iter(data['loc']))
                    else:
                        query = data['loc'].split('nq question: ')[1] + '?'

                phrases.append([data["id"],query])
        # 2. 生成句子和对应的para
        batch_size = args.batch_size  # 获取批量大小参数
        num_batches = (len(phrases) + batch_size - 1) // batch_size  # 计算总批次数
        all_phrase_para = {}
        all_is_rel_kns = []  # 用于存储所有 batch 的 is_rel_kns
        all_true_count=0
        all_false_count=0
        for batch_idx in range(num_batches):
            batch_data = phrases[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if args.data_type=="zsre":
                inputs = [SHORT_ANSWER_PROMPT[args.model_para_type].format(data[1].capitalize()) for data in batch_data]
                querys = [data[1].capitalize() for data in batch_data]
            else:
                inputs = [SHORT_ANSWER_PROMPT[args.model_para_type].format(data[1]) for data in batch_data]
                querys = [data[1] for data in batch_data]
            # 确定是否为有关知识
            if args.is_rel:
                is_rel_kns = test(args.bmodel_train_state_path, args.data_type, args.model_para_type, args.fi, querys,
                                  device)
            else:
                is_rel_kns = torch.ones(len(querys), dtype=torch.bool).to(device)  # 全 True
            true_count = sum(is_rel_kns)  # True 作为 1 计算
            false_count = len(is_rel_kns) - true_count  # False 作为 0 计算
            print(f"True: {true_count}, False: {false_count}")
            all_true_count+=true_count
            all_false_count+=false_count
            all_is_rel_kns.extend(is_rel_kns)
            # 生成参数
            with torch.no_grad():
                paras = para_factory.generate_paral(is_rel_kns, inputs, args.gtype, srcmodel=edit_model,
                                                    srctokenizer=tokenizer,
                                                    layer=args.layer)
                # paras = torch.zeros(batch_size, 8193).to(device)
                for data, input, para in zip(batch_data, inputs, paras):
                    print(f"匹配：{input}")
                    l2_norms_min = float('inf')  # 初始化为无穷大
                    l2_norms_min_caseid = -1
                    # for id, x_orig in zip(all_ids, xparas):
                    #     x_orig = x_orig.to(device)
                    #     l2_norms = torch.norm(x_orig - para.squeeze(1), p=2, dim=1)
                    #     if l2_norms < l2_norms_min:
                    #         l2_norms_min = l2_norms
                    #         l2_norms_min_caseid = id
                    # all_phrase_para[input]=[l2_norms_min_caseid,l2_norms_min.item(),para]
                    all_phrase_para[input]=[l2_norms_min_caseid,l2_norms_min,para]
    mid_time = time.time()
    # 迭代评估
    step = 0
    accuracys_tf=[]
    for index, data in enumerate(zsre_datas_range):
        with torch.no_grad():
            step = step + 1
            if args.data_type=="cf":
                # query = data['neighborhood_prompts'][0]
                query = next(iter(data['neighborhood_prompts']))  # 获取第一个键
            else:
                if args.temp == "newcf":
                    query = next(iter(data['loc']))
                else:
                    query = data['loc'].split('nq question: ')[1] + '?'
            # input=SHORT_ANSWER_PROMPT['phi2'].format(data['src'])
            if args.data_type=="zsre":
                input=SHORT_ANSWER_PROMPT[args.model_para_type].format(query.capitalize())
            else:
                input=SHORT_ANSWER_PROMPT[args.model_para_type].format(query)
            # 生成参数
            if args.type == "paradit":
                para = all_phrase_para[input][2]
                data['l2_norms_min_caseid'] = all_phrase_para[input][0]
                data['l2_norms_min'] = all_phrase_para[input][1]
                # 编辑模型
                edit_model = get_edit_model(args.model_para_type,edit_model, para, args.layer)

            all_self_recursive = self_recursive(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model, all_false_count,
                                 index, input, mid_time, query, start_time, step, tokenizer, all_true_count)

            all_teach_forcing = teach_forcing(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model, false_count,
                                              index, input, mid_time, query, start_time, step, tokenizer, true_count,accuracys_tf)

    return all_self_recursive["accuracy"],all_teach_forcing["accuracy_mean"],all_teach_forcing["accuracy_std"]


def self_recursive(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model, all_false_count, index, input,
                   mid_time, query, start_time, step, tokenizer, all_true_count):
    input = tokenizer(input, return_tensors="pt", return_attention_mask=False).to(device)
    if args.model_para_type == "phi2":

        outputs = edit_model.generate(**input, max_length=200,
                                      temperature=0.0,
                                      do_sample=False,  # 关闭随机取样
                                      # num_beams=1  # 贪心搜搜
                                      )
    elif args.model_para_type == "gptj":
        outputs = edit_model.generate(**input, max_length=input['input_ids'].shape[1] + 20)
    outputs = outputs[:, input['input_ids'].shape[1]:]
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    if args.model_para_type == "gptj":
        # stop_sequence = "\\"
        # prediction = prediction.split(stop_sequence)[0]
        stop_sequence = "\n"
        prediction = prediction.split(stop_sequence)[0]
    prediction = prediction.strip().rstrip('.')
    # if data['pred'] == prediction:
    if args.data_type == "cf":
        label = data['neighborhood_prompts'][query]
    else:

        if args.temp == "newcf":
            # query=next(iter(data['loc']))
            label = data['loc'][query]
        else:
            label = data['loc_pred']
    if label == prediction:
        data['generate'] = True
    else:
        data['generate'] = prediction
    data['isFTSuccess'] = data["id"] in all_ids
    data['is_rel_kn'] = all_is_rel_kns[index].item()
    if args.type == "memit":
        file_path = os.path.join(f"{args.result_path}/result",
                                 f"validate_lr_zsre_{args.model_state_dir.split('/')[-2].split('_')[3]}_{current_time}.json")
    elif args.type == "paradit":
        file_path = os.path.join(f"{args.result_path}/result_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
                                 f"validate_lr_zsre_{args.data_range}_{args.model_state_dir.split('_')[-1].split('.')[0]}_{current_time}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []
        # 添加新的detail
    existing_data.append(data)
    # 写入更新后的数据
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    # 每一步都统计接入文件
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
    elapsed_time = format_elapsed_time(end_time - start_time)  # 格式化耗时
    all["end_cost_time"] = elapsed_time
    if args.type == "memit":
        file_path = os.path.join(f"{args.result_path}/result", f'all_lr_zsre.json')
    elif args.type == "paradit":
        file_path = os.path.join(f"{args.result_path}/result_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
                                 f'all_lr_zsre.json')
    # if os.path.exists(file_path):
    #     with open(file_path, 'r') as file:
    #         existing_data = json.load(file)
    # else:
    #     existing_data = []
    #     # 添加新的detail
    existing_data = []
    existing_data.append(all)
    # 写入更新后的数据
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    return all

def teach_forcing(all_ids, all_is_rel_kns, args, current_time, data, device, edit_model, all_false_count, index, input,
                   mid_time, query, start_time, step, tokenizer, all_true_count,accuracys_tf):
    # teach forcing测试
    prob_prompts = [[input]]
    inp_prompts_og = list(chain(*prob_prompts))
    if args.data_type == "cf":
        label = data['neighborhood_prompts'][query]
    else:
        if args.temp == "newcf":
            # query=next(iter(data['loc']))
            label = data['loc'][query]
        else:
            label = data['loc_pred']

        # label = data['loc'][query]
    target_tok = tokenizer(' ' + label)["input_ids"]
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
            f"validate_lr_zsre_{args.model_state_dir.split('/')[-2].split('_')[3]}_{current_time}.json")
    elif args.type == "paradit":
        file_path = os.path.join(
            f"{args.result_path}/result_tf_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
            f"validate_lr_zsre_{args.data_range}_{args.model_state_dir.split('_')[-1].split('.')[0]}_{current_time}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []
        # 添加新的detail
    existing_data.append(data)
    # 写入更新后的数据
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
    elapsed_time = format_elapsed_time(end_time - start_time)  # 格式化耗时
    all["end_cost_time"] = elapsed_time
    if args.type == "memit":
        file_path = os.path.join(f"{args.result_path}/result_tf", f'all_lr_zsre.json')
    elif args.type == "paradit":
        file_path = os.path.join(f"{args.result_path}/result_tf_{args.layer}_fc2bias_{args.is_fc2bias}_nt{args.noisetype}",
                                 f'all_lr_zsre.json')
    # if os.path.exists(file_path):
    #     with open(file_path, 'r') as file:
    #         existing_data = json.load(file)
    # else:
    #     existing_data = []
    existing_data = []

    # 添加新的detail
    existing_data.append(all)
    # 写入更新后的数据
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    return all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--model_state_dir", type=str,
                        default="/home/wentao/xzw/checkpoint/004_20241214165933/checkpoints/model_epoch_all_20000.pth")
    parser.add_argument("--data_dir", type=str, default="zsre_phi2_pred_correct_loc.json")
    parser.add_argument("--para_dir", type=str, default="/home/wentao/xzw/phi2_29_after_fc2_bias_orig_init_paras/phi2_prompt_1_1_neuron_1_layer_29")
    parser.add_argument("--data_range", type=int, default=1024)
    parser.add_argument("--fi", type=int, default=1027)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--ps", type=int, default=100)

    parser.add_argument("--is_fc2bias", action="store_true")
    parser.add_argument("--noisetype", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--gtype", type=str, default='bert2para')
    parser.add_argument("--bmodel_train_state_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="result/")

    # 新的 1713=1024 3424=2048  6773=4096  13207=8192

    args = parser.parse_args()
    main(args)

