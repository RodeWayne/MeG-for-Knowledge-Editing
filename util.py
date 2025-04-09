import copy
import os
import random
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn


SHORT_ANSWER_PROMPT = {'phi2':"Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                       'gptj':'Q: Answer the following question in less than 5 words. {}\nA:'}

query_prompt_dict = {1:"Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                        2:"[INST]{}[/INST]",3:'Instruct: {}\nOutput:',
                        3:"Q: Answer the following question in less than 5 words. {}\nA:"}
answer_prompt_dict = {1:'{}<|endoftext|>', 2:'{}.', 3:'{}</s>', 4:'</s>{}</s>', 5:'{}.<|endoftext|>',
                        6:' {}.<|endoftext|>', 7:' {}<|endoftext|>', 8:' {}.\n<|endoftext|>'}

def adjust_dots(s):
    if s.endswith('..'):  # 如果结尾是两个句号
        return s[:-1]  # 保留一个句号
    elif s.endswith('.'):  # 如果结尾是一个句号
        return s[:-1]  # 去掉句号
    return s  # 其他情况保持不变

def add_neuron_for_phi2(model, neuron_num, layer):
    num = 10240
    original_layer_1 = copy.deepcopy(model.model.layers[layer].mlp.fc1)
    original_layer_2 = copy.deepcopy(model.model.layers[layer].mlp.fc2)

    model.model.layers[layer].mlp.fc1 = nn.Linear(2560, num + neuron_num, dtype=torch.float16)    # [10241 * 2560]
    model.model.layers[layer].mlp.fc2 = nn.Linear(num + neuron_num, 2560, dtype=torch.float16)    # [2560 * 10241]

    with torch.no_grad():
        model.model.layers[layer].mlp.fc1.weight[:num, :] = original_layer_1.weight[:num, :].clone().detach()
        model.model.layers[layer].mlp.fc1.bias[:num] = original_layer_1.bias[:num].clone().detach()
        model.model.layers[layer].mlp.fc2.weight[:, :num] = original_layer_2.weight[:, :num].clone().detach()
        model.model.layers[layer].mlp.fc2.bias = original_layer_2.bias

    return model

def add_neuron_for_gptj(model, neuron_num, layer):
    num = 16384

    original_layer_1 = copy.deepcopy(model.transformer.h[layer].mlp.fc_in)  # [16384 * 4096]
    original_layer_2 = copy.deepcopy(model.transformer.h[layer].mlp.fc_out) # [4096, 16384]

    model.transformer.h[layer].mlp.fc_in = nn.Linear(4096, num + neuron_num, dtype=torch.float16)
    model.transformer.h[layer].mlp.fc_out = nn.Linear(num+neuron_num, 4096, dtype=torch.float16)

    with torch.no_grad():
        model.transformer.h[layer].mlp.fc_in.weight[:num, :] = original_layer_1.weight[:num, :].clone().detach()
        model.transformer.h[layer].mlp.fc_in.bias[:num] = original_layer_1.bias[:num].clone().detach()
        model.transformer.h[layer].mlp.fc_out.weight[:, :num] = original_layer_2.weight[:, :num].clone().detach()
        model.transformer.h[layer].mlp.fc_out.bias = original_layer_2.bias

    return model


# 冻结phi2除最后第layer层的ffn层外的参数
def set_grad_phi2(model, layer):
    for n, p in model.named_parameters():
        p.requires_grad = False

    model.model.layers[layer].mlp.fc1.weight.requires_grad = True
    model.model.layers[layer].mlp.fc1.bias.requires_grad = True
    model.model.layers[layer].mlp.fc2.weight.requires_grad = True

    return model


# 冻结gptj除最后第layer层的ffn层外的参数
def set_grad_gptj(model, layer):
    for n, p in model.named_parameters():
        p.requires_grad = False

    model.transformer.h[layer].mlp.fc_in.weight.requires_grad = True
    model.transformer.h[layer].mlp.fc_in.bias.requires_grad = True
    model.transformer.h[layer].mlp.fc_out.weight.requires_grad = True

    return model

# 获取第layer层的FFN层添加neuron_num个新神经元后的gpt-j模型，冻结新增神经元外的所有参数
def initial_gptj_model(neuron_num, layer):
    # model, tokenizer = get_model("/home/wentao/CL_fusion/lqq/add_neuron/GPTJ6B")
    model, tokenizer = get_model('/home/wentao/CL_fusion/lqq/gpt-j-6b')
    model = add_neuron_for_gptj(model, neuron_num, layer)
    return model, tokenizer

# 冻结gpt-j新增神经元外的所有参数
def freeze_gptj(model,layer):
    model = set_grad_gptj(model,layer)
    num = 16384

    freeze_partial_weights_1(model.transformer.h[layer].mlp.fc_in.weight, 0, num)
    freeze_partial_weights_1(model.transformer.h[layer].mlp.fc_in.bias, 0, num)
    freeze_partial_weights_2(model.transformer.h[layer].mlp.fc_out.weight, num)
    return model

# 将neuron_num个神经元的参数载入gpt-j模型第layer层的FFN层
def set_neuron_gptj(fc1_weight, fc1_bias, fc2_weight, model, neuron_num, layer):
    with torch.no_grad():
        model.transformer.h[layer].mlp.fc_in.weight[-neuron_num:, :] = torch.tensor(fc1_weight, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_in.bias[-neuron_num:] = torch.tensor(fc1_bias, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_out.weight[:, -neuron_num:] = torch.tensor(fc2_weight, dtype=torch.float16)
    return model

# 读取params文件，将文件中一个神经元的参数载入gpt-j模型
def set_model_gptj(model, params, neuron_num, layer):
    fc1_add_weight = params['transformer.h.{}.mlp.fc_in.weight'.format(layer)]
    fc1_add_bias = params['transformer.h.{}.mlp.fc_in.bias'.format(layer)]
    fc2_add_weight = params['transformer.h.{}.mlp.fc_out.weight'.format(layer)]
    model = set_neuron_gptj(fc1_add_weight, fc1_add_bias, fc2_add_weight, model, neuron_num, layer)
    return model

# 将neuron_num个神经元的参数载入phi-2模型第layer层的FFN层
def set_neuron_phi2(fc1_weight, fc1_bias, fc2_weight, model, neuron_num, layer):
    with torch.no_grad():
        model.model.layers[layer].mlp.fc1.weight[-neuron_num:, :] = torch.tensor(fc1_weight, dtype=torch.float16)
        model.model.layers[layer].mlp.fc1.bias[-neuron_num:] = torch.tensor(fc1_bias, dtype=torch.float16)
        model.model.layers[layer].mlp.fc2.weight[:, -neuron_num:] = torch.tensor(fc2_weight, dtype=torch.float16)
    return model

# 读取params文件，将文件中一个神经元的参数载入phi-2模型
def set_model_phi2(model, params, neuron_num, layer):
    fc1_add_weight = params['model.layers.{}.mlp.fc1.weight'.format(layer)]
    fc1_add_bias = params['model.layers.{}.mlp.fc1.bias'.format(layer)]
    fc2_add_weight = params['model.layers.{}.mlp.fc2.weight'.format(layer)]
    model = set_neuron_phi2(fc1_add_weight, fc1_add_bias, fc2_add_weight, model, neuron_num, layer)
    return model

# 获取第layer层的FFN层添加n新神经元后的phi-2模型
def initial_phi2_model(neuron_num, layer):
    model, tokenizer = get_model("/home/wentao/CL_fusion/lqq/add_neuron/phi-2")
    model = add_neuron_for_phi2(model, neuron_num, layer)
    return model, tokenizer

# 冻结phi2新增神经元外的所有参数
def freeze_phi2(model,layer):
    model = set_grad_phi2(model,layer)
    num = 10240
    freeze_partial_weights_1(model.model.layers[layer].mlp.fc1.weight, 0, num)
    freeze_partial_weights_1(model.model.layers[layer].mlp.fc1.bias, 0, num)
    freeze_partial_weights_2(model.model.layers[layer].mlp.fc2.weight, num)
    return model

# 通过模型路径获取原始模型
def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# 固定随机数种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义一个函数来冻结权重张量的一部分
# 冻结start_idx行到end_idx行
def freeze_partial_weights_1(param, start_idx, end_idx):
    def hook(grad):
        grad[start_idx:end_idx] = 0
        return grad
    param.register_hook(hook)
    
# 冻结0到n列
def freeze_partial_weights_2(param, n):
    def hook(grad):
        grad[:, :n] = 0
        return grad
    param.register_hook(hook)

# 注册钩子以获取某一层的输入
def hook(module, input, output):
    global layer_input
    layer_input = input[0]