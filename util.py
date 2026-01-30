import copy
import os
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

class PatchedLlamaMLP(nn.Module):
    def __init__(self, neuron_num,original_mlp):
        super().__init__()
        self.hidden_size = original_mlp.hidden_size
        self.intermediate_size = original_mlp.intermediate_size
        self.config = original_mlp.config
        self.act_fn = original_mlp.act_fn

        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj

        self.extra_proj = nn.Linear(self.hidden_size, neuron_num, bias=False,dtype=original_mlp.down_proj.weight.dtype)

        # down_proj: [hidden_size, intermediate_size + 1]
        self.down_proj = nn.Linear(self.intermediate_size + neuron_num, self.hidden_size, bias=False,dtype=original_mlp.down_proj.weight.dtype)

        with torch.no_grad():
            self.down_proj.weight[:, :-neuron_num] = original_mlp.down_proj.weight
        

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("pretraining_tp > 1 is not supported in patched version.")

        gate_out = self.gate_proj(x)  # [B, T, I]
        up_out = self.up_proj(x)      # [B, T, I]
        intermediate = self.act_fn(gate_out) * up_out  # [B, T, I]

        extra = self.extra_proj(x)  # [B, T, 1]

        intermediate_aug = torch.cat([intermediate, extra], dim=-1)  # [B, T, I+1]

        out = self.down_proj(intermediate_aug)  # [B, T, H]
        return out
    
def initial_llama_model(neuron_num, layer):
    model, tokenizer = get_model('meta-llama/Meta-Llama-3-8B-Instruct')
    original_mlp = model.model.layers[layer].mlp
    patched_mlp = PatchedLlamaMLP(neuron_num, original_mlp)
    model.model.layers[layer].mlp = patched_mlp
    return model, tokenizer



SHORT_ANSWER_PROMPT = {'phi2':"Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                       'gptj':'Q: Answer the following question in less than 5 words. {}\nA:',
                       'llama3':'Answer the following question in less than 5 words: {} \nAnswer:'}

query_prompt_dict = {1:"Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                        2:"[INST]{}[/INST]",3:'Instruct: {}\nOutput:',
                        3:"Q: Answer the following question in less than 5 words. {}\nA:",
                        4:'Answer the following question in less than 5 words: {} \nAnswer:'}
answer_prompt_dict = {1:'{}<|endoftext|>', 2:'{}.', 3:'{}</s>', 4:'</s>{}</s>', 5:'{}.<|endoftext|>',
                        6:' {}.<|endoftext|>', 7:' {}<|endoftext|>', 8:' {}.\n<|endoftext|>', 
                        9:' {}.<|end_of_text|>'}

def adjust_dots(s):
    if s.endswith('..'):  # If the string ends with two dots
        return s[:-1]  # Keep one dot
    elif s.endswith('.'):  # If the string ends with one dot
        return s[:-1]  # Remove the dot
    return s  # Otherwise, return the string unchanged

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


# Freeze all parameters in phi2 except the FFN (Feed-Forward Network) layers in the selected `layer` layers
def set_grad_phi2(model, layer):
    for n, p in model.named_parameters():
        p.requires_grad = False

    model.model.layers[layer].mlp.fc1.weight.requires_grad = True
    model.model.layers[layer].mlp.fc1.bias.requires_grad = True
    model.model.layers[layer].mlp.fc2.weight.requires_grad = True

    return model


# Freeze all parameters in gptj except the FFN (Feed-Forward Network) layers in the selected `layer` layers
def set_grad_gptj(model, layer):
    for n, p in model.named_parameters():
        p.requires_grad = False

    model.transformer.h[layer].mlp.fc_in.weight.requires_grad = True
    model.transformer.h[layer].mlp.fc_in.bias.requires_grad = True
    model.transformer.h[layer].mlp.fc_out.weight.requires_grad = True

    return model

# modify a GPT-J model by adding new neurons to a specific layer's FFN and freeze all parameters except these new neurons
def initial_gptj_model(neuron_num, layer):
    model, tokenizer = get_model("EleutherAI/gpt-j-6B")
    model = add_neuron_for_gptj(model, neuron_num, layer)
    return model, tokenizer

# Freeze all parameters in GPT-J except the newly added neurons
def freeze_gptj(model,layer):
    model = set_grad_gptj(model,layer)
    num = 16384

    freeze_partial_weights_1(model.transformer.h[layer].mlp.fc_in.weight, 0, num)
    freeze_partial_weights_1(model.transformer.h[layer].mlp.fc_in.bias, 0, num)
    freeze_partial_weights_2(model.transformer.h[layer].mlp.fc_out.weight, num)
    return model

# load neuron_num new neurons into a specific FFN layer of a GPT-J model while keeping all other parameters frozen
def set_neuron_gptj(fc1_weight, fc1_bias, fc2_weight, model, neuron_num, layer):
    with torch.no_grad():
        model.transformer.h[layer].mlp.fc_in.weight[-neuron_num:, :] = torch.tensor(fc1_weight, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_in.bias[-neuron_num:] = torch.tensor(fc1_bias, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_out.weight[:, -neuron_num:] = torch.tensor(fc2_weight, dtype=torch.float16)
    return model

# load neuron's parameters from a params file into a specific FFN layer of GPT-J
def set_model_gptj(model, params, neuron_num, layer):
    fc1_add_weight = params['transformer.h.{}.mlp.fc_in.weight'.format(layer)]
    fc1_add_bias = params['transformer.h.{}.mlp.fc_in.bias'.format(layer)]
    fc2_add_weight = params['transformer.h.{}.mlp.fc_out.weight'.format(layer)]
    model = set_neuron_gptj(fc1_add_weight, fc1_add_bias, fc2_add_weight, model, neuron_num, layer)
    return model

# load neuron_num new neurons into a specific FFN layer of a Phi-2 model while keeping all other parameters frozen
def set_neuron_phi2(fc1_weight, fc1_bias, fc2_weight, model, neuron_num, layer):
    with torch.no_grad():
        model.model.layers[layer].mlp.fc1.weight[-neuron_num:, :] = torch.tensor(fc1_weight, dtype=torch.float16)
        model.model.layers[layer].mlp.fc1.bias[-neuron_num:] = torch.tensor(fc1_bias, dtype=torch.float16)
        model.model.layers[layer].mlp.fc2.weight[:, -neuron_num:] = torch.tensor(fc2_weight, dtype=torch.float16)
    return model

# load neuron's parameters from a params file into a specific FFN layer of Phi-2
def set_model_phi2(model, params, neuron_num, layer):
    fc1_add_weight = params['model.layers.{}.mlp.fc1.weight'.format(layer)]
    fc1_add_bias = params['model.layers.{}.mlp.fc1.bias'.format(layer)]
    fc2_add_weight = params['model.layers.{}.mlp.fc2.weight'.format(layer)]
    model = set_neuron_phi2(fc1_add_weight, fc1_add_bias, fc2_add_weight, model, neuron_num, layer)
    return model

# modify a Phi-2 model by adding new neurons to a specific layer's FFN and freeze all parameters except these new neurons
def initial_phi2_model(neuron_num, layer):
    model, tokenizer = get_model("microsoft/phi-2")
    model = add_neuron_for_phi2(model, neuron_num, layer)
    return model, tokenizer

# Freeze all parameters in Phi-2 except the newly added neurons
def freeze_phi2(model,layer):
    model = set_grad_phi2(model,layer)
    num = 10240
    freeze_partial_weights_1(model.model.layers[layer].mlp.fc1.weight, 0, num)
    freeze_partial_weights_1(model.model.layers[layer].mlp.fc1.bias, 0, num)
    freeze_partial_weights_2(model.model.layers[layer].mlp.fc2.weight, num)
    return model

# load the original model from a given model path
def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# fix random seeds
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define a function to freeze a portion of a weight tensor
# Freeze rows from start_idx to end_idx
def freeze_partial_weights_1(param, start_idx, end_idx):
    def hook(grad):
        grad[start_idx:end_idx] = 0
        return grad
    param.register_hook(hook)
    
# Freeze columns from start_idx to end_idx
def freeze_partial_weights_2(param, n):
    def hook(grad):
        grad[:, :n] = 0
        return grad
    param.register_hook(hook)

# register a hook to get the inputs to a specific layer
def hook(module, input, output):
    global layer_input
    layer_input = input[0]