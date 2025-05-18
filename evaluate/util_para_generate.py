from transformers import BertTokenizer, BertModel
import torch
from  my_model_five_bert_text import myDiT
from diffusion import create_diffusion
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertModel
import copy
import torch.nn as nn
import os
import json
import logging
from pathlib import Path
import typing
import gc
import torch.nn.functional as F

# from classifier_test import  test


from transformers import AutoTokenizer, AutoModel
import paramiko
import socket
def load_model(model, checkpoint_path, device='cuda:'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    print(f"Model loaded from epoch {epoch}")
    return model, epoch
class ParaFactory:
    def __init__(self,args,model_para_type,device,ps,hidden,nheads,nblocks,modelPth,bmodelTrainStatePath):
        # init bert
        self.device = device
        cache_dir = 'bert-base-uncased'
        self.model_bert = BertModel.from_pretrained(cache_dir)
        self.tokenizer_bert = BertTokenizer.from_pretrained(cache_dir)
        self.args=args

        if bmodelTrainStatePath != "None":
            print("load bert model state ... ")
            self.model_bert, _ = load_model(self.model_bert, bmodelTrainStatePath, device)
        # init model
        if model_para_type =="phi2":
            self.seq_len = 5121
        elif model_para_type =="gptj":
            self.seq_len = 8193
        seq_len_y = 768
        patchsize = ps
        denoise = myDiT(seq_len=self.seq_len, patch_size=patchsize, hidden_size=hidden, num_heads=nheads,
                        num_blocks=nblocks).to(device)
        self.denoise = denoise.to(device)
        state_dict = torch.load(
            modelPth,
            map_location=device
        )
        if(nheads!=12):
            self.denoise.load_state_dict(state_dict)
        else:
            self.denoise.load_state_dict(state_dict["model_state_dict"])
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        self.denoise.eval()  # important!
        self.diffusion = create_diffusion(str(100),predict_v=True)


    def generate(self,input,gtype,srcmodel,srctokenizer,layer):
        layer_inputs = []
        def hook(module, input, output):
            global layer_input
            layer_input = input[0]
            layer_input = layer_input[0]
            layer_input = layer_input[-1]
            layer_inputs.append(layer_input)
        if(gtype=="bert2para"):
            encoded_bert_input = self.tokenizer_bert(input, return_tensors='pt',truncation=True,max_length=self.tokenizer_bert.model_max_length)
            output_bert = self.model_bert(**encoded_bert_input)
            outhidden = output_bert.last_hidden_state[:, 0, :]
            y0 = outhidden.to(self.device)
            y0 = y0.float()
            x_gen = torch.randn(1, 1, self.seq_len).to(self.device)
            model_kwargs = dict(y=y0)
            samples = self.diffusion.p_sample_loop(
                self.denoise, x_gen.shape, x_gen, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=self.device
            )
            x_recov = samples * 0.01
        else:
            encoded_input = srctokenizer(input, return_tensors='pt').to(self.device)
            # select model layer
            # target_layer = srcmodel.model.layers[layer].mlp.fc1
            target_layer = srcmodel.transformer.h[layer].mlp.fc_in
            # register hook
            hook_handle = target_layer.register_forward_hook(hook)
            outputs = srcmodel.generate(**encoded_input, max_length=200)
            outhidden = layer_inputs[0]
            y0 = outhidden.to(self.device)
            y0 = y0.float()
            x_gen = torch.randn(1, 1, self.seq_len).to(self.device)
            model_kwargs = dict(y=y0)
            samples = self.diffusion.p_sample_loop(
                self.denoise, x_gen.shape, x_gen, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=self.device
            )
            x_recov = samples * 0.01

        return x_recov
    def generate_paral(self,is_rel_kns,inputs,gtype,srcmodel,srctokenizer,layer):
        # add index information to inputs
        indexed_inputs = list(enumerate(inputs))
        # split into two parts based on is_rel_kns
        rel_inputs = [inp for idx, inp in indexed_inputs if is_rel_kns[idx]]
        non_rel_inputs = [inp for idx, inp in indexed_inputs if not is_rel_kns[idx]]
        results = {}
        if(gtype=="bert2para" and rel_inputs):
            print(f"generate para number：{len(rel_inputs)}")
            encoded_bert_input = self.tokenizer_bert(rel_inputs, return_tensors='pt',padding=True,return_attention_mask=True)
            max_length = self.tokenizer_bert.model_max_length  #
            # check input length
            input_length = encoded_bert_input['input_ids'].shape[1]
            if input_length > max_length:
                return None
            output_bert = self.model_bert(**encoded_bert_input)
            hidden_states = output_bert.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            attention_mask = encoded_bert_input['attention_mask']
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            outhidden = masked_hidden_states[:, 0, :]  # [CLS] 表示序列语义信息
            if self.args.is_normal:
                outhidden = F.normalize(outhidden, p=2, dim=-1)  # norm
            y0 = outhidden.to(self.device)
            y0 = y0.float()
            x_gen = torch.randn(len(rel_inputs),1, self.seq_len).to(self.device)
            model_kwargs = dict(y=y0)
            samples = self.diffusion.p_sample_loop(
                self.denoise, x_gen.shape, x_gen, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=self.device
            )
            x_recov = samples * 0.01
            for i, idx in enumerate([idx for idx, _ in indexed_inputs if is_rel_kns[idx]]):
                results[idx] = x_recov[i]
        elif gtype=="hidden2para":
            x_recov=None
            encoded_input = srctokenizer(inputs, return_tensors='pt',padding=True,return_attention_mask=True).to(self.device)
            # select model layer
            target_layer = srcmodel.transformer.h[layer].mlp.fc_in
            # register hook
            model_layer_input = None
            def hook(module, input, output):
                nonlocal model_layer_input
                layer_input = input[0]
                attention_mask = encoded_input['attention_mask']
                last_token_idx = attention_mask.sum(dim=1) - 1
                batch_size = layer_input.size(0)
                model_layer_input = layer_input[range(batch_size), last_token_idx]
            hook_handle = target_layer.register_forward_hook(hook)
            outputs = srcmodel(**encoded_input)
            outhidden=model_layer_input
            hook_handle.remove()
            y0 = outhidden.to(self.device)
            y0 = y0.float()
            x_gen = torch.randn(len(inputs), 1, self.seq_len).to(self.device)
            model_kwargs = dict(y=y0)
            samples = self.diffusion.p_sample_loop(
                self.denoise, x_gen.shape, x_gen, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=self.device
            )
            x_recov = samples * 0.01
        # assign all-zero tensor to non_rel_inputs
        zero_tensor = torch.zeros(1,self.seq_len).to(self.device)
        for idx in [idx for idx, _ in indexed_inputs if not is_rel_kns[idx]]:
            results[idx] = zero_tensor
        # restore result order by original indices
        ordered_results = torch.stack([results[idx] for idx, _ in indexed_inputs])
        return ordered_results

def init_model_phi(model_path,device):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True)
    return model.to(device)

def init_model_phi2(model_para_type,model_path,device,layer,is_fc2bias):
    if model_para_type == "phi2":
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True
                                                     )
        original_layer_1 = copy.deepcopy(model.model.layers[layer].mlp.fc1)
        original_layer_2 = copy.deepcopy(model.model.layers[layer].mlp.fc2)
        model.model.layers[layer].mlp.fc1 = nn.Linear(2560, 10241, dtype=torch.float16)  # [10241 * 2560]
        model.model.layers[layer].mlp.fc2 = nn.Linear(10241, 2560, dtype=torch.float16)  # [2560 * 10241]
        num = 10240
        with torch.no_grad():
            model.model.layers[layer].mlp.fc1.weight[:num, :] = original_layer_1.weight[:num, :].clone().detach()
            model.model.layers[layer].mlp.fc1.bias[:num] = original_layer_1.bias[:num].clone().detach()
            model.model.layers[layer].mlp.fc2.weight[:, :num] = original_layer_2.weight[:, :num].clone().detach()
            if is_fc2bias:
                model.model.layers[layer].mlp.fc2.bias = original_layer_2.bias
    elif model_para_type == "gptj":
        num = 16384
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        original_layer_1 = copy.deepcopy(model.transformer.h[layer].mlp.fc_in)  # [16384 * 4096]
        original_layer_2 = copy.deepcopy(model.transformer.h[layer].mlp.fc_out)  # [4096, 16384]

        model.transformer.h[layer].mlp.fc_in = nn.Linear(4096, num + 1, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_out = nn.Linear(num + 1, 4096, dtype=torch.float16)

        with torch.no_grad():
            model.transformer.h[layer].mlp.fc_in.weight[:num, :] = original_layer_1.weight[:num, :].clone().detach()
            model.transformer.h[layer].mlp.fc_in.bias[:num] = original_layer_1.bias[:num].clone().detach()
            model.transformer.h[layer].mlp.fc_out.weight[:, :num] = original_layer_2.weight[:, :num].clone().detach()
            model.transformer.h[layer].mlp.fc_out.bias = original_layer_2.bias
    return model.to(device)

def get_edit_model_memit(model,device,model_state_dir):
    params_dir = Path(model_state_dir)
    params_state=torch.load(params_dir, map_location=device)
    model.load_state_dict(params_state,strict=False)
    params_state = {k: v.cpu() for k, v in params_state.items()}
    del params_state
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    return model
def get_edit_model(model_para_type,model,x_recov,layer):
    sample = x_recov.view(-1)
    if model_para_type == "phi2":
        fc1_weight_recovered = sample[:2560]
        fc1_bias_recovered = sample[2560:2560 + 1]
        fc2_weight_recovered = sample[2560 + 1:]
        fc2_weight_recovered = fc2_weight_recovered.view(2560, 1)
        model.model.layers[layer].mlp.fc1.weight[-1:, :] = torch.tensor(fc1_weight_recovered, dtype=torch.float16)
        model.model.layers[layer].mlp.fc1.bias[-1] = torch.tensor(fc1_bias_recovered, dtype=torch.float16)
        model.model.layers[layer].mlp.fc2.weight[:, -1:] = torch.tensor(fc2_weight_recovered, dtype=torch.float16)
    elif model_para_type == "gptj":
        fc1_weight_recovered = sample[:4096]
        fc1_bias_recovered = sample[4096:4096 + 1]
        fc2_weight_recovered = sample[4096 + 1:]
        fc2_weight_recovered = fc2_weight_recovered.view(4096, 1)
        model.transformer.h[layer].mlp.fc_in.weight[-1:, :] = torch.tensor(fc1_weight_recovered, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_in.bias[-1] = torch.tensor(fc1_bias_recovered, dtype=torch.float16)
        model.transformer.h[layer].mlp.fc_out.weight[:, -1:] = torch.tensor(fc2_weight_recovered, dtype=torch.float16)
    return model

def getValidFileid(model_para_type,root_dir,casenum,layer):
    files = []
    xparas = []
    root_dir = root_dir
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.json'):
                    file_path = os.path.join(subdir_path, file)
                    if file == "params_0.json" and int(subdir_path.split("_")[-1]) < casenum:  # 12=10 1185=1000
                        files.append(file_path)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        if model_para_type == "phi2":
                            fc1_weight = torch.tensor(data[f'model.layers.{layer}.mlp.fc1.weight'])
                            fc1_bias = torch.tensor(data[f'model.layers.{layer}.mlp.fc1.bias'])
                            fc2_weight = torch.tensor(data[f'model.layers.{layer}.mlp.fc2.weight'])
                        elif model_para_type == "gptj":
                            fc1_weight = torch.tensor(data[f'transformer.h.{layer}.mlp.fc_in.weight'])
                            fc1_bias = torch.tensor(data[f'transformer.h.{layer}.mlp.fc_in.bias'])
                            fc2_weight = torch.tensor(data[f'transformer.h.{layer}.mlp.fc_out.weight'])
                        x = torch.cat([fc1_weight.flatten(), fc1_bias.flatten(), fc2_weight.flatten()])
                        xparas.append(x)
                        print(len(xparas))

    all_id=[]
    for file in files:
        trainid = int(file.split('/')[-2].split('_')[-1])
        all_id.append(trainid)
    return all_id,xparas

def getNoiseFileid(root_path,noisecase_num):
    xparas=[]
    all_id = []
    root_dir = root_path
    for file in os.listdir(root_dir):
        all_id.append(int(file.split('.')[0].split('_')[-1])) #
        file_path = os.path.join(root_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            para=torch.tensor(data["para"])
            xparas.append(para)
            if(len(xparas)==noisecase_num):break;
    return all_id,xparas


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target,device):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)
        correct_id = tok(target, padding=True, return_tensors="pt").to(device)[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()
        return (ans == correct_id).detach().cpu().numpy().tolist()
def adjust_dots(s):
    if s.endswith('..'):  # If the string ends with two periods
        return s[:-1]  # Keep one period
    elif s.endswith('.'):  # If the string ends with one period
        return s[:-1]  # Remove the period
    return s  # Keep unchanged in other cases

def initDeviceModelDataAndParadit(args):
    # Initialize variables
    para_factory = None
    all_ids = None
    xparas = None
    device = torch.device(f"cuda:{args.gpu}")
    # load data
    with open(args.data_dir, "r") as f:
        zsre_datas = json.load(f)
    if hasattr(args, 'is_parallel') and args.is_parallel:
        # need to distribute data to be parallel
        zsre_datas_range = zsre_datas[args.start_index:args.end_index]
    else:
        zsre_datas_range = zsre_datas[:args.data_range]
    # load model
    if args.type == "memit":
        model = init_model_phi(args.model_path, device)
        # edit_model=model
        edit_model = get_edit_model_memit(model, device, model_state_dir=args.model_state_dir)
    elif args.type == "paradit":
        all_ids, xparas = getValidFileid(args.model_para_type, args.para_dir, args.fi, args.layer)
        edit_model = init_model_phi2(args.model_para_type, args.model_path, device, layer=args.layer,
                                is_fc2bias=args.is_fc2bias)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # load para_factory
    if args.type == "paradit":
        para_factory = ParaFactory(args,args.model_para_type, device, ps=args.ps, hidden=args.hidden, nheads=args.nheads,
                                   nblocks=12, modelPth=args.model_state_dir,
                                   bmodelTrainStatePath=args.bmodel_train_state_path)
    return device, edit_model, tokenizer, para_factory, zsre_datas_range,all_ids, xparas
def getLabel(args, data):
    if args.type == "memit":
        if (args.data_type == "cf"):
            label = ' ' + data['requested_rewrite']['target_new']['str']
        else:
            label = ' ' + data['answers'][0]
    elif args.type == "paradit":
        if (args.data_type == "cf"):
            label = ' ' + data['requested_rewrite']['target_new']['str']
        else:
            if args.temp=="newzsre":
                label = ' ' + data['answers'][0]
            else:
                label = ' ' + data['label']

    return label
from datetime import datetime, timedelta
def format_elapsed_time(seconds):
    """Format time as d day h hous m minutes s seconds"""
    delta = timedelta(seconds=int(seconds))
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"
SHORT_ANSWER_PROMPT = {'phi2': "Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                       'gptj': 'Q: Answer the following question in less than 5 words. {}\nA:'}



if __name__ == '__main__':
    a=getNoiseFileid()


