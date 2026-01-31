import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import BertTokenizer, BertModel
import numpy as np
from argparse import Namespace
import torch.nn.functional as F


def load_model(model, checkpoint_path, device='cuda:'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # use strict=False to allow partial parameter mismatch
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    print(f"Model loaded from epoch {checkpoint_path}")
    print(f"Model loaded from epoch {epoch}")
    return model, epoch
# custom dataset class
class MyDataset(Dataset):

    def __init__(self, args, paras_dir,rephrases_dir, gpu,is_noise,noisetype,model_para_type,layer=31,fileindex=12):
        self.samples = []
        self.files=[]
        self.xparas=[]

        cache_dir = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(cache_dir)
        model = BertModel.from_pretrained(cache_dir)
        model = model.to(f"cuda:{gpu}")

        if args.isbert_0or1==1:
            model, _ = load_model(model, args.bertft_dir, f"cuda:{gpu}")

        with open(rephrases_dir, "r") as f:
            edit_data = json.load(f)
        data_by_id = {item['id']: item['query'] for item in edit_data}

        ## load noise para
        noisephrases_dir=args.noisephrases_dir
        with open(noisephrases_dir, "r") as f:
            noise_data = json.load(f)

        SHORT_ANSWER_PROMPT = {'phi2': "Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                               'gptj': 'Q: Answer the following question in less than 5 words. {}\nA:',
                               'llama3': 'Answer the following question in less than 5 words: {} \nAnswer:'
                               }
        for subdir in os.listdir(paras_dir):
            subdir_path = os.path.join(paras_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(subdir_path, file)
                        if file=="params_0.json" and int(subdir_path.split("_")[-1])<fileindex:
                            self.files.append(file_path)
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
                            elif model_para_type == "llama3":
                                fc1_weight = torch.tensor(data[f'model.layers.{layer}.mlp.extra_proj.weight'])
                                fc1_bias = torch.tensor([])
                                fc2_weight = torch.tensor(data[f'model.layers.{layer}.mlp.down_proj.weight'])
                            x = torch.cat([fc1_weight.flatten(), fc1_bias.flatten(), fc2_weight.flatten()])
                            self.xparas.append(x)
                            learning_prompt = SHORT_ANSWER_PROMPT[model_para_type].format(data_by_id[int(subdir_path.split('_')[-1])])
                            encoded_input = tokenizer(learning_prompt, return_tensors='pt')
                            outhidden=""
                            with torch.no_grad():
                                encoded_input=encoded_input.to(f"cuda:{gpu}")
                                output = model(**encoded_input)
                            # 1. Method 1: Get pooler output
                            # outhidden=output.pooler_output
                            # 2. Method 2: Get [CLS] token
                            outhidden = output.last_hidden_state[:, 0, :]
                            outhidden=outhidden.squeeze(0)
                            if args.is_bert_norm:
                                outhidden = F.normalize(outhidden, p=2, dim=-1)
                            self.samples.append([x,outhidden])
                            print(len(self.xparas))
        print(len(self.xparas))

        if is_noise:
            if args.noisetype_10or2==0:
                # For each case, take 10 corresponding loc cases
                zsre_datas_range_all = noise_data[-int(fileindex / 10):]
                zsre_datas_range=[]
                for noise_loc_datas in zsre_datas_range_all:
                    for loc_data in noise_loc_datas["neighborhood_prompts"]:
                        zsre_datas_range.append(loc_data)
            else:
                # For each case, take the first and last entry of the corresponding loc
                a=-fileindex/2
                zsre_datas_range_all = noise_data[-int((fileindex*args.noise_n1024)/2 ):]
                zsre_datas_range = []
                for noise_loc_datas in zsre_datas_range_all:
                    zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][0])
                    # zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][1])
                    # zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][2])
                    # zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][3])


                    zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][-1])
                    # zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][-2])
                    # zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][-3])
                    # zsre_datas_range.append(noise_loc_datas["neighborhood_prompts"][-4])


            for index, data in enumerate(zsre_datas_range):
                noise=torch.zeros(args.seq_len)

                query = data
                input = SHORT_ANSWER_PROMPT[model_para_type].format(query.capitalize())
                # input = query.capitalize()
                encoded_input = tokenizer(input, return_tensors='pt')
                outhidden = ""
                with torch.no_grad():
                    encoded_input = encoded_input.to(f"cuda:{gpu}")
                    output = model(**encoded_input)
                # 1. Method 1: Get pooler output
                # outhidden=output.pooler_output
                # 2. Method 2: Get [CLS] token
                outhidden = output.last_hidden_state[:, 0, :]
                outhidden = outhidden.squeeze(0)
                if args.is_bert_norm:
                    outhidden = F.normalize(outhidden, p=2, dim=-1)
                self.samples.append([noise, outhidden])
        del model,tokenizer

    def get_file_name(self):
        return self.files
    def get_xparas(self):
        return self.xparas
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return x, y

