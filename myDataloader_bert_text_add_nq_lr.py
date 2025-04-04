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
    # 重新加载模型和优化器的状态
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 由于我们只保存了部分参数，需要使用 strict=False 来允许部分参数不匹配
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # 获取保存时的epoch信息
    epoch = checkpoint['epoch']
    print(f"Model loaded from epoch {epoch}")
    return model, epoch
# 自定义数据集类
class MyDataset(Dataset):

    def __init__(self,args, paras_dir,rephrases_dir, gpu,is_noise,noisetype,model_para_type,layer=31,fileindex=12):
        self.samples = []
        self.files=[]
        self.xparas=[]

        cache_dir = '/home/wentao/xzw/LLM/bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(cache_dir)
        model = BertModel.from_pretrained(cache_dir)
        model = model.to(f"cuda:{gpu}")
        # model, _ = load_model(model, "/home/wentao/xzw/model_checkpoints/model_epoch_3000.pth", f"cuda:{gpu}")
        if args.isbert_0or1==1:
            # model, _ = load_model(model, "/home/wentao/xzw/model_checkpoints/model_epoch_3000.pth", f"cuda:{gpu}")
            model, _ = load_model(model, args.bertft_dir, f"cuda:{gpu}")

        with open(rephrases_dir, "r") as f:
            edit_data = json.load(f)
        data_by_id = {item['id']: item['query'] for item in edit_data}

        if is_noise:
            ## 读取噪音文件
            noisephrases_dir='/home/wentao/xzw/data_paras/method_1_6/filter_from_same_src.json'
            with open(noisephrases_dir, "r") as f:
                noise_data = json.load(f)
        # a=0
        # 遍历所有子文件夹
        SHORT_ANSWER_PROMPT = {'phi2': "Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                               'gptj': 'Q: Answer the following question in less than 5 words. {}\nA:'}

        for subdir in os.listdir(paras_dir):
            subdir_path = os.path.join(paras_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(subdir_path, file)
                        if file=="params_0.json" and int(subdir_path.split("_")[-1])<fileindex:  # 12=10 1185=1000
                            # print(file_path)
                            # a+=1
                            # print(a)
                            self.files.append(file_path)
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            if model_para_type =="phi2":
                                fc1_weight = torch.tensor(data[f'model.layers.{layer}.mlp.fc1.weight'])
                                fc1_bias = torch.tensor(data[f'model.layers.{layer}.mlp.fc1.bias'])
                                fc2_weight = torch.tensor(data[f'model.layers.{layer}.mlp.fc2.weight'])
                            elif model_para_type == "gptj":
                                fc1_weight = torch.tensor(data[f'transformer.h.{layer}.mlp.fc_in.weight'])
                                fc1_bias = torch.tensor(data[f'transformer.h.{layer}.mlp.fc_in.bias'])
                                fc2_weight = torch.tensor(data[f'transformer.h.{layer}.mlp.fc_out.weight'])
                            x = torch.cat([fc1_weight.flatten(), fc1_bias.flatten(), fc2_weight.flatten()])
                            self.xparas.append(x)
                            learning_prompt = SHORT_ANSWER_PROMPT[model_para_type].format(data_by_id[int(subdir_path.split('_')[-1])])
                            # learning_prompt = data_by_id[int(subdir_path.split('_')[-1])]

                            encoded_input = tokenizer(learning_prompt, return_tensors='pt')
                            # 前向传播
                            outhidden=""
                            with torch.no_grad():  # 不计算梯度
                                encoded_input=encoded_input.to(f"cuda:{gpu}")
                                output = model(**encoded_input)
                            # 1. 方法一：获取pooleroutput
                            # outhidden=output.pooler_output
                            # 2. 方法二；获取cls token
                            outhidden = output.last_hidden_state[:, 0, :]
                            outhidden=outhidden.squeeze(0)
                            if args.is_bert_norm:
                                outhidden = F.normalize(outhidden, p=2, dim=-1)  # L2 归一化
                            self.samples.append([x,outhidden])
                            print(len(self.xparas))
        print(len(self.xparas))

        if is_noise:

            zsre_datas_range = noise_data[:int(len(self.xparas)*args.noise_n1024)]
            # 获取噪音参数
            for index, data in enumerate(zsre_datas_range):
                noise=torch.zeros(args.seq_len)

                query = data['loc'].split('nq question: ')[1] + '?'
                input = SHORT_ANSWER_PROMPT[model_para_type].format(query.capitalize())
                # input = query.capitalize()
                encoded_input = tokenizer(input, return_tensors='pt')
                # 前向传播
                outhidden = ""
                with torch.no_grad():  # 不计算梯度
                    encoded_input = encoded_input.to(f"cuda:{gpu}")
                    output = model(**encoded_input)
                # 1. 方法一：获取pooleroutput
                # outhidden=output.pooler_output
                # 2. 方法二；获取cls token
                outhidden = output.last_hidden_state[:, 0, :]
                outhidden = outhidden.squeeze(0)
                if args.is_bert_norm:
                    outhidden = F.normalize(outhidden, p=2, dim=-1)  # L2 归一化
                self.samples.append([noise, outhidden])
                print(f"len(self.samples): {len(self.samples)}")


    def get_file_name(self):
        return self.files
    def get_xparas(self):
        return self.xparas
    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x,y = self.samples[idx]
        return x, y


# # # 主文件夹路径
# paras_dir = '/home/wentao/xzw/phi2_pth_start_from_all_token_and_rephrases_20241001_noPhrases'
# rephrases_dir = '/home/wentao/xzw/data_orig/zsre-edit-phi2.json'

# paras_dir = '/home/wentao/xzw/data_paras/zsre_phi2_neuron_start_from_new/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/zsre_phi2_neuron_start_from_new/train_success_data_phi2_all.json'

#
# paras_dir = '/home/wentao/xzw/data_paras/zsre_phi2_neuron_start_from_new_random/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/zsre_phi2_neuron_start_from_new_random/train_success_data_phi2_all.json.json'

# paras_dir = '/home/wentao/xzw/phi2_29_after_fc2_bias_orig_init_paras/phi2_prompt_1_1_neuron_1_layer_29'
# rephrases_dir='/home/wentao/xzw/phi2_29_after_fc2_bias_orig_init_paras/train_success_data_phi2_prompt_1_1_neuron_1_layer_29.json'
# 29层after：1027=1024

# paras_dir = '/home/wentao/xzw/data_paras/method_1_6/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/method_1_6/train_success_data_phi2_prompt_1_6_neuron_1_layer_29_len_17203.json'

# paras_dir = '/home/wentao/xzw/data_paras/data_phi2/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/data_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29_len_16987.json'

# 训练ft gptj zsre
# paras_dir = '/home/wentao/CL_fusion/lqq/add_neuron/step_1_data/zsre_gptj/all'
# rephrases_dir = '/home/wentao/CL_fusion/lqq/add_neuron/step_1_data/zsre_gptj/train_success_data_gptj_prompt_1_6_neuron_1_layer_20.json'

# 训练ft gptj zsre new para 换了一个gptj模型文件
# paras_dir = '/home/wentao/xzw/data_paras/new_test_gptj_1000_select/zsre_layer_19/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/new_test_gptj_1000_select/zsre_layer_19/train_success_data_gptj_prompt_3_6_neuron_1_layer_19.json'

# 训练gpt zsre 9 层 1024
# paras_dir = '/home/wentao/xzw/data_paras/zsre_layer_9/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/zsre_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json'

# gptj zsre 10000
# paras_dir = '/home/wentao/xzw/data_paras/zsre_gptj/zsre_layer_9/all'
# rephrases_dir = '/home/wentao/xzw/data_paras/zsre_gptj/zsre_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json'



# 旧的case：12=10 57=50 112=100 299=200 599=500 1185=1000
# 新的case(seed)：12=10 76=50 156=100 318=200 809=500 1674=1000 8209=5000 14531=9000 16109=10000
# 新的case(noseed)：18=10 1725=1000 8803=5000 15883=9000

#
# for i in range(1720,2000):
#     dataset = MyDataset(paras_dir,rephrases_dir,fileindex=i)
#     print(i,len(dataset))
# args = Namespace(isbert_0or1=0,seq_len=5121,noise_n1024=1,bertft_dir="/home/wentao/xzw/LLM/bert_checkpoints/model_epoch_30000.pth")  # 默认值设置为 0 # 8193   # 5121 8193
# # #
# dataset = MyDataset(args, paras_dir,rephrases_dir,gpu=4,is_noise=True, noisetype=0, model_para_type="phi2", layer=29,fileindex=1024)
# dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
# bat=0
# for x, y in dataloader:
#     bat += 1
#     print(f'bat:{bat},Input (x): {x.shape}, Target (y): {y.shape}')
#
# l2_norms=[]
# criterion = nn.MSELoss()
#
# srph=y[:1022]
# lrph=y[1022:]
#
# # 计算 lrph 到 srph 和 lrph 的 L2 距离
# dist_srph = torch.cdist(lrph, srph, p=2)  # 计算 lrph 到 srph 的距离 (1026, 1022)
# dist_lrph = torch.cdist(lrph, lrph, p=2)  # 计算 lrph 之间的距离 (1026, 1026)
#
# # 排除自身距离
# dist_lrph.fill_diagonal_(float('inf'))
#
# # 找到最近邻的索引
# min_srph_dist, min_srph_idx = torch.min(dist_srph, dim=1)  # 最近的 srph 距离
# min_lrph_dist, min_lrph_idx = torch.min(dist_lrph, dim=1)  # 最近的 lrph 距离
#
# # 确定最近的点是 srph 还是 lrph
# is_srph = min_srph_dist < min_lrph_dist  # 最近邻是 srph 的布尔掩码
# is_lrph = ~is_srph  # 最近邻是 lrph
#
# # 计算分类统计
# num_srph = is_srph.sum().item()
# num_lrph = is_lrph.sum().item()
#
# # 计算平均 loss
# mean_loss_srph = min_srph_dist[is_srph].mean().item() if num_srph > 0 else 0
# mean_loss_lrph = min_lrph_dist[is_lrph].mean().item() if num_lrph > 0 else 0
#
# print(f"最近邻属于 srph: {num_srph} 个, 平均 loss: {mean_loss_srph:.6f}")
# print(f"最近邻属于 lrph: {num_lrph} 个, 平均 loss: {mean_loss_lrph:.6f}")
#
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# # 先将 y 移动到 CPU 并转换为 NumPy 数组
# y_cpu = y.cpu().numpy()
#
# # 使用 PCA 降维到 2 维
# pca = PCA(n_components=2)
# y_2d = pca.fit_transform(y_cpu)
#
# # 分离 srph 和 lrph
# srph_2d = y_2d[:1022]
# lrph_2d = y_2d[1022:]
#
# # 画出散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(srph_2d[:, 0], srph_2d[:, 1], label="srph", alpha=0.6, color="blue")
# plt.scatter(lrph_2d[:, 0], lrph_2d[:, 1], label="lrph", alpha=0.6, color="red")
# plt.legend()
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("PCA Visualization of srph and lrph")
# plt.show()