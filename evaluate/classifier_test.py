from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import os
from tqdm import *
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=10):
        super(Classifier, self).__init__()

        # FFN层1：扩展 → 压缩（隐藏层维度：768 → 3072 → 768）
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, 3072),  # 扩展维度
            nn.GELU(),
            nn.Linear(3072, hidden_dim),  # 压缩回原维度
            nn.Dropout(0.1)
        )

        # FFN层2：扩展 → 压缩（隐藏层维度：768 → 3072 → 768）
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, 3072),
            nn.GELU(),
            nn.Linear(3072, hidden_dim),
            nn.Dropout(0.1)
        )

        self.ffn3 = nn.Sequential(
            nn.Linear(hidden_dim, 3072),
            nn.GELU(),
            nn.Linear(3072, hidden_dim),
            nn.Dropout(0.1)
        )

        self.ffn4 = nn.Sequential(
            nn.Linear(hidden_dim, 3072),
            nn.GELU(),
            nn.Linear(3072, hidden_dim),
            nn.Dropout(0.1)
        )

        self.ffn5 = nn.Sequential(
            nn.Linear(hidden_dim, 3072),
            nn.GELU(),
            nn.Linear(3072, hidden_dim),
            nn.Dropout(0.1)
        )

        # 最终分类头（将FFN输出映射到类别空间）
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # FFN 层 1 处理 + 残差连接
        x = self.ffn1(x) + x  # 残差连接
        # FFN 层 2 处理 + 残差连接
        x = self.ffn2(x) + x  # 残差连接
        x = self.ffn3(x) + x  # 残差连接
        x = self.ffn4(x) + x  # 残差连接
        x = self.ffn5(x) + x  # 残差连接
        main_logits = self.classifier(x)  # 分类头
        return main_logits


def calculate_entropy(probs):
    """
    计算概率分布的熵。
    probs: 概率分布，形状为 (batch_size, num_classes)
    """
    # 防止 log(0)，用一个非常小的值来代替
    epsilon = 1e-8
    probs = torch.clamp(probs, min=epsilon)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy


# 熟悉网络路径
basedir = '/home/wentao/xzw/LLM/inference/familiar_checkpoints'

checkpoint_path_map = {
    'phi2_zsre_5': os.path.join(basedir, 'phi2_zsre_1024.pth'),
    'phi2_zsre_10': os.path.join(basedir, 'phi2_zsre_1024.pth'),
    'phi2_zsre_100': os.path.join(basedir, 'phi2_zsre_10000.pth'),
    'phi2_zsre_1024': os.path.join(basedir, 'phi2_zsre_1024.pth'),
    'phi2_zsre_10000': os.path.join(basedir, 'phi2_zsre_10000.pth'),
    'phi2_cf_10': os.path.join(basedir, 'phi2_cf_10000.pth'),
    'phi2_cf_100': os.path.join(basedir, 'phi2_cf_10000.pth'),
    'phi2_cf_1024': os.path.join(basedir, 'phi2_cf_1024.pth'),
    'phi2_cf_10000': os.path.join(basedir, 'phi2_cf_10000.pth'),
    'gptj_zsre_10': os.path.join(basedir, 'gptj_zsre_10000.pth'),
    'gptj_zsre_100': os.path.join(basedir, 'gptj_zsre_10000.pth'),
    'gptj_zsre_1024': os.path.join(basedir, 'gptj_zsre_1024.pth'),
    'gptj_zsre_10000': os.path.join(basedir, 'gptj_zsre_10000.pth'),
    'gptj_cf_5': os.path.join(basedir, 'gptj_cf_1024.pth'),
    'gptj_cf_10': os.path.join(basedir, 'gptj_cf_1024.pth'),
    'gptj_cf_1024': os.path.join(basedir, 'gptj_cf_1024.pth'),
    'gptj_cf_100': os.path.join(basedir, 'gptj_cf_10000.pth'),
    'gptj_cf_10000': os.path.join(basedir, 'gptj_cf_10000.pth')

}
# 阈值
entropy_threshold_map = {'gptj_zsre': 0.3, 'gptj_cf': 0.7, 'phi2_zsre': 0.1, 'phi2_cf': 0.1}


# 返回True表示判断为有关知识，返回False表示判断为无关知识
def test(bert_path, dataset, model_name, data_size, query, device):
    checkpoint_path = checkpoint_path_map['{}_{}_{}'.format(model_name, dataset, data_size)]
    # checkpoint_path = "/home/wentao/xzw/LLM/inference/familiar_checkpoints/model_epoch_1000.pth"
    entropy_threshold = entropy_threshold_map['{}_{}'.format(model_name, dataset)]

    # 加载训练好的bert和bert tokenizer
    cache_dir = '/home/wentao/xzw/LLM/bert-base-uncased'
    bert_model = BertModel.from_pretrained(cache_dir)
    bert_model_checkpoint = torch.load(bert_path, map_location=device)
    bert_model.load_state_dict(bert_model_checkpoint["model_state_dict"], strict=False)
    bert_model.to(device)

    query_encode = BertTokenizer.from_pretrained(cache_dir)(
        query, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # 加载familiar网络
    model = Classifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        query_encode.to(device)
        x = bert_model(**query_encode)["last_hidden_state"][:, 0, :]  # 取 [CLS] token 的输出
        logits = model(x)
    probs = F.softmax(logits, dim=-1)
    entropy = calculate_entropy(probs)
    print(entropy)
    print(entropy_threshold)
    output=entropy < entropy_threshold
    true_outputs = torch.tensor( [
       True for _ in output
    ]).to(device)

    return output

import json
if __name__ == "__main__":
    data_dir='/home/wentao/xzw/data_paras_v2/loc/cf_phi2_loc_10000.json'
    with open(data_dir, "r") as f:
        zsre_datas = json.load(f)
    zsre_datas_range = zsre_datas[:1024]
    phrases = []
    for index, data in enumerate(zsre_datas_range):
        rephrases = data['paraphrase_prompts']
        for index, rephrase in enumerate(rephrases):
            if (index > 0): continue
            phrases.append([data["id"], rephrase])
    querys = [data[1] for data in phrases]

    gpu = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = 'cf'  # 'zsre' / 'cf'
    model_name = 'phi2'  # 'phi2' / 'gptj'
    data_size = 1024  # '1024' / '10000'
    # query = ['What university did Watts Humphrey attend?', 'who played desmond doss father in hacksaw ridge?']  # 问题
    bert_path = '/home/wentao/xzw/LLM/bert_phi2_cf_checkpoints_infoNCE/model_epoch_5100.pth'  # 训练好的bert路径
    output = test(bert_path, dataset, model_name, data_size, querys, device)
    is_rel_kns=output
    # print(output)
    true_count = sum(is_rel_kns)  # True 作为 1 计算
    false_count = len(is_rel_kns) - true_count  # False 作为 0 计算
    print(f"True: {true_count}, False: {false_count}")
