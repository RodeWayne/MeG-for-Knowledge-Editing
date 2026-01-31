from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import os
from tqdm import *
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=10):
        super(Classifier, self).__init__()

        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, 3072),
            nn.GELU(),
            nn.Linear(3072, hidden_dim),
            nn.Dropout(0.1)
        )

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

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.ffn1(x) + x
        x = self.ffn2(x) + x
        x = self.ffn3(x) + x
        x = self.ffn4(x) + x
        x = self.ffn5(x) + x
        main_logits = self.classifier(x)
        return main_logits


def calculate_entropy(probs):
    """
    calculate_entropy
    probs: shape:(batch_size, num_classes)
    """
    epsilon = 1e-8
    probs = torch.clamp(probs, min=epsilon)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy


basedir = 'familiar_network/checkpoints/'

checkpoint_path_map = {
    'phi2_zsre_1024': os.path.join(basedir, 'phi2_zsre_1024.pth'),
    'phi2_zsre_2048': os.path.join(basedir, 'phi2_zsre_2048.pth'),
    'phi2_zsre_4096': os.path.join(basedir, 'phi2_zsre_4096.pth'),
    'phi2_zsre_10000': os.path.join(basedir, 'phi2_zsre_10000.pth'),
    'phi2_cf_1024': os.path.join(basedir, 'phi2_cf_1024.pth'),
    'phi2_cf_2048': os.path.join(basedir, 'phi2_cf_2048.pth'),
    'phi2_cf_4096': os.path.join(basedir, 'phi2_cf_4096.pth'),
    'phi2_cf_10000': os.path.join(basedir, 'phi2_cf_10000.pth'),
    'gptj_zsre_1024': os.path.join(basedir, 'gptj_zsre_1024.pth'),
    'gptj_zsre_2048': os.path.join(basedir, 'gptj_zsre_2048.pth'),
    'gptj_zsre_4096': os.path.join(basedir, 'gptj_zsre_4096.pth'),
    'gptj_zsre_10000': os.path.join(basedir, 'gptj_zsre_10000.pth'),
    'gptj_cf_1024': os.path.join(basedir, 'gptj_cf_1024.pth'),
    'gptj_cf_2048': os.path.join(basedir, 'gptj_cf_2048.pth'),
    'gptj_cf_4096': os.path.join(basedir, 'gptj_cf_4096.pth'),
    'gptj_cf_10000': os.path.join(basedir, 'gptj_cf_10000.pth'),
    'llama3_zsre_1024': os.path.join(basedir, 'llama3_zsre_1024.pth'),
    'llama3_zsre_2048': os.path.join(basedir, 'llama3_zsre_2048.pth'),
    'llama3_zsre_4096': os.path.join(basedir, 'llama3_zsre_4096.pth'),
    'llama3_zsre_10000': os.path.join(basedir, 'llama3_zsre_10000.pth'),
    'llama3_cf_1024': os.path.join(basedir, 'llama3_cf_1024.pth'),
    'llama3_cf_2048': os.path.join(basedir, 'llama3_cf_2048.pth'),
    'llama3_cf_4096': os.path.join(basedir, 'llama3_cf_4096.pth'),
    'llama3_cf_10000': os.path.join(basedir, 'llama3_cf_10000.pth'),

}

entropy_threshold_map = {'gptj_zsre': 0.3, 'gptj_cf': 0.7, 'phi2_zsre': 0.1, 'phi2_cf': 0.1, 'llama3_zsre':0.3,'llama3_cf':0.6}


# Return True if it is determined to be relevant knowledge, otherwise return False.
def test(bert_path, dataset, model_name, data_size, query, device):
    checkpoint_path = checkpoint_path_map['{}_{}_{}'.format(model_name, dataset, data_size)]
    entropy_threshold = entropy_threshold_map['{}_{}'.format(model_name, dataset)]

    cache_dir = 'bert-base-uncased'
    bert_model = BertModel.from_pretrained(cache_dir)
    bert_model_checkpoint = torch.load(bert_path, map_location=device)
    bert_model.load_state_dict(bert_model_checkpoint["model_state_dict"], strict=False)
    bert_model.to(device)

    query_encode = BertTokenizer.from_pretrained(cache_dir)(
        query, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # load familiar network
    model = Classifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        query_encode.to(device)
        x = bert_model(**query_encode)["last_hidden_state"][:, 0, :]
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
    data_dir='data/edit_data/gptj_cf_edit.json'
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
    # query = ['What university did Watts Humphrey attend?', 'who played desmond doss father in hacksaw ridge?']
    bert_path = 'checkpoints_trained_bert/bert_phi2_cf/model_epoch_5100.pth'
    output = test(bert_path, dataset, model_name, data_size, querys, device)
    is_rel_kns=output
    # print(output)
    true_count = sum(is_rel_kns)  #
    false_count = len(is_rel_kns) - true_count
    print(f"True: {true_count}, False: {false_count}")
