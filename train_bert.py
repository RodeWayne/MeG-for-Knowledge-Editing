import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np
import time
from datetime import datetime
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import sys
import argparse

class CustomDataset(Dataset):
    def __init__(self, data_dir,model_para_type,data_type):
        self.datas = []
        self.SHORT_ANSWER_PROMPT = {'phi2': "Instruct:Answer the following question in less than 5 words. {}\nOutput:",
                               'gptj': 'Q: Answer the following question in less than 5 words. {}\nA:'}
        # 加载数据
        with open(data_dir, "r") as f:
            self.orig_data = json.load(f)
        raw_data = self.orig_data[-2000:]
        if data_type == 'zsre':
            # zsre
            for item in raw_data:
                query = item["src"]
                query = self.SHORT_ANSWER_PROMPT[model_para_type].format(query)
                # 添加等价表述
                for index, phrase in enumerate(item["rephrase"]):
                    if index < 2:
                        phrase = self.SHORT_ANSWER_PROMPT[model_para_type].format(phrase)
                        self.datas.append({
                            "orig": query,
                            "phrase": phrase,
                        })
        else:
            # cf
            for item in raw_data:
                query=item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])
                query = self.SHORT_ANSWER_PROMPT[model_para_type].format(query)
                # 添加等价表述
                for phrase in item["paraphrase_prompts"]:
                    phrase=self.SHORT_ANSWER_PROMPT[model_para_type].format(phrase)
                    self.datas.append({
                        "orig": query,
                        "phrase": phrase,
                    })

    def __len__(self):
        # 返回原始数据和重述数据的总数
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]


def info_nce_loss(query, positive_key, negative_keys, temperature=0.1):
    batch_size = query.shape[0]
    # **L2 归一化** (确保余弦相似度计算正确)
    query = F.normalize(query, p=2, dim=-1)
    positive_key = F.normalize(positive_key, p=2, dim=-1)
    negative_keys = F.normalize(negative_keys, p=2, dim=-1)
    # 计算正样本相似度
    positive_sim = F.cosine_similarity(query, positive_key, dim=-1).unsqueeze(1)  # (batch_size, 1)

    # 计算负样本相似度，但去掉对应位置的正样本
    all_sim = torch.matmul(query, negative_keys.T)  # (batch_size, batch_size)
    mask = torch.eye(batch_size, device=query.device, dtype=torch.bool)
    negatives = all_sim.masked_fill(mask, float('-inf'))  # 屏蔽自身

    # 组合 logits
    logits = torch.cat([positive_sim, negatives], dim=1)  # (batch_size, batch_size)

    # InfoNCE Loss
    labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
    loss = F.cross_entropy(logits / temperature, labels)
    return loss

def train_model(data_dir,save_path, gpu, model_para_type,data_type,temperature, writer, epochs=10, batch_size=16, lr=1e-4):
    # 加载 tokenizer 和模型
    cache_dir = '/home/wentao/xzw/LLM/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(cache_dir)
    model = BertModel.from_pretrained(cache_dir)
    model = model.to(f"cuda:{gpu}")

    # 冻结所有层的参数，除了最后一层
    for name, param in model.named_parameters():
        print(name, param.shape)
        if "encoder.layer.11" not in name:  # 只训练最后一层（BERT-base 第12层，索引为11）
            param.requires_grad = False

    # 自定义数据集和 DataLoader
    dataset = CustomDataset(data_dir,model_para_type,data_type)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    # 定义损失函数和优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # 训练循环
    model.train()
    start_time = datetime.now()
    print(f"len(dataset) = {len(dataset)}")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            orig = tokenizer(data["orig"], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu}")
            phrase = tokenizer(data["phrase"], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu}")
            # 获取嵌入向量
            orig_embed = model(**orig).last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出
            phrase_embed = model(**phrase).last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出
            # 计算 InfoNCE Loss
            loss = info_nce_loss(phrase_embed, orig_embed, orig_embed,temperature=temperature)  # orig_embed 作为负样本池
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(
            f"epoch: {epoch + 1}, batchid: {batch_idx + 1}, len(data): {len(data['orig'])}, loss: {loss.item():.6f}",
            flush=True)
            # 计算从训练开始到当前的时间差
        current_time = datetime.now()
        elapsed_time = current_time - start_time
        # 将时间差转换为天、小时、分钟
        days = elapsed_time.days
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mean_loss = total_loss / len(dataloader)
        print(
            f"[{current_time}] Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.4f}, Elapsed Time: {days} days {hours} hours {minutes} minutes {seconds} seconds",
            flush=True)
        writer.add_scalar("loss",mean_loss, epoch)
        if (epoch + 1) % 100 == 0:
            # 只保存活跃的参数（requires_grad=True的参数）111
            os.makedirs(save_path, exist_ok=True)
            checkpoint_path=os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                "model_state_dict": {name: param.data for name, param in model.named_parameters() if
                                     param.requires_grad},
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"Model saved at epoch {epoch + 1}", flush=True)
            model.eval()
            val(data_type,dataset.orig_data,dataset.SHORT_ANSWER_PROMPT[model_para_type],model,tokenizer,epoch,writer,checkpoint_path)
            model.train()
    print("Training complete!")
    return model

def val(datatype,raw_data,prompt_template,model,tokenizer,epoch,writer,checkpoint_path):
    outs_orig, outs_reph = [], []
    raw_data=raw_data[0:1024]
    # 遍历数据 cf
    if datatype == "cf":
        for item in raw_data:
            query = item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])
            prompts = prompt_template.format(query)
            encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu}")

            with torch.no_grad():
                outputs_locs = model(**encoded_inputs)["last_hidden_state"][:, 0, :]
            outs_orig.append(outputs_locs)

            query = item['paraphrase_prompts'][0]
            prompts = prompt_template.format(query)
            encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu}")

            with torch.no_grad():
                outputs_locs = model(**encoded_inputs)["last_hidden_state"][:, 0, :]
            outs_reph.append(outputs_locs)
    elif datatype == "zsre":
        # 遍历数据 zsre
        for item in raw_data:
            query = item["src"]
            prompts = prompt_template.format(query)
            encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu}")

            with torch.no_grad():
                outputs_locs = model(**encoded_inputs)["last_hidden_state"][:, 0, :]
            outs_orig.append(outputs_locs)

            query = item['rephrase'][0]
            prompts = prompt_template.format(query)
            encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{gpu}")

            with torch.no_grad():
                outputs_locs = model(**encoded_inputs)["last_hidden_state"][:, 0, :]
            outs_reph.append(outputs_locs)

    # 转换为 Tensor
    outs_orig = torch.cat(outs_orig, dim=0)  # (1024, hidden_size)
    outs_reph = torch.cat(outs_reph, dim=0)  # (1024, hidden_size)

    num_samples = outs_reph.shape[0]

    # 计算 MSE Loss
    loss_matrix = torch.cdist(outs_reph, outs_orig, p=2) ** 2

    # 获取最小 MSE 及对应索引
    min_loss, min_indices = loss_matrix.min(dim=1)

    # 统计相同索引的情况
    same_position = (min_indices == torch.arange(num_samples, device=min_indices.device))
    diff_position = ~same_position

    # 统计数量和均值 MSE
    num_same = same_position.sum().item()
    num_diff = diff_position.sum().item()
    mean_loss_same = min_loss[same_position].mean().item() if num_same > 0 else 0
    mean_loss_diff = min_loss[diff_position].mean().item() if num_diff > 0 else 0

    # 输出结果
    print(f"模型: {checkpoint_path}")
    print(f"相同索引数量: {num_same}, 均值 MSE loss: {mean_loss_same}")
    print(f"不同索引数量: {num_diff}, 均值 MSE loss: {mean_loss_diff}")
    # 记录到 TensorBoard
    writer.add_scalar("MSE_loss/mean_same", mean_loss_same, epoch)
    writer.add_scalar("MSE_loss/mean_diff", mean_loss_diff, epoch)
    writer.add_scalar("MSE_loss/num_same", num_same, epoch)
    writer.add_scalar("MSE_loss/num_diff", num_diff, epoch)

# 训练入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model with configurable parameters")
    parser.add_argument("--gpu", type=int, default=4, help="GPU index to use")
    parser.add_argument("--model_para_type", type=str, default="gptj", help="Model parameter type (e.g., gptj)")
    parser.add_argument("--data_type", type=str, default="cf", choices=["zsre", "cf"], help="Data type")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4000, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_logs = "tensorboard_logs"
    log_dir = os.path.join(tensorboard_logs, f"run_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    # 重定向 print 输出到 log 文件
    log_file_path = os.path.join(log_dir, "log.txt")
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout  # 也可以重定向错误日志
    if args.data_type=="zsre":
        data_dir = "data/4000_rephrase.json"  # zsre
    else:
        data_dir = "data/multi_counterfact_new_id.json" # cf1
    save_path = f"checkpoints_trained_bert/bert_{args.model_para_type}_{args.data_type}"  # 保存模型的路径
    pid=os.getpid()
    print(f"gpu:{args.gpu}, model_para_type:{args.model_para_type}, data_type:{args.data_type}, temperature:{args.temperature}, save_path:{save_path}, pid:{pid}")
    model = train_model(data_dir,save_path, args.gpu,args.model_para_type,args.data_type,args.temperature,writer, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
