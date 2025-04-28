import argparse
from transformers import BertTokenizer, BertModel
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from util import *
import yaml

class CustomDataset(Dataset):
    def __init__(self, data_type,data_size,model_name):
        self.original_data = []
        # edit data dir
        if data_type == 'cf':
            data_dir = 'data/train_familiar/cf_fake_id_class.json'
        else:
            data_dir = 'data/train_familiar/zsre_{}_fake_id_class.json'.format(model_name)
        # load data
        with open(data_dir, "r") as f:
            raw_data = json.load(f)
        raw_data = raw_data[:data_size]
        for item in raw_data:
            if data_type == 'zsre':
                query=item["src"]
                rephrase = item['rephrase'][0]
                loc = item['loc'].split('nq question: ')[1] + '?'
            else:
                query=item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])
                rephrase = item['paraphrase_prompts'][0]
                loc = item['neighborhood_prompts'][0]
            self.original_data.append({
                "query":query,
                "label":item['fake_label'],
                "rephrase":rephrase,
                "loc":loc
            })
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        data = self.original_data[idx]

        return data["query"], data["label"], data["rephrase"], data["loc"]


class BertClassifier(nn.Module):
    def __init__(self,checkpoint_path,gpu, hidden_dim=768, num_classes=10):
        super(BertClassifier, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu}")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        for param in self.model.parameters():
            param.requires_grad = False

        # FFN1：（768 → 3072 → 768）
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, 3072), 
            nn.GELU(),
            nn.Linear(3072, hidden_dim), 
            nn.Dropout(0.1)
        )
        
        # FFN2：（768 → 3072 → 768）
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
        
        # maps FFN output to class space
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, encoded_inputs):
        x = self.model(**encoded_inputs)["last_hidden_state"][:, 0, :]  # get [CLS] token 
        x = self.ffn1(x) + x
        x = self.ffn2(x) + x
        x = self.ffn3(x) + x
        x = self.ffn4(x) + x
        x = self.ffn5(x) + x 
        main_logits = self.classifier(x)
        return main_logits


def calculate_entropy(probs):
    """
    Calculate the entropy of a probability distribution.
    probs: Probability distribution with shape (batch_size, num_classes)
    """
    # To prevent log(0), replace it with a very small value
    epsilon = 1e-8
    probs = torch.clamp(probs, min=epsilon)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy


def train_model(model_name,log_save_path,data_type,checkpoint_path,save_path,data_size, gpu, epochs, batch_size, lr, entropy_threshold):
    layers_to_save = ['ffn1','ffn2','ffn3','ffn4','ffn5','classifier']
    train_writer = SummaryWriter(log_dir=log_save_path.format('train'))
    rephrase_writer = SummaryWriter(log_dir=log_save_path.format('rephrase'))
    loc_writer = SummaryWriter(log_dir=log_save_path.format('loc'))
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    # get dataset and DataLoader
    dataset = CustomDataset(data_type, data_size,model_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load model
    model = BertClassifier(checkpoint_path,gpu).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # train
    pbar = tqdm(range(0,epochs), desc="training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        train_correct = 0
        test_correct = 0
        total = 0
        train_entropy_correct, test_entropy_correct, loc_entropy_correct = 0,0,0

        for queries, labels, rephrases, locs in dataloader:
            encoded_inputs = BertTokenizer.from_pretrained('bert-base-uncased')(
                queries, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels = labels.to(device).squeeze()

            optimizer.zero_grad()
            logits = model(encoded_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)

        train_writer.add_scalar('loss', avg_loss, epoch)

        pbar.set_postfix(loss=avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}", flush=True)

        # eval
        if (epoch + 1) % 10 == 0:
            model.eval()
            for queries, labels, rephrases, locs in dataloader:
                labels = labels.to(device).squeeze()
                encoded_inputs = BertTokenizer.from_pretrained('bert-base-uncased')(
                    queries, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    logits = model(encoded_inputs)
                probs = F.softmax(logits, dim=-1)
                train_predictions = torch.argmax(probs, dim=1)
                # correct
                train_correct += (train_predictions == labels).sum().item()
                entropy = calculate_entropy(probs)
                # high_entropy
                high_entropy_labels = entropy > entropy_threshold
                predicted_labels = torch.where(high_entropy_labels, torch.tensor(-1), torch.argmax(probs, dim=-1))
                train_entropy_correct += sum(1 for label in predicted_labels if label != -1)

                train_writer.add_scalar('ori_acc', train_correct / total, epoch)
                train_writer.add_scalar('entropy_acc', train_entropy_correct/total, epoch)
                train_writer.add_scalar('entropy', torch.mean(entropy).item(), epoch)

                with torch.no_grad():
                    rephrases_encoded_inputs = BertTokenizer.from_pretrained('bert-base-uncased')(
                        rephrases, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                    rephrases_logits = model(rephrases_encoded_inputs)
                probs = F.softmax(rephrases_logits, dim=-1)
                test_predictions = torch.argmax(probs, dim=1)
                test_correct += (test_predictions == labels).sum().item()
                entropy = calculate_entropy(probs)
                high_entropy_labels = entropy > entropy_threshold
                predicted_labels = torch.where(high_entropy_labels, torch.tensor(-1), torch.argmax(probs, dim=-1))
                test_entropy_correct += sum(1 for label in predicted_labels if label != -1)
                rephrase_writer.add_scalar('entropy', torch.mean(entropy).item(), epoch)
                
                with torch.no_grad():
                    locs_encoded_inputs = BertTokenizer.from_pretrained('bert-base-uncased')(
                        locs, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                    locs_logits = model(locs_encoded_inputs)
                probs = F.softmax(locs_logits, dim=-1)
                entropy = calculate_entropy(probs)
                high_entropy_labels = entropy > entropy_threshold
                predicted_labels = torch.where(high_entropy_labels, torch.tensor(-1), torch.argmax(probs, dim=-1))
                loc_entropy_correct += sum(1 for label in predicted_labels if label == -1)
                loc_writer.add_scalar('entropy', torch.mean(entropy).item(), epoch)

                rephrase_writer.add_scalar('ori_acc', test_correct / total, epoch)
                rephrase_writer.add_scalar('entropy_acc', test_entropy_correct/total, epoch)
                loc_writer.add_scalar('entropy_acc', loc_entropy_correct/total, epoch)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save familiar network
    partial_state_dict = {k: v for k, v in model.state_dict().items() if any(layer in k for layer in layers_to_save)}
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': partial_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_path, f"{model_name}_{data_type}_{data_size}.pth"))
    print(f"Model saved at epoch {epoch + 1}")
    print("Training complete.")


def get_config(model_name, data_type, edit_size):
    yaml_data = './hparams/stage_2/config.yaml'
    with open(yaml_data, "r") as f:
        data = yaml.safe_load(f)
    """get config"""
    task_data = data["models"][model_name][data_type]
    for edit in task_data["edits"]:
        if edit["edit_size"] == edit_size:
            return {
                "batch_size": edit["batch_size"],
                "epoch": edit["epoch"],
                "threshold": task_data["threshold"]
            }
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train familiarity network with configurable parameters")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--model_type", type=str, default="phi2", choices=["gptj", "phi2"], help="Model parameter type (e.g., gptj)")
    parser.add_argument("--data_type", type=str, default="zsre", choices=["zsre", "cf"], help="Data type")
    parser.add_argument("--data_size", type=int, default=1024, help="edit data size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    model_name = args.model_type
    data_type = args.data_type
    data_size = args.data_size

    # checkpoint path for Text Encoder
    checkpoint_path = f"checkpoints_trained_bert/bert_{model_name}_{data_type}"

    config = get_config(model_name, data_type, data_size)

    print(config)

    save_path = "./familiar_network/checkpoints/"
    log_save_path = f"./familiar_network/train_log/{model_name}_{data_type}_{data_size}/"+'{}'
    
    set_seed(42)
    model = train_model(model_name,log_save_path,data_type,checkpoint_path,save_path,
                        data_size, args.gpu, config['epoch'], config['batch_size'], args.lr, config['threshold'])
