import json
import os
import random

os.makedirs('./data/train_familiar', exist_ok=True)

data_dir = './data/new_split_data/multi_counterfact_new_id.json'
with open(data_dir, "r") as f:
    raw_data = json.load(f)
raw_data = raw_data[0:10000]

for d in raw_data:
    d['fake_label'] = random.randint(0, 9)

with open('./data/train_familiar/cf_fake_id_class.json','w') as file:
    json.dump(raw_data, file, indent=4) 


data_dir = './data/edit_data/gptj_zsre_edit.json'
with open(data_dir, "r") as f:
    raw_data = json.load(f)
raw_data = raw_data[0:10000]

for d in raw_data:
    d['fake_label'] = random.randint(0, 9)

with open('./data/train_familiar/zsre_gptj_fake_id_class.json','w') as file:
    json.dump(raw_data, file, indent=4) 


data_dir = './data/edit_data/phi2_zsre_edit.json'
with open(data_dir, "r") as f:
    raw_data = json.load(f)
raw_data = raw_data[0:10000]

for d in raw_data:
    d['fake_label'] = random.randint(0, 9)

with open('./data/train_familiar/zsre_phi2_fake_id_class.json','w') as file:
    json.dump(raw_data, file, indent=4) 