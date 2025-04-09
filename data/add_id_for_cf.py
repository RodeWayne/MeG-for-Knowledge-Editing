import json
import os

data_path_1 = './raw_data/multi_counterfact.json'

output_path = './new_split_data/multi_counterfact_new_id.json'

os.makedirs('./new_split_data', exist_ok=True)

with open(data_path_1, "r") as f:
    data_1 = json.load(f)

num = 0
for d in data_1:
    d['id'] = num
    num+=1

with open(output_path, 'w') as file:
    json.dump(data_1, file, indent=4)
