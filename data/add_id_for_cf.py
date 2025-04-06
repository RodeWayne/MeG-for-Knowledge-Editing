import json

data_path_1 = './raw_data/multi_counterfact.json'

output_path = './final_use_data/temp/multi_counterfact_new_id.json'

with open(data_path_1, "r") as f:
    data_1 = json.load(f)

num = 0
for d in data_1:
    d['id'] = num
    num+=1

with open(output_path, 'w') as file:
    json.dump(data_1, file, indent=4)
