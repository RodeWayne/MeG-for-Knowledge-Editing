# For duplicate subjects, retain the first occurrence
# add an ID number to the remaining data
import json

data_path_1 = './raw_data/zsre_mend_eval.json'
data_path_2 = './raw_data/zsre_mend_train_10000.json'
data_path_3 = './raw_data/zsre_mend_train.json'

output_path_1 = './new_split_data/zsre_mend_eval_new.json'
output_path_2 = './new_split_data/zsre_train_new.json'

with open(data_path_1, "r") as f:
    data_1 = json.load(f)

with open(data_path_2, "r") as f:
    data_2 = json.load(f)

with open(data_path_3, "r") as f:
    data_3 = json.load(f)

# get new split of zsre data for eval(edit)
data = data_1 + data_2

new_data = []
num = 0
subjects = []

# For duplicate subjects, retain the first occurrence
for d in data:
    subject = d['subject']
    if subject in subjects:
        continue
    subjects.append(d['subject'])
    d['id'] = num
    new_data.append(d)
    num += 1

with open(output_path_1, 'w') as file:
    json.dump(new_data, file, indent=4)

print('thera are still {} items'.format(len(new_data)))

# get new split of zsre data for training bert and ddp 
new_data_1 = []
map_src_data = []
for d in data_2:
    map_src_data.append(d['src'])
# remove data appeared in eval set
for d in data_3:
    if d['src'] not in map_src_data:
        new_data_1.append(d)

num = 0
subjects = []
new_data_2 = []
# For duplicate subjects, retain the first occurrence
for d in new_data_1:
    subject = d['subject']
    if subject in subjects:
        continue
    subjects.append(d['subject'])
    d['id'] = num
    new_data_2.append(d)
    num += 1

with open(output_path_2, 'w') as file:
    json.dump(new_data_2, file, indent=4)