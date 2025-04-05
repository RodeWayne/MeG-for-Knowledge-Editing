# For duplicate subjects, retain the first occurrence
# add an ID number to the remaining data
import json

data_path_1 = './raw_data/zsre_mend_eval.json'
data_path_2 = './raw_data/zsre_mend_train_10000.json'

output_path = './final_use_data/temp/zsre_mend_eval_new.json'

with open(data_path_1, "r") as f:
    data_1 = json.load(f)

with open(data_path_2, "r") as f:
    data_2 = json.load(f)

data = data_1 + data_2

new_data = []
num = 0
subjects = []

for d in data:
    subject = d['subject']
    if subject in subjects:
        continue
    subjects.append(d['subject'])
    d['id'] = num
    new_data.append(d)
    num += 1

with open(output_path, 'w') as file:
    json.dump(new_data, file, indent=4)

print('thera are still {} items'.format(len(new_data)))