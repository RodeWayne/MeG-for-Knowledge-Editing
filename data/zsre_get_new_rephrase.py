import re
import json
from tqdm import *
import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = "cuda"

def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

model,tokenizer = get_model("THUDM/glm-4-9b-chat")

model.to(device)
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

with open('./raw_data/get_rephrase.prompt') as f:
    prompt = f.read().strip()

data_path = './new_split_data/zsre_mend_eval_new.json'

output_path = './new_split_data/zsre_mend_eval_new_rephrase.json'

with open(data_path, "r") as f:
    data = json.load(f)

pattern = r'<list>\s*(\[.*?\])\s*</list>'

num = 0

error = {}
for d in tqdm(data):
    num += 1
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt.format(d['src'])}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to(device)
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]

    res = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    try:
        result = re.search(pattern, res, re.DOTALL)

        match = result.group(1)
        
        res_rephrase = eval(match)

        if d['src'] in res_rephrase:
            res_rephrase.remove(d['src'])

        d['rephrase'] = res_rephrase[:10]

        if num == 10 or num % 200 == 0:
            with open(output_path, 'w') as file:
                json.dump(data, file, indent=4)

    except Exception as e:
        error[d['id']] = e


with open(output_path, 'w') as file:
    json.dump(data, file, indent=4)
