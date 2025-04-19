import re
import json
from tqdm import *
import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_rephrase(data, output_path):
    num = 0
    pattern = r'<list>\s*(\[.*?\])\s*</list>'
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
            pass

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)


def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = "cuda"

    model,tokenizer = get_model("THUDM/glm-4-9b-chat")

    model.to(device)
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

    with open('./raw_data/get_rephrase.prompt') as f:
        prompt = f.read().strip()

    eval_zsre_data_path = './new_split_data/zsre_mend_eval_new.json'

    bert_train_zsre_data_path = './new_split_data/zsre_train_new.json'

    eval_zsre_output_path = './new_split_data/zsre_mend_eval_new_rephrase.json'

    bert_train_zsre_output_path = './new_split_data/zsre_for_bert_train.json'

    # get rephrase
    with open(eval_zsre_data_path, "r") as f:
        eval_zsre_data = json.load(f)
    get_rephrase(eval_zsre_data,eval_zsre_output_path)


    with open(bert_train_zsre_data_path, "r") as f:
        bert_train_zsre_data = json.load(f)
        bert_train_zsre_data = bert_train_zsre_data[:4000]
    get_rephrase(bert_train_zsre_data,bert_train_zsre_output_path)