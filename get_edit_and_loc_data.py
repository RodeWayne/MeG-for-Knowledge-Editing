import argparse
import json
import os 
from tqdm import tqdm
from util import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gptj", choices=["gptj", "phi2"], help="Model parameter type (e.g., gptj)")
    parser.add_argument("--data_type", type=str, default="cf", choices=["zsre", "cf"], help="Data type")
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()

def get_llm_response(model_name, model, tokenizer, query):
    with torch.no_grad():
        inputs = tokenizer(SHORT_ANSWER_PROMPT[model_name].format(query), return_tensors="pt", return_attention_mask=False).to(device = device)
        inputs.to(device)
        outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 20)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        if model_name == 'gptj':
            stop_sequence="\n"
            prediction = prediction.split(stop_sequence)[0]
        prediction = prediction.strip().rstrip('.')
    return prediction

if __name__ == '__main__':
    os.makedirs('./data/edit_data', exist_ok=True)
    set_seed(42)
    args = get_args()
    model_name = args.model_type
    dataset = args.data_type
    output_path = './data/edit_data/{}_{}_edit.json'.format(model_name,dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == 'zsre':
        data_path = './data/new_split_data/zsre_mend_eval_new_rephrase.json'
    
    if dataset == 'cf':
        data_path = './data/new_split_data/multi_counterfact_new_id.json'

    if model_name == 'phi2':
        model, tokenizer = get_model("microsoft/phi-2")

    if model_name == 'gptj':
        model, tokenizer = get_model("EleutherAI/gpt-j-6B")

    model.to(device)
    model.eval()

    with open(data_path, "r") as f:
        data = json.load(f)

    output_data = []

    num = 0

    for d in tqdm(data):
        if dataset == 'cf':
            # all data in counterfact dataset need edit
            # get loc ans
            neighborhood_prompts = d['neighborhood_prompts']
            loc_ans_dict = {}
            for loc_query in neighborhood_prompts:
                loc_ans_dict[loc_query] = get_llm_response(model_name, model, tokenizer, loc_query)
            d['neighborhood_prompts'] = loc_ans_dict

            output_data.append(d)
            num += 1
        elif dataset == 'zsre':
            query = d['src']
            prediction = get_llm_response(model_name, model, tokenizer, query)
            # answer is wrong, need edit
            if not target.lower() == prediction.lower():
                d['pred'] = prediction
                target = d['answers'][0]
                d['id'] = num
                # get loc ans 
                loc_query = (d['loc'].split('nq question: ')[1] + '?').capitalize()
                d['loc_pred'] = get_llm_response(model_name, model, tokenizer, loc_query)

                output_data.append(d)
                num += 1
        if num == 10 or num % 1000 == 0:
            with open(output_path, 'w') as file:
                json.dump(output_data, file, indent=4)

    with open(output_path, 'w') as file:
        json.dump(output_data, file, indent=4)
