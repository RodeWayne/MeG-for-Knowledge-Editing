import json
from util import *
import argparse
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

def test(model,tokenizer,query_prompt,train_item,dataset):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(query_prompt, return_tensors="pt", return_attention_mask=False).to(device)
        max_length = inputs['input_ids'].shape[1] + 20
        outputs = model.generate(**inputs, max_length=max_length)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        if dataset == 'zsre':
            label = train_item["answers"][0]
        if dataset == 'cf':
            label = train_item['requested_rewrite']['target_new']['str']

        label = ' ' + label
        prediction = adjust_dots(prediction.rstrip())

    return label,prediction

def save_param(save_path, model):
    os.makedirs(save_path, exist_ok=True)
    model_state_dict = model.state_dict()
    selected_layers_state_dict = {}
    for layer_name, params in model_state_dict.items():
        if model_name == 'phi2':
            if layer_name == 'model.layers.{}.mlp.fc1.weight'.format(layer):
                selected_layers_state_dict[layer_name] = params[-neuron_num:, :].tolist()
            elif layer_name == 'model.layers.{}.mlp.fc1.bias'.format(layer):
                selected_layers_state_dict[layer_name] = params[-neuron_num:].tolist()
            elif layer_name == 'model.layers.{}.mlp.fc2.weight'.format(layer):
                selected_layers_state_dict[layer_name] = params[:, -neuron_num:].tolist()

        if model_name == 'gptj':
            if layer_name == 'transformer.h.{}.mlp.fc_in.weight'.format(layer):
                selected_layers_state_dict[layer_name] = params[-neuron_num:, :].tolist()
            elif layer_name == 'transformer.h.{}.mlp.fc_in.bias'.format(layer):
                selected_layers_state_dict[layer_name] = params[-neuron_num:].tolist()
            elif layer_name == 'transformer.h.{}.mlp.fc_out.weight'.format(layer):
                selected_layers_state_dict[layer_name] = params[:, -neuron_num:].tolist()

    with open(os.path.join(save_path, 'params_0.json'), 'w') as file:
        json.dump(selected_layers_state_dict, file, indent=4)


def train_once(model, tokenizer, train_item, initial_layer_1_add_weight, initial_layer_1_add_bias, initial_layer_2_add_weight, 
               epochs, model_name, dataset, device, save_root_path, method, neuron_num, layer, config):

    query_method = int(method.split('_')[0])
    answer_method = int(method.split('_')[1])

    if model_name == 'phi2':
        with torch.no_grad():
            model.model.layers[layer].mlp.fc1.weight[-neuron_num:, :] = initial_layer_1_add_weight
            model.model.layers[layer].mlp.fc1.bias[-neuron_num] = initial_layer_1_add_bias
            model.model.layers[layer].mlp.fc2.weight[:, -neuron_num:] = initial_layer_2_add_weight

    if model_name == 'gptj':
        patience_counter = 0
        patience_epochs = config['patience_epochs']
        min_loss_threshold = config['min_loss_threshold']
        patience = config['patience']  # the number of epochs where the loss is continuously less than the min_loss_threshold

        with torch.no_grad():
            model.transformer.h[layer].mlp.fc_in.weight[-neuron_num:, :] = initial_layer_1_add_weight
            model.transformer.h[layer].mlp.fc_in.bias[-neuron_num:] = initial_layer_1_add_bias
            model.transformer.h[layer].mlp.fc_out.weight[:, -neuron_num:] = initial_layer_2_add_weight


    if dataset == 'zsre':
        query_prompt = query_prompt_dict[query_method].format(train_item['src'])
        answer_prompt = answer_prompt_dict[answer_method].format(train_item["answers"][0])
    if dataset == 'cf':
        requested_rewrite = train_item['requested_rewrite']
        query_prompt = query_prompt_dict[query_method].format(requested_rewrite['prompt'].format(requested_rewrite['subject']))
        answer_prompt = answer_prompt_dict[answer_method].format(requested_rewrite['target_new']['str'])

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=0)
    if model_name == 'gptj':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['scheduler_T_max'], eta_min=0)

    single_input_ids = tokenizer.encode(query_prompt + answer_prompt, return_tensors='pt').to(device)
    set100 = len(tokenizer.encode(query_prompt))
    labels = single_input_ids.tolist()[0]
    labels[0:set100] = [-100]*set100
    labels_tensor = torch.tensor([labels])
    single_labels = labels_tensor

    model.to(device)
    
    epoch = 0
    success = False

    log_save_path = os.path.join(save_root_path+'_log', 'data_id_{}'.format(train_item["id"]))
    writer = SummaryWriter(log_dir=log_save_path)

    while(not success):
        epoch += 1
        model.train()
        optimizer.zero_grad()
        res = model(input_ids=single_input_ids,labels=single_labels)
        loss = res.loss
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss', loss.item(), epoch)

        if model_name == 'phi2':
            if epoch % epochs == 0:
                label,prediction = test(model,tokenizer,query_prompt,train_item,dataset)
                if label.lower() == prediction.lower():
                    success = True
            if epoch == epochs * 2:
                break
        
        if model_name == 'gptj':
            scheduler.step()
            if loss.item() < min_loss_threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    label,prediction = test(model,tokenizer,query_prompt,train_item,dataset)
                    if label.lower() == prediction.lower():
                        success = True
                        break
                    # test fail, continue to train
                    else:
                        print(f"Test fail, will continue for another {patience_epochs} epochs.")
                        for new_epoch in range(patience_epochs):
                            model.train()
                            optimizer.zero_grad()
                            res = model(input_ids=single_input_ids,labels=single_labels)
                            loss = res.loss
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            writer.add_scalar('loss', loss.item(), epoch+new_epoch+1)
                        
                        label,prediction = test(model,tokenizer,query_prompt,train_item,dataset)
                        if label.lower() == prediction.lower():
                            success = True
                        break

    if success:
        save_path = os.path.join(save_root_path, 'data_id_{}'.format(train_item["id"]))
        save_param(save_path, model)
        if dataset == 'zsre':
            return {'id':train_item['id'], 'query':train_item['src'], 'label':train_item["answers"][0],
                    'wrong_answer':train_item['pred'], 'rephrase':train_item['rephrase'],'loc': train_item['loc'],
                    'loc_ans':train_item['loc_ans'], 'loc_pred':train_item['loc_pred']}
        if dataset == 'cf':
            return {'id':train_item['id'], 'query':requested_rewrite['prompt'].format(requested_rewrite['subject']), 'label':requested_rewrite['target_new']['str'],
                'target_true':requested_rewrite['target_true']['str'], 'rephrase':train_item['paraphrase_prompts'],'loc': train_item['neighborhood_prompts'],}
    return None


def get_config(model_name, data_type):
    yaml_data = './hparams/stage_3/config.yaml'
    with open(yaml_data, "r") as f:
        data = yaml.safe_load(f)
    """get config"""
    task_data = data["models"][model_name][data_type]
    return task_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--model_type", type=str, default="phi2", choices=["gptj", "phi2"], help="Model parameter type (e.g., gptj)")
    parser.add_argument("--data_type", type=str, default="zsre", choices=["zsre", "cf"], help="Data type")
    parser.add_argument('--data_size', type=int, default=10000)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    model_name = args.model_type
    edit_size = args.data_size
    dataset = args.data_type

    config = get_config(model_name, dataset)

    print(config)
    
    data_path = config['data_path']
    epochs = config['epochs']
    method = config['method']
    neuron_num=config['neuron_num']
    layer = config['layer']

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_root_path = './data_paras/{}_{}_{}/paras'.format(model_name, dataset, edit_size)

    os.makedirs(save_root_path, exist_ok=True)

    set_seed(42)

    # set model
    if model_name == 'phi2':
        model, tokenizer = initial_phi2_model(neuron_num, layer)
        model = freeze_phi2(model,layer)

        initial_layer_1_add_weight = model.model.layers[layer].mlp.fc1.weight[-neuron_num:, :].clone().detach()
        initial_layer_1_add_bias = model.model.layers[layer].mlp.fc1.bias[-neuron_num].clone().detach()
        initial_layer_2_add_weight = model.model.layers[layer].mlp.fc2.weight[:, -neuron_num:].clone().detach()


    if model_name == 'gptj':
        model, tokenizer = initial_gptj_model(neuron_num, layer)
        model = freeze_gptj(model,layer)

        initial_layer_1_add_weight = model.transformer.h[layer].mlp.fc_in.weight[-neuron_num:, :].clone().detach()
        initial_layer_1_add_bias = model.transformer.h[layer].mlp.fc_in.bias[-neuron_num].clone().detach()
        initial_layer_2_add_weight = model.transformer.h[layer].mlp.fc_out.weight[:, -neuron_num:].clone().detach()

    model.to(device)
    
    # get edit data
    with open(data_path, "r") as f:
        edit_data = json.load(f)
    
    if edit_size is not None:
        edit_data = edit_data[:edit_size]

    new_data = []
    for train_item in tqdm(edit_data, total=len(edit_data)):
        new_item = train_once(
                        model, tokenizer,train_item, initial_layer_1_add_weight, initial_layer_1_add_bias, initial_layer_2_add_weight, 
                        epochs, model_name, dataset, device, save_root_path, method, neuron_num, layer, config)
        if new_item is not None:
            new_data.append(new_item)
        if len(new_data) == 10 or len(new_data) % 100 == 0:
            with open(os.path.join(save_root_path.rsplit('/',1)[0], 'train_success_data_prompt_{}_neuron_{}_layer_{}.json'.format(method, neuron_num, layer)), 'w') as file:
                json.dump(new_data, file, indent=4)
    with open(os.path.join(save_root_path.rsplit('/',1)[0], 'train_success_data_prompt_{}_neuron_{}_layer_{}.json'.format(method, neuron_num, layer)), 'w') as file:
        json.dump(new_data, file, indent=4)
