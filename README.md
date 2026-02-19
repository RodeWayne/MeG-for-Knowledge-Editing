# MeG-for-Knowledge-Editing
This is the official implementation of the ICLR-2026 accepted paper "Massive Editing for Large Language Models Based on Dynamic Weight Generation."



## Requirements

First, download and set up the repo:

```setup
git clone https://github.com/RodeWayne/MeG-for-Knowledge-Editing.git
cd MeG-for-Knowledge-Editing
```
We provide an environment.yml file that can be used to create a Conda environment. 

```setup
conda env create -f environment.yml
conda activate Meg
```
## Training

To train the model(s) in the paper, run this command:

#### Stage 0: Data Prepare:

download data to folder ```./data```

if you want to know how we get our splited data from raw data, please run:
```cmd
cd data
bash split_data.sh
```

to get data for editing, run
```cmd
python get_edit_and_loc_data.py --model_type <model name> --data_type <dataset name>
```

#### Stage 1: Text Encoder Training:

To prepare corresponding YAML configuration files in `/hparams/stage_1/` directory, and then run this command:

```cmd
python train_bert.py --hparams hparams/stage_1/<model_data>.yaml
```

#### Stage 2: Familiarity Network Training:
```python
# add fake id to data for Familiarity Network Training
python add_fake_id_for_familiar.py
# training
python train_familiar.py --model_type <model name> --data_type <dataset name> --data_size <edit size>
```

#### Stage 3: Neuron Weight Training:
```cmd
python train_neuron.py --model_type <model name> --data_type <dataset name> --data_size <edit size>
```

#### Stage 4: Weight-Generation Model Training

To prepare corresponding YAML configuration files in /hparams/stage_4/ directory, and then run this command:

```cmd
python train.py  --hparams hparams/stage_4/<model_data_size>.yaml
```
For datasets with more than 1024 samples, it is recommended to lauch training with N GPUs on one node:
```cmd
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16  train_ddp.py --hparams hparams/stage_4/<model_data_size>.yaml
```

## Evaluation

To prepare corresponding YAML configuration files in `/evaluate/hparams/` directory, and then run this command:

```eval
python evaluate/run_all_evaluation.py --model_para_type phi2 --data_type zsre
```

To evaluate using N GPUs on one node, configure `Run_evaluation_parallel.sh`.
The value for `TOTAL_DATA` will be passed to `config.fi` at runtime. Therefore, you do not need to manually set it when using parallel evaluation.
Run this commmand to start the parallel evaluation:
```eval
./Run_evaluation_parallel.sh
```
A new `Output` folder will be created after running this command. This folder will store the complete running logs for each parallel process.
