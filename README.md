# MeG-for-Knowledge-Editing
This is the official implementation of paper "Massive Editing for Large Language Models Based on Dynamic Weight Generation"



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

Stage 0: Data Prepare:

download the data we final use to folder ```./data```

if you want to know how we get our data for final use from raw data, please run:
```cmd
cd data
bash filter_data.sh
```

Stage 1: Text Encoder Training:

To prepare corresponding YAML configuration files in `/hparams/stage_1/` directory, and then run this command:

```cmd
python train_bert.py --hparams hparams/stage_1/<model_data>.yaml
```

Stage 2: Familiarity Network Training:
```cmd
python train_familiar.py --gpu 0 --model_para_type phi2 --data_type zsre --data_size 1024 --epochs 1800 --batch_size 1024 --lr 0.001
```


Stage 4: Weight-Generation Model Training
```cmd
python train.py 
```
or lauch training with N GPUs on one node:
```cmd
torchrun --nnodes=1 --nproc_per_node=N train_ddp.py
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