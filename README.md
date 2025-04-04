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

Stage 1: Text Encoder Training:
```trainbert
python train_bert.py --gpu 0 --model_para_type phi2 --data_type zsre --temperature 1 --epochs 30000 --batch_size 4000 --lr 1e-4
```

Stage 4: Weight-Generation Model Training
```train
python train.py 
```
or lauch training with N GPUs on one node:
```train_ddp
torchrun --nnodes=1 --nproc_per_node=N train.py
```

## Evaluation

To prepare corresponding YAML configuration files in `/evaluate/hparams/` directory, and then run this command:

```eval
python evaluate/run_all_evaluation.py --model_para_type phi2 --data_type zsre
```

