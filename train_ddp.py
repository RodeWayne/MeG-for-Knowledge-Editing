# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script bu  t True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from my_model_five_bert_text import myDiT
from diffusion import create_diffusion
from datetime import datetime
from myDataloader_cf_bert_text_add_nq_lr import MyDataset as MyDataset_cf
from myDataloader_bert_text_add_nq_lr import MyDataset as MyDataset_zsre
# from myDataloader_phi2_hidden import MyDataset

import json
import socket
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # 1. 准备日志
    if rank == 0:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        now = datetime.now()
        time_format_1 = now.strftime("%Y%m%d%H%M%S")  # 年月日时分秒，例如：20240831123045
        experiment_index = len(glob(f"{results_dir}/*"))
        experiment_dir = f"{results_dir}/{experiment_index:03d}_{time_format_1}"  # Create an experiment folder
        checkpoint_root_dir = f"/home/wentao/xzw/checkpoint/{experiment_index:03d}_{time_format_1}"  # Create an experiment folder
        checkpoint_dir = f"{checkpoint_root_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
    else:
        logger = create_logger(None)
    # 4. 配置基础模型
    seq_len = args.seq_len  # 5121 8193
    seq_len_y = 768
    patchsize = args.ps
    # 384 768
    denoise = myDiT(seq_len=seq_len, patch_size=patchsize, hidden_size=args.hidden, num_heads=args.nheads,
                    num_blocks=args.nblocks).to(device)
    denoise = denoise.to(device)
    denoise = DDP(denoise.to(device), device_ids=[rank],find_unused_parameters=True)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    opt = torch.optim.AdamW(denoise.parameters(), lr=args.lr, weight_decay=0)

    # 尝试加载检查点
    start_epoch = 0
    # save_filename="/home/wentao/xzw/checkpoint/temp/186_026_20250205001625/checkpoints/model_epoch_all_30000.pth"
    #
    # checkpoint = torch.load(save_filename, map_location=torch.device("cuda", rank))
    # denoise.module.load_state_dict(checkpoint["model_state_dict"])
    # opt.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint["epoch"] + 1
    # print(f"Resumed from checkpoint: {save_filename} (Epoch {start_epoch})")
    # del checkpoint

    # 3. 配置数据
    # paras_dir = '/home/wentao/xzw/data_paras/method_1_6/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/method_1_6/train_success_data_phi2_prompt_1_6_neuron_1_layer_29_len_17203.json'
    # paras_dir = '/home/wentao/xzw/data_paras/data_phi2/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/data_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29_len_16987.json'

    # paras_dir = '/home/wentao/xzw/data_paras/cf_layer_9/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/cf_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json'

    # phi2 cf zsre 1
    # paras_dir = '/home/wentao/xzw/data_paras/cf_phi2_epoch_80/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/cf_phi2_epoch_80/multi_counterfact_new_id.json'

    # 最终版本： phi zsre 10000
    # paras_dir = '/home/wentao/xzw/data_paras/zsre_phi2/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/zsre_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29.json'

    # 最终版本： gptj zsre 10000
    # paras_dir = '/home/wentao/xzw/data_paras/zsre_gptj/zsre_layer_9/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/zsre_gptj/zsre_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json'

    # 最终版本： phi2 cf 10000
    # paras_dir = '/home/wentao/xzw/data_paras/cf_phi2/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/cf_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29.json'

    # 最终版本： gptj cf 10000
    # paras_dir = '/home/wentao/xzw/data_paras/cf_gptj/cf_layer_9/all'
    # rephrases_dir = '/home/wentao/xzw/data_paras/cf_gptj/cf_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json'
    paras_dir = args.paras_dir
    rephrases_dir = args.rephrases_dir
    if args.data_type == "cf":
        dataset = MyDataset_cf(args, paras_dir, rephrases_dir, gpu=device, is_noise=args.is_noise,
                               noisetype=args.noisetype, model_para_type=args.model_para_type, layer=args.layer,
                               fileindex=args.fi)
    elif args.data_type == "zsre":
        dataset = MyDataset_zsre(args, paras_dir, rephrases_dir, gpu=device, is_noise=args.is_noise,
                                 noisetype=args.noisetype, model_para_type=args.model_para_type, layer=args.layer,
                                 fileindex=args.fi)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    print("batchsize:"+str(int(args.global_batch_size // dist.get_world_size())))
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        # num_workers=args.num_workers,
        # pin_memory=True,
        # drop_last=True
    )
    ###=================展示配置
    if rank == 0:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = current_dir + "/" + experiment_dir + "/" + "log.txt"
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"---->Begin training ,add y, Train {len(dataset)},  "
                    f"IP:{local_ip}, process_id:{os.getpid()}, "
                    f"log_path:{log_path}")
        args_dict = vars(args)
        json_output = json.dumps(args_dict, indent=4)
        logger.info(f"training setting:\n{json_output}")
        showName = f"{time_format_1}_ddp_dit_1_6_ddp_addnqlr_initp_seed_droup_berttextClsOut2para_ip{local_ip.split('.')[-1]}_case{len(dataset)}_ps{args.ps}_h{args.hidden}_nb{args.nblocks}_nh{args.nheads}_lr4"
        server_info = {
            "name": showName,
            "hostname": local_ip,
            "username": "wentao",
            "password": "wentao@123",
            "log_path": log_path
        }
        logger.info(f"show setting:\n{json.dumps(server_info, indent=4)}")
    ###=================

    # Variables for monitoring/logging purposes:
    n_steps = 1000  # number of denoising time steps
    train_steps = 0
    log_steps = 0
    running_loss = 0
    # start_time = time()

    top3=[]
    for epoch in range(start_epoch,args.epochs):
        sampler.set_epoch(epoch)
        denoise.train()
        epoch_loss = 0
        num_batches = 0  # 记录 batch 数
        for x, y in loader:
            # if rank == 0:
            # print(f"epoch:{epoch}, rank:{rank}, x_shape:{x.shape}, y_shape:{y.shape}")
            x = x.to(device)  # 1*1*5121
            y = y.to(device)  # 1*2560
            x_src = x
            y_src = y
            x0 = x_src / 0.01
            y0 = y_src
            t = torch.randint(0, n_steps, (x0.shape[0],)).to(device)  # Pick random time step
            model_kwargs = dict(y=y0)
            loss_dict = diffusion.training_losses(denoise, x0, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss_vb = loss_dict["vb"].mean()
            loss_mse = loss_dict["mse"].mean()
            print(f"epoch:{epoch}, rank:{rank}, x_shape:{x.shape}, y_shape:{y.shape}, loss: {loss}")


            opt.zero_grad()
            loss.backward()
            opt.step()
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            epoch_loss += loss.item()  # 累积 loss
            num_batches += 1  # 记录 batch 数
        if (epoch+1) % 100 == 0:
            # Reduce loss history over all processes:
            # avg_loss = torch.tensor(running_loss / log_steps, device=device)
            # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            # avg_loss = avg_loss.item() / dist.get_world_size()
            # logger.info(f"step:{epoch+1}; loss={avg_loss}; loss_vb:{loss_vb}; loss_mse:{loss_mse}; ")

            avg_epoch_loss = torch.tensor(epoch_loss / num_batches, device=device)
            dist.all_reduce(avg_epoch_loss, op=dist.ReduceOp.SUM)
            avg_epoch_loss = avg_epoch_loss.item() / dist.get_world_size()
            logger.info(f"step:{epoch+1}; loss={avg_epoch_loss}; loss_vb:{loss_vb}; loss_mse:{loss_mse}; ")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
        if (epoch + 1) % 10000 == 0:
            # 保存模型参数
            if rank == 0:
                save_checkpoint(checkpoint_dir, denoise, epoch, logger, opt)
            dist.barrier()
        if (epoch+1) % 1000 == 0:
            if rank == 0:
                denoise.eval()
                with  torch.no_grad():
                    sums=[]
                    for _ in range(3):
                        x_gen = torch.randn( 1, seq_len).to(device)
                        x_gen = x_gen.repeat(x_src.shape[0], 1)
                        model_kwargs=dict(y=y_src)
                        samples = diffusion.p_sample_loop(
                            denoise, x_gen.shape, x_gen, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                            device=device
                        )
                        x_recov = samples * 0.01
                        l2_norms = torch.norm(x_src - x_recov, p=2, dim=1)
                        mean_l2_norm = l2_norms.mean()
                        std_l2_norm = l2_norms.std()
                        sum=mean_l2_norm+std_l2_norm
                        sums.append(sum)
                        logger.info(f"step:{epoch+1};l2_norm:{l2_norms}")
                        logger.info(f"step:{epoch+1};mean_l2_norm:{mean_l2_norm};std_l2_norm:{std_l2_norm}")
                    # 保存lr2效果前三的checkpoint
                    sums_tensor = torch.tensor(sums)
                    # sum = sums_tensor.mean()
                    sum = round(sums_tensor.mean().item(), 4)  # 取 4 位小数

                    if len(top3) < 3:
                        top3.append([sum,epoch])  # 如果长度小于 3，直接添加
                        save_checkpoint(checkpoint_dir, denoise, epoch, logger, opt,str(sum))
                    else:
                        max_index = max(enumerate(top3), key=lambda x: x[1][0])[0]
                        max_value = top3[max_index][0]
                        old_epoch = top3[max_index][1]
                        if sum < max_value:
                            # 删除
                            old_checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_all_{old_epoch + 1}_sum{max_value}.pth')
                            os.remove(old_checkpoint_path)
                            # 替换最大值
                            top3[max_index] = [sum, epoch]
                            save_checkpoint(checkpoint_dir, denoise, epoch, logger, opt,str(sum))
            torch.distributed.barrier()  # 等待所有进程同步

    cleanup()

def save_checkpoint(checkpoint_dir, denoise, epoch, logger, opt,name=None):
    save_filename = os.path.join(checkpoint_dir, f'model_epoch_all_{epoch + 1}.pth')
    if name is not None:
        save_filename = os.path.join(checkpoint_dir, f'model_epoch_all_{epoch + 1}_sum{name}.pth')
    # torch.save(denoise.module.state_dict(), save_filename)
    torch.save({
        "model_state_dict": denoise.module.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
    }, save_filename)
    logger.info(f"epoch:{epoch};save checkpoint at {save_filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ## 固定配置
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--ps", type=int, default=100)
    parser.add_argument("--nheads", type=int, default=12)
    parser.add_argument("--nblocks", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--noisepara_dir", type=str, default='/home/wentao/xzw/data_paras/my_noise_paras')
    parser.add_argument("--is_bert_norm", action="store_true",default=True)
    # 动态固定
    parser.add_argument("--bertft_dir", type=str, default='/home/wentao/xzw/LLM/bert_phi2_zsre_checkpoints_infoNCE_case2/model_epoch_100.pth')
    parser.add_argument("--paras_dir", type=str, default='/home/wentao/xzw/data_paras/zsre_phi2/all')
    parser.add_argument("--rephrases_dir", type=str,default='/home/wentao/xzw/data_paras/zsre_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29.json')
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--seq_len", type=int, default=5121)  # 5121 8193
    parser.add_argument("--isbert_0or1", type=int, default=1)
    ## noise para
    parser.add_argument("--is_noise", action="store_true")
    parser.add_argument("--noisetype", type=int, default=0)
    parser.add_argument("--noise_n1024", type=float, default=0.2)
    parser.add_argument("--noisetype_10or2", type=int, default=1)
    ## 超参配置
    parser.add_argument("--model_para_type", type=str, default='gptj')
    parser.add_argument("--data_type", type=str, default='zsre')
    parser.add_argument("--gpus", type=str, default='3')
    parser.add_argument("--global_batch_size", type=int, default=2392)
    parser.add_argument("--fi", type=int, default=10000)


    # 旧的case：12=10 57=50 112=100 299=200 599=500 1185=1000
    # 新的case(seed)：12=10 76=50 156=100 318=200 809=500 1674=1000 8209=5000 14531=9000 16109=10000
    # 新的case(noseed)：18=10 1725=1000 8803=5000 15883=9000
    # 新的 1713=1024 3424=2048  6773=4096  13207=8192
###### #
    args = parser.parse_args()
    path_config = {
        "gptj": {
            "zsre": {
                "paras_dir": "/home/wentao/xzw/data_paras/zsre_gptj/zsre_layer_9/all",
                "rephrases_dir": "/home/wentao/xzw/data_paras/zsre_gptj/zsre_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json",
                "bertft_dir": "/home/wentao/xzw/LLM/bert_gptj_zsre_checkpoints_infoNCE_case2/model_epoch_100.pth",
                "layer": 9,
                "seq_len": 8193,
            },
            "cf": {
                "paras_dir": "/home/wentao/xzw/data_paras/cf_gptj/cf_layer_9/all",
                "rephrases_dir": "/home/wentao/xzw/data_paras/cf_gptj/cf_layer_9/train_success_data_gptj_prompt_3_6_neuron_1_layer_9.json",
                "bertft_dir": "/home/wentao/xzw/LLM/bert_gptj_cf_checkpoints_infoNCE/model_epoch_9600.pth",
                "layer": 9,
                "seq_len": 8193,
            }
        },
        "phi2": {
            "zsre": {
                "paras_dir": "/home/wentao/xzw/data_paras/zsre_phi2/all",
                "rephrases_dir": "/home/wentao/xzw/data_paras/zsre_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29.json",
                "bertft_dir": "/home/wentao/xzw/LLM/bert_phi2_zsre_checkpoints_infoNCE_case2/model_epoch_100.pth",
                "layer":29,
                "seq_len":5121,
            },
            "cf": {
                "paras_dir": "/home/wentao/xzw/data_paras/cf_phi2/all",
                "rephrases_dir": "/home/wentao/xzw/data_paras/cf_phi2/train_success_data_phi2_prompt_1_6_neuron_1_layer_29.json",
                "bertft_dir": "/home/wentao/xzw/LLM/bert_phi2_cf_checkpoints_infoNCE/model_epoch_5100.pth",
                "layer": 29,
                "seq_len": 5121,
            }
        }
    }
    args.paras_dir = path_config[args.model_para_type][args.data_type]["paras_dir"]
    args.rephrases_dir = path_config[args.model_para_type][args.data_type]["rephrases_dir"]
    args.bertft_dir = path_config[args.model_para_type][args.data_type]["bertft_dir"]
    args.layer = path_config[args.model_para_type][args.data_type]["layer"]
    args.seq_len = path_config[args.model_para_type][args.data_type]["seq_len"]
    main(args)
