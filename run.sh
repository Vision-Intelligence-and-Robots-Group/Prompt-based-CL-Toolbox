# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --master_port=29500 \
# --use_env main.py sprompt_cddb_slip

# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --master_port=29501 \
# --use_env main.py l2p_cifar100

python -m torch.distributed.launch \
--nproc_per_node 2 \
--master_port=29502 \
--use_env main.py dualp_cifar100
