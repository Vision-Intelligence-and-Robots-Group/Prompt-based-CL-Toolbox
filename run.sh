# !/bin/bash

# export CUDA_VISIBLE_DEVICES=1

# python -m torch.distributed.launch \
# --nproc_per_node [num_gpus] \
# --master_port=[your_port] \
# --use_env main.py l2p_cifar100

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29566 \
# --use_env main.py l2p_imagenetr

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29535 \
# --use_env main.py dualp_cifar100

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29505 \
# --use_env main.py dualp_imagenetr

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29500 \
# --use_env main.py apil_cifar100

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29519 \
# --use_env main.py sprompt_cddb_slip

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29577 \
# --use_env main.py sprompt_cddb_sip