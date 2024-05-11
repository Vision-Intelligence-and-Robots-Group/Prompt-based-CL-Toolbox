# !/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --master_port=29501 \
# --use_env main.py l2p_cifar100

# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --master_port=29502 \
# --use_env main.py dualp_cifar100

# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --master_port=29501 \
# --use_env main.py l2p_imagenetr

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29501 \
# --use_env main.py dualp_imagenetr

python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port=29502 \
--use_env main.py dualp_core50

# python -m torch.distributed.launch \
# --nproc_per_node 1 \
# --master_port=29501 \
# --use_env main.py l2p_core50

# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --master_port=29500 \
# --use_env main.py sprompt_cddb_slip
