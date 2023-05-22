import os
import os.path
import sys
import logging
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import numpy as np
import random

def train(args):
    setup_logs(args)
    init_distributed_mode(args)
    _set_random(args)
    _train(args)

def _train(args):
    data_manager = DataManager(args.dataset, args.shuffle, args.seed, args.init_cls, args.increment, args)
    args.class_order = data_manager._class_order
    model = factory.get_model(args.engine_name, args)
    data_manager.build_continual_dataloader()

    for task_id in range(0 ,data_manager.nb_tasks):
        train_loader = data_manager.continual_data[task_id]['train']
        test_loader = data_manager.continual_data[task_id]['test']
        
        model.incremental_train(train_loader, test_loader, data_manager, task_id)
        if model.local_rank == 0 :
            model.eval_task(test_loader)  
        model.after_task()

def setup_logs(args):
    args.localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logfilename = 'logs/{}_{}_{}_{}_{}_{}_{}_'.format(args.prefix, args.seed, args.engine_name, args.net_type,
                                                args.dataset, args.init_cls, args.increment)+ args.localtime

    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    print(logfilename)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.local_rank)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
        else:
            logging.disable(logging.CRITICAL)

    __builtin__.print = print

def _set_random(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False