import os
import sys
import logging
import time
import torch
from data.data_manager import DataManager
from utils.vis_utils import draw_acc_hot, draw_acc_line
from utils.dist_utils import get_rank, init_distributed_mode,is_main_process
import numpy as np
import random

from methods.l2p_engine import L2P
from methods.dualp_engine import DualPrompt
from methods.sprompt_engine import SPrompts
from methods.apil_engine import APIL

def get_model(model_name, args):
    name = model_name.lower()
    options = {
        'l2p': L2P,
        'dualp':DualPrompt,
        'sprompt':SPrompts,
        'apil':APIL,
    }
    return options[name](args)

def train(args):
    setup_logs(args)
    init_distributed_mode(args)
    set_random(args)
    _train(args)

def _train(args):
    data_manager = DataManager(args.dataset, args.shuffle, args.seed, args.init_cls, args.increment, args)
    args.class_order = data_manager._class_order
    model = get_model(args.engine_name, args)
    data_manager.build_continual_dataloader()

    for task_id in range(0 ,data_manager.nb_tasks):
        train_loader = data_manager.continual_data[task_id]['train']
        test_loader = data_manager.continual_data[task_id]['test']
        
        model.incremental_train(train_loader, test_loader, data_manager, task_id)
        if is_main_process():
            model.eval_task(test_loader)  
        model.after_task()
        
        if is_main_process():
            if args.save_checkpoints:
                torch.save(model, os.path.join(args.logfilename, 'checkpoints', "task_{}_{}_{}.pth".format(int(task_id),args.dataset,args.engine_name)))
            if args.save_vis:
                draw_acc_hot(args.num_tasks,args.init_cls,args.increment,model.acc_table_cnn,args.logfilename)
                draw_acc_line(args.dataset,args.engine_name,args.num_tasks,model.acc_table_cnn,args.logfilename)
                if hasattr(model, '_class_means'):
                    draw_acc_hot(args.num_tasks,args.init_cls,args.increment,model.acc_table_nme,args.logfilename)
                    draw_acc_line(args.dataset,args.engine_name,args.num_tasks,model.acc_table_nme,args.logfilename)  
                
def setup_logs(args):
    args.localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logfilename = 'logs/{}_{}_{}_{}_{}_{}_{}_'.format(args.prefix, args.seed, args.engine_name, args.net_type,
                                                args.dataset, args.init_cls, args.increment)+ args.localtime

    args.logfilename = logfilename
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
    if is_main_process():
        os.makedirs(logfilename)
        print(logfilename)
        if args.save_checkpoints:
            os.makedirs(os.path.join(logfilename, 'checkpoints'), exist_ok=True)
        if args.save_vis:
            os.makedirs(os.path.join(logfilename, 'vis'), exist_ok=True)
    
    # print args
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))

def set_random(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False