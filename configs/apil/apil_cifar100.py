import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--prefix', default='best', type=str, help='')

    #dataset
    subparsers.add_argument('--dataset', default="cifar100_vit", type=str, help='')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--data_path', default="/home/pinna/data/cifar100", type=str, help='')
    subparsers.add_argument('--shuffle', default=True, type=bool, help='')
    
    subparsers.add_argument('--normalize_train', default=True, type=bool, help='')
    subparsers.add_argument('--normalize_test', default=True, type=bool, help='')
    subparsers.add_argument('--color_jitter', default=True, type=bool, help='')
    subparsers.add_argument('--pin_mem', default=True, type=bool, help='')

    # increment
    subparsers.add_argument('--init_cls', default=10, type=int, help='')
    subparsers.add_argument('--increment', default=10, type=int, help='')
    subparsers.add_argument('--num_tasks', default=10, type=int, help='')
    subparsers.add_argument('--scenario', default='class', type=str, help='')
    subparsers.add_argument('--extra_eval', default='', type=str, help='')

    # rehearsal
    subparsers.add_argument('--memory_size', default=0, type=int, help='')
    subparsers.add_argument('--memory_per_class', default=0, type=int, help='')
    subparsers.add_argument('--fixed_memory', default=True, type=bool, help='')

    # train model
    subparsers.add_argument('--engine_name', default="apil", type=str, help='')
    subparsers.add_argument('--net_type', default='vit', type=str, help='')
    subparsers.add_argument('--seed', default=1993, type=int, help='')
    subparsers.add_argument('--pretrained_model', default='vit_base_patch16_224', type=str, help='')
    subparsers.add_argument('--pretrained', default=True, type=bool, help='')
    subparsers.add_argument('--embd_dim', default=768, type=int, help='')
    subparsers.add_argument('--save_checkpoints', default=True, type=bool, help='')
    subparsers.add_argument('--save_vis', default=True, type=bool, help='')

    # training stage
    subparsers.add_argument('--batch_size', default=128, type=int, help='')
    subparsers.add_argument('--num_workers', default=16, type=int, help='')
    subparsers.add_argument('--EPSILON', default=1e-8, type=float, help='')
    subparsers.add_argument('--init_lr', default=0.001, type=float, help='')
    subparsers.add_argument('--lrate', default=0.001, type=float, help='')
    subparsers.add_argument('--init_epoch', default=5, type=int, help='')
    subparsers.add_argument('--epochs', default=5, type=int, help='')
    subparsers.add_argument('--init_weight_decay', default=0.0005, type=float, help='')
    subparsers.add_argument('--weight_decay', default=0.0002, type=float, help='')
    subparsers.add_argument('--T', default=2.0, type=float, help='')

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int, help='')
    subparsers.add_argument('--dist_url', default='env://', help='')
    subparsers.add_argument('--device', default='cuda', help='')
    subparsers.add_argument('--local_rank', default=0, type=int, help='')
    subparsers.add_argument('--distributed', default=True, type=bool, help='')

    # auxiliary
    subparsers.add_argument('--sched', default=False, type=bool, help='')
    subparsers.add_argument('--calculate_distributed', default=False, type=bool, help='')
    subparsers.add_argument('--unscale_lr', type=bool, default=False, help='')
    
    # anchor train
    subparsers.add_argument('--anchor_lr', default=1.0, type=float, help='')
    subparsers.add_argument('--anchor_epochs', default=5, type=int, help='')
    subparsers.add_argument('--anchor_num', default=5, type=int, help='')
    subparsers.add_argument('--cluster_num', default=6, type=int, help='')
    subparsers.add_argument('--sample_num', default=50, type=int, help='')
    subparsers.add_argument('--sigma', default=0.05, type=float, help='')
    
    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')