import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--prefix', default='best', type=str, help='')

    #dataset
    subparsers.add_argument('--dataset', default="cddb", type=str, help='')
    subparsers.add_argument('--input_size', default=224, type=int, help='images input size')
    subparsers.add_argument('--data_path', default="/home/pinna/data/CDDB", type=str, help='')
    subparsers.add_argument('--shuffle', default=False, type=bool, help='')
    subparsers.add_argument('--normalize', default=True, type=bool, help='')
    subparsers.add_argument('--color_jitter', default=True, type=bool, help='')
    subparsers.add_argument('--pin_mem', default=True, type=bool, help='')

    #domain_option
    subparsers.add_argument('--class_order', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], type=list, help='')
    subparsers.add_argument('--task_name', default=["gaugan", "biggan", "wild", "whichfaceisreal", "san"], type=list, help='')
    subparsers.add_argument('--multiclass', default=[0, 0, 0, 0, 0], type=list, help='')

    # increment
    subparsers.add_argument('--init_cls', default=2, type=int, help='')
    subparsers.add_argument('--increment', default=2, type=int, help='')
    subparsers.add_argument('--num_tasks', default=7, type=int, help='')
    subparsers.add_argument('--scenario', default='domain', type=str, help='')
    subparsers.add_argument('--extra_eval', default='S-Prompt', type=str, help='')

    # rehearsal
    subparsers.add_argument('--memory_size', default=0, type=int, help='')
    subparsers.add_argument('--memory_per_class', default=0, type=int, help='')
    subparsers.add_argument('--fixed_memory', default=True, type=bool, help='')

    # train model
    subparsers.add_argument('--engine_name', default="sprompt", type=str, help='')
    subparsers.add_argument('--net_type', default='slip', type=str, help='')
    subparsers.add_argument('--seed', default=1993, type=int, help='')
    subparsers.add_argument('--pretrained_model', default='clip', type=str, help='')
    subparsers.add_argument('--pretrained', default=True, type=bool, help='')
    subparsers.add_argument('--embd_dim', default=768, type=int, help='')

    #prompt parameter
    subparsers.add_argument('--prompt_length', default=10, type=int, help='')

    # training stage
    subparsers.add_argument('--batch_size', default=128, type=int, help='')
    subparsers.add_argument('--num_workers', default=16, type=int, help='')
    subparsers.add_argument('--EPSILON', default=1e-8, type=float, help='')
    subparsers.add_argument('--init_lr', default=0.001, type=float, help='')
    subparsers.add_argument('--lrate', default=0.001, type=float, help='')
    subparsers.add_argument('--init_epoch', default=20, type=int, help='')
    subparsers.add_argument('--epochs', default=50, type=int, help='')
    subparsers.add_argument('--init_weight_decay', default=0.0005, type=float, help='')
    subparsers.add_argument('--weight_decay', default=0.0002, type=float, help='')

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int, help='')
    subparsers.add_argument('--dist_url', default='env://', help='')
    subparsers.add_argument('--device', default='cuda', help='')
    subparsers.add_argument('--local_rank', default=0, type=int, help='')

    # auxiliary
    subparsers.add_argument('--sched', default=False, type=bool, help='')
    subparsers.add_argument('--calculate_distributed', default=False, type=bool, help='')