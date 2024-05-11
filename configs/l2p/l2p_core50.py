import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--prefix', default='best', type=str, help='')

    #dataset
    subparsers.add_argument('--dataset', default="core50", type=str, help='')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--data_path', default="/home/pinna/data/core50_128x128", type=str, help='')
    subparsers.add_argument('--shuffle', default=False, type=bool, help='')
    subparsers.add_argument('--normalize', default=True, type=bool, help='')
    subparsers.add_argument('--color_jitter', default=True, type=bool, help='')
    subparsers.add_argument('--pin_mem', default=True, type=bool, help='')

    # increment
    subparsers.add_argument('--init_cls', default=50, type=int, help='')
    subparsers.add_argument('--increment', default=50, type=int, help='')
    subparsers.add_argument('--num_tasks', default=8, type=int, help='')
    subparsers.add_argument('--scenario', default='domain', type=str, help='')
    subparsers.add_argument('--extra_eval', default='L2P', type=str, help='')

    # rehearsal
    subparsers.add_argument('--memory_size', default=0, type=int, help='')
    subparsers.add_argument('--memory_per_class', default=0, type=int, help='')
    subparsers.add_argument('--fixed_memory', default=True, type=bool, help='')

    # train model
    subparsers.add_argument('--engine_name', default="l2p", type=str, help='')
    subparsers.add_argument('--net_type', default='vit', type=str, help='')
    subparsers.add_argument('--seed', default=42, type=int, help='')
    subparsers.add_argument('--pretrained_model', default='vit_base_patch16_224_base', type=str, help='')
    subparsers.add_argument('--pretrained', default=True, type=bool, help='')
    subparsers.add_argument('--embd_dim', default=768, type=int, help='')

    # training stage
    subparsers.add_argument('--batch_size', default=16, type=int, help='')
    subparsers.add_argument('--num_workers', default=16, type=int, help='')
    subparsers.add_argument('--EPSILON', default=1e-8, type=float, help='')
    subparsers.add_argument('--init_lr', default=0.01, type=float, help='')
    subparsers.add_argument('--lrate', default=0.01, type=float, help='')
    subparsers.add_argument('--init_epoch', default=10, type=int, help='')
    subparsers.add_argument('--epochs', default=10, type=int, help='')
    subparsers.add_argument('--init_weight_decay', default=0.0, type=float, help='')
    subparsers.add_argument('--weight_decay', default=0.0, type=float, help='')
    subparsers.add_argument('--T', default=2.0, type=float, help='')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM')

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int, help='')
    subparsers.add_argument('--dist_url', default='env://', help='')
    subparsers.add_argument('--device', default='cuda', help='')
    subparsers.add_argument('--local_rank', default=0, type=int, help='')
    subparsers.add_argument('--distributed', default=False, type=bool, help='')

    # auxiliary
    subparsers.add_argument('--sched', default=False, type=bool, help='')
    subparsers.add_argument('--calculate_distributed', default=True, type=bool, help='')
    subparsers.add_argument('--scale_batch', default=False, type=bool, help='')
    subparsers.add_argument('--scale_lr', type=bool, default=True, help='')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    #prompt parameter
    subparsers.add_argument('--top_k', default=5, type=int, )
    subparsers.add_argument('--size', default=10, type=int,)
    subparsers.add_argument('--prompt_length', default=10, type=int, help='')
    subparsers.add_argument('--length', default=5,type=int, )
    subparsers.add_argument('--prompt_pool', default=True, type=bool,)
    subparsers.add_argument('--shared_prompt_pool', default=False, type=bool)
    subparsers.add_argument('--prompt_key', default=True, type=bool,)
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    #vit parameter
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')
    subparsers.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')

    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')