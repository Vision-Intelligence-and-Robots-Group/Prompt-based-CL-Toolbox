import argparse
from trainer import train

def main(args):
    train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce of prompt-based continual learning algorthms.')

    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='sub settings.')

    if config == 'l2p_cifar100':
        from configs.l2p.l2p_cifar100 import get_args_parser
        config_parser = subparser.add_parser('l2p_cifar100', help='')    
    elif config == 'l2p_imagenetr':
        from configs.l2p.l2p_imagenetr import get_args_parser
        config_parser = subparser.add_parser('l2p_imagenetr', help='')
    elif config == 'dualp_cifar100':
        from configs.dualp.dualp_cifar100 import get_args_parser
        config_parser = subparser.add_parser('dualp_cifar100', help='')     
    elif config == 'dualp_imagenetr':
        from configs.dualp.dualp_imagenetr import get_args_parser
        config_parser = subparser.add_parser('dualp_imagenetr', help='')
    elif config == 'apil_cifar100':
        from configs.apil.apil_cifar100 import get_args_parser
        config_parser = subparser.add_parser('apil_cifar100', help='')
    elif config == 'sprompt_cddb_slip':
        from configs.sprompt.sprompt_cddb_slip import get_args_parser
        config_parser = subparser.add_parser('sprompt_cddb_slip', help='')
    elif config == 'sprompt_cddb_sip':
        from configs.sprompt.sprompt_cddb_sip import get_args_parser
        config_parser = subparser.add_parser('sprompt_cddb_sip', help='')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    main(args)