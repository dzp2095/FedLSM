import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/deng/project/FedOmniCls/configs/full_conf.yaml", help='config file')
    parser.add_argument('--eval_only', type=bool, default=False, help='whether only test')
    parser.add_argument('--resume_path', type=str, default="", help='saved checkpoin')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='whether use gpu')
    parser.add_argument('--val_interval', type=int, default=1, help='valdation interval')
    parser.add_argument('--amp', type=int, default=False, help='mixed precision')
    parser.add_argument('--path_checkpoint', type=str, default=None, help='path of the checkpoint of the model')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--train_dataset', type=int,  default=1, help='index of train dataset. c1, c2')
    parser.add_argument('--test_csv_path', type=str, default="default", help='name of this run in wandb log')
    parser.add_argument('--resolution', type=int,default="320", help='resolution of the image')

    args, unknown = parser.parse_known_args()
    return args

args = args_parser()
