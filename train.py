#
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import weakref

from src.utils.args_parser import args
import yaml
import logging

import random
import numpy as np
import torch
from torch.backends import cudnn

from src.fl.client import Client
from src.fl.server import Server
from datetime import datetime

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

now = str(datetime.timestamp(datetime.now()))
logging.basicConfig(filename=f'log_fl_{now}.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

if __name__ == "__main__":

    try:
        cfg = yaml.safe_load(open(args.config))
        cfg["train"]["checkpoint_dir"] = f'{cfg["train"]["checkpoint_dir"]}{now}'
        if args.eval_only:
            server = Server([], cfg)
            cfg["train"]["resume_path"] = args.resume_path
            server.global_test()
        else:
            client_num = cfg['fl']['num_clients']
            clients = []
            for i in range(client_num):
                client = Client('client_' + str(i), args, cfg)
                clients.append(client)
            server = Server(clients, cfg)
            server.federated_train()
    except Exception as e:
        logging.critical(e, exc_info=True)

        