import torch
import os
import numpy as np
import logging

from src.utils.args_parser import args

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def get_free_device_name():
    if (torch.cuda.is_available() and args.gpu):
        return f'cuda:{get_free_gpu()}'
    else:
        return 'cpu'