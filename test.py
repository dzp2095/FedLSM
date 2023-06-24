#
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import logging
import torch
import yaml

from src.datasets.dataset_cxr import ChestDataset
from src.datasets.dataset_cxr import class_names as cxr_class_names

from src.datasets.dataset_isic import SkinDataset
from src.datasets.dataset_isic import class_names as skin_class_names

from torch.utils.data import DataLoader
from src.utils.metric_logger import MetricLogger
from statistics import mean
from src.utils.cxr_metrics import cxr_epoch_val
from src.utils.skin_metrics import skin_epoch_val

from src.model.net import DenseNet121

from src.utils.device_selector import get_free_device_name
from src.utils.args_parser import args

logging.basicConfig(filename=f'log_eval.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

if __name__ == "__main__":

        resume_path = args.resume_path
        
        if resume_path is not None and os.path.isfile(resume_path):
            logging.info(f"Resume from: {resume_path}")
            w = torch.load(resume_path)
            cfg = yaml.safe_load(open(args.config))

            if cfg['task']==0:
                class_names = cxr_class_names
                test_dataset = ChestDataset(csv_file=args.test_csv_path, cfg=cfg)

            elif cfg['task']==1:
                class_names = skin_class_names
                test_dataset = SkinDataset(csv_file=args.test_csv_path, cfg=cfg)


            metric_logger = MetricLogger()
            batch_size = 64
            num_workers = 8
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, \
                num_workers=num_workers, pin_memory=True)

            model = DenseNet121(cfg)
            model.load_state_dict(w)
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
            device = get_free_device_name()
            model = model.to(device)
            
            if cfg['task']==0:
                _, AUROCs = cxr_epoch_val(cfg, model, test_dataloader, loss_fn, device, cal_metrics=False)
                metric_dict = {}
                for i, _ in enumerate(class_names):
                    metric_dict[f"test/auc_{class_names[i]}"] = AUROCs[i]
                metric_dict["test/auc_avg"] =  mean(AUROCs)
            elif cfg['task']==1:
                roc_auc, acc, pre, recal, f1 = skin_epoch_val(model, test_dataloader, device)
                metric_dict = {}
                metric_dict["val/auc_avg"] =  roc_auc
                metric_dict["val/f1_avg"] =  f1
                metric_dict["val/acc"] =  acc
                metric_dict["val/precision"] =  pre
                metric_dict["val/recall"] =  recal

            metric_logger.update(**metric_dict)
            logging.info(f"Test result:  {metric_dict}")



