# encoding: utf-8
import os 
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch 

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report

import logging

def compute_metrics_multiclass(gt, pred_labels, pred_probs):
    roc_auc, acc, pre, recal, f1 = 0, 0, 0, 0, 0
    gt_np = np.asarray(gt)
    pred_labels_np = np.asarray(pred_labels)
    pred_probs_np = np.asarray(pred_probs)
    try:
        acc = accuracy_score(gt_np, pred_labels_np)
        res = classification_report(gt_np, pred_labels_np, output_dict=True)['macro avg']
        recal = res['recall']
        pre = res['precision']
        f1 = res['f1-score']
        roc_auc = roc_auc_score(gt_np, pred_probs_np, average='macro', multi_class='ovr')
    except ValueError as error:
        logging.exception(error)

    return roc_auc, acc, pre, recal, f1

def skin_epoch_val(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    gt = []
    pred_labels = []
    pred_probs = []
    with torch.no_grad():
        # iterate over the validation set
        for _, images, labels in tqdm(dataloader, total=num_val_batches, desc='Evaluation', unit='batch', leave=False):
            # move images and labels to correct device and type
            images = images.to(device=device)
            labels = labels.to(device=device)
            pred = net(images)
            output = F.one_hot(torch.argmax(pred, dim=1), num_classes = labels.shape[1])
            prob = F.softmax(pred, dim=1)
            gt = gt + labels.detach().cpu().numpy().tolist()
            pred_labels = pred_labels + output.detach().cpu().numpy().tolist()
            pred_probs = pred_probs + prob.detach().cpu().numpy().tolist()
    roc_auc, acc, pre, recal, f1 = compute_metrics_multiclass(gt, pred_labels,pred_probs)

    return roc_auc, acc, pre, recal, f1