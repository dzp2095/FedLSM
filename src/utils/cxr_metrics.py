# encoding: utf-8
import os 
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

from src.utils.metric_logger import EMAMetricLogger
from src.datasets.dataset_cxr import class_names


import logging

def compute_AUCs(gt, pred, cfg, competition=True):
    """
    Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    for i, cls in enumerate(class_names):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            #print('Error in computing roc_auc_score for {}.\n Error msg:{} '.format(cls, error))
            AUROCs.append(0)
    return AUROCs

def compute_APs(gt, pred, cfg, competition=True):
    APs = []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    for i, cls in enumerate(class_names):
        try:
            APs.append(average_precision_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            #print('Error in computing roc_auc_score for {}.\n Error msg:{} '.format(cls, error))
            APs.append(0)
    return APs


def compute_metrics(gt, pred, cfg, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """
    Accus, Precs, Recas, F1scs = [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    THRESH = 0.5
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    #indexes = range(n_classes)
    
#     pdb.set_trace()
    
    for i, cls in enumerate(class_names):
        try:
            Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError as error:
            #print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Precs.append(precision_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            #print('Error in computing precision for {}.'.format(i))
            Precs.append(0)
        
        try:
            Recas.append(recall_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            #print('Error in computing recall for {}.'.format(i))
            Recas.append(0)
        
        try:
            F1scs.append(f1_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            #print('Error in computing F1-score for {}.'.format(i))
            F1scs.append(0)
    
    return Accus, Precs, Recas, F1scs

def cxr_epoch_val(cfg, model, dataLoader, loss_fn, device, cal_metrics=False):
    level = cfg["eval"]["level"]
    assert level in ('study', 'image')
    model.eval()
    m_logger = EMAMetricLogger()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    save_test = cfg["test"]["save_as_csv"]
    if level == 'study':
        gt_study   = {}
        pred_study = {}
        studies    = []
    
    if save_test:
        studies = []

    
    with torch.no_grad():
        tick = datetime.now()
        for i, (study, image, label) in enumerate(tqdm(dataLoader)):
            image, label = image.to(device), label.to(device)
            if cfg["train"]["use_CMSL"]:
                output = model(image, label)
            else:
                output = model(image)
                             
            loss = loss_fn(output, label.clone())
            m_logger.update(loss=loss)

            if level == 'study':
                for i in range(len(study)):
                    if study[i] in pred_study:
                        assert torch.equal(gt_study[study[i]], label[i])
                        pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                    else:
                        gt_study[study[i]] = label[i]
                        pred_study[study[i]] = output[i]
                        studies.append(study[i])
            elif level == 'image':
                gt = torch.cat((gt, label), 0)
                pred = torch.cat((pred, output), 0)
        
        if level == 'study':
            for study in studies:
                gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
                pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

        AUROCs = compute_AUCs(gt, pred, cfg, competition=True)

        target_dir = cfg["test"]["target_dir"]
        if level == 'study' and save_test:
            gt_np = gt.cpu().detach().numpy()
            pred_np = pred.cpu().detach().numpy()
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            logging.info(class_names)
            _label = np.array([[studies[i]]+gt_np[i].tolist() for i in range(len(studies))], dtype=object)
            _pred = np.array([[studies[i]]+pred_np[i].tolist() for i in range(len(studies))], dtype=object)
            pd.DataFrame(_label).to_csv(os.path.join(target_dir, 'label.csv'), header=['path']+class_names, index=None)
            pd.DataFrame(_pred).to_csv(os.path.join(target_dir, 'pred.csv'), header=['path']+class_names, index=None)
            logging.info('=======> csv file saved <=========')
            
        if cal_metrics:
            metrics = compute_metrics(gt, pred, cfg, competition=True)
            APs = compute_APs(gt, pred, cfg, competition=True)

    logging.info("Time spent for val: {}".format(str(datetime.now() - tick).split('.')[0]))
    if cal_metrics:
        return m_logger.loss.global_avg, AUROCs, APs, metrics
    else:
        return m_logger.loss.global_avg, AUROCs
