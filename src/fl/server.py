import os
import logging
import torch
import copy
import weakref
from typing import List
import wandb
import numpy as np
from src.datasets.dataset_cxr import ChestDataset, class_names
from src.datasets.dataset_isic import SkinDataset

from torch.utils.data import DataLoader
from src.utils.metric_logger import MetricLogger
from statistics import mean
from src.utils.cxr_metrics import cxr_epoch_val
from src.utils.skin_metrics import skin_epoch_val
from src.model.net import DenseNet121

from src.fl.client import Client
from src.utils.device_selector import get_free_device_name

class Server:
    def __init__(self, clients: List[Client], cfg):
        self.clients = clients
        for client in self.clients:
            client.server = weakref.proxy(self)
        self.save_checkpoints = True  # turn on to save the checkpoints
        self.rounds = cfg['fl']['rounds']
        self.weights_ratio = []
        self.eval_start_round = cfg['fl']['eval_start_round']
        self.test_start_round = cfg['fl']['test_start_round']
        self.save_start_round = cfg['fl']['save_start_round']

        self.metric_logger = MetricLogger()
        self.cfg = copy.deepcopy(cfg)
        self.label_masks = [client.trainer.label_mask for client in clients ] 
        if self.cfg["fl"]["wandb_global"]:
            self.wandb_init()
        self.estimate_global_pos_weight()
        self.load_model()

        # config val/test func
        if self.cfg['task']==0:
            self.global_validate = self.cxr_validate
            self.global_test = self.cxr_test
        elif self.cfg['task']==1:
            self.global_validate = self.skin_validate
            self.global_test = self.skin_test
       

    def estimate_global_pos_weight(self):
        self.global_pos_weight = []
        clients_l = [np.array(client.trainer.data_pool.train_raw_dataset.labels) for client in self.clients]
        for i in range(self.cfg['model']['num_classes']):
            global_pos = 0
            global_neg = 0
            for j, client in enumerate(self.clients):
                global_pos += len(np.where(clients_l[j][:,i]==1)[0])
                global_neg += len(np.where(clients_l[j][:,i]==0)[0])
            self.global_pos_weight.append(global_neg*1. / global_pos)
        
        for client in self.clients:
            client.trainer.global_pos_weight = self.global_pos_weight
        logging.info('Estimated global pos_weight of each class', self.global_pos_weight)

    def wandb_init(self):
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.experiment.config.update(
            dict(steps=self.cfg["train"]["max_iter"], batch_size=  self.cfg["train"]["batch_size"],
                 learning_rate = self.cfg["train"]["optimizer"]["lr"]), allow_val_change=True)
       
        logging.info("######## Running wandb logger")

    def wandb_upload(self, metric_logger):
        self.experiment.log(metric_logger._dict)

    def _cal_weights_ratio(self):
        data_num = list(map(lambda x: x.train_data_num, self.clients))
        s = sum(data_num)
        self.weights_ratio = list(map(lambda num: num / s, data_num))

        class_nums = np.array([x.class_nums for x in self.clients])
        # class_nums = class_nums > 0
        class_sums = np.sum(class_nums, axis=0)
        self.classifier_ratio = torch.from_numpy(class_nums/class_sums)

    def aggregate(self):
        self._cal_weights_ratio()
        # FedAvg
        w_avg = copy.deepcopy(self.clients[0].model.state_dict())
        for k in w_avg.keys():    
            if 'classifier' in k:
                    w_avg[k] = torch.zeros_like(w_avg[k])
                    self.classifier_ratio = self.classifier_ratio.to(w_avg[k].device)
                    for i, client in enumerate(self.clients):
                        w_avg[k] += torch.mul(client.model.state_dict()[k].T, self.classifier_ratio[i]).T
            else:
                w_avg[k] = torch.multiply(w_avg[k], self.weights_ratio[0])
                for i in range(1, len(self.clients)):
                    w_avg[k] += torch.multiply(self.clients[i].model.state_dict()[k], self.weights_ratio[i])
        return w_avg

    # control the federated training process
    def federated_train(self):
        best_metric = 0
        for self.r in range(self.rounds):
            logging.info(f'''Federated Training, round {self.r} ''')
            
            # 1.client train
            for client in self.clients:
                client.train()
                
            # 2.aggregation
            with torch.no_grad():
                w_avg = self.aggregate()
            
            if (self.r >= self.eval_start_round):
                self.global_validate(w_avg)
                metric_name = "val/auc_avg"
                if metric_name in self.metric_logger._dict:
                    eval_metric = self.metric_logger._dict[metric_name]
                    if best_metric < eval_metric:
                        best_metric = eval_metric
                        self.save_model(w_avg, f"best_{metric_name}".replace('/',"_"))
                        logging.info(f"######## New best {metric_name}: {best_metric}")
            
            if (self.r >= self.save_start_round):
                self.save_model(w_avg, f"global_model_round_{self.r}")

            if (self.r >= self.test_start_round):
                self.global_test(w_avg)

            if self.cfg["fl"]["wandb_global"]:
                self.wandb_upload(self.metric_logger)
            
            # 3.send aggregated model
            for client in self.clients:
                client.load_model(w_avg)

        if self.cfg["fl"]["wandb_global"]:
            self.experiment.finish()

    def load_model(self):
        resume_path = self.cfg['train']['resume_path']
        if resume_path is not None and os.path.isfile(resume_path):
            logging.info(f"Resume from: {resume_path}")
            w = torch.load(resume_path)
            # send global model
            for client in self.clients:
                client.load_model(w)

    def save_model(self, w_avg, name):
        torch.save(w_avg,
            os.path.join(self.cfg["train"]["checkpoint_dir"], name + '.pth')
        )

    def global_validate(self):
        raise NotImplementedError

    def global_test(self):
        raise NotImplementedError

    def cxr_validate(self, w_avg):
        val_dataset = ChestDataset(csv_file=self.cfg['dataset']['val'], cfg=self.cfg)
        batch_size = self.cfg["train"]["batch_size"]
        num_workers = 8
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, pin_memory=True)

        model = DenseNet121(self.cfg)
        model.load_state_dict(w_avg)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        device = get_free_device_name()
        model = model.to(device)
        val_loss, AUROCs = cxr_epoch_val(self.cfg, model, val_dataloader, loss_fn, device, cal_metrics=False)
        metric_dict = {}
        for i, _ in enumerate(class_names):
            metric_dict[f"val/auc_{class_names[i]}"] = AUROCs[i]
        metric_dict["val/auc_avg"] =  mean(AUROCs)
        metric_dict["val/epoch_loss"] =  val_loss
        self.metric_logger.update(**metric_dict)
        logging.info(f"Evaluation result:  {metric_dict}")

    def cxr_test(self, w_avg):
        test_dataset = ChestDataset(csv_file=self.cfg['dataset']['test'], cfg=self.cfg)
        batch_size = self.cfg["train"]["batch_size"]
        num_workers = 8
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, pin_memory=True)

        model = DenseNet121(self.cfg)
        model.load_state_dict(w_avg)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        device = get_free_device_name()
        model = model.to(device)
        _, AUROCs = cxr_epoch_val(self.cfg, model, test_dataloader, loss_fn, device, cal_metrics=False)
        metric_dict = {}
        for i, _ in enumerate(class_names):
            metric_dict[f"test/auc_{class_names[i]}"] = AUROCs[i]
        metric_dict["test/auc_avg"] =  mean(AUROCs)
        self.metric_logger.update(**metric_dict)
        logging.info(f"Test result:  {metric_dict}")

    def skin_validate(self, w_avg):
        val_dataset = SkinDataset(csv_file=self.cfg['dataset']['val'], cfg=self.cfg)
        batch_size = self.cfg["train"]["batch_size"]
        num_workers = 8
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, pin_memory=True)

        model = DenseNet121(self.cfg)
        model.load_state_dict(w_avg)
        device = get_free_device_name()
        model = model.to(device)
        roc_auc, acc, pre, recal, f1 = skin_epoch_val(model, val_dataloader, device)
        metric_dict = {}
        metric_dict["val/auc_avg"] =  roc_auc
        metric_dict["val/f1_avg"] =  f1
        metric_dict["val/acc"] =  acc
        metric_dict["val/precision"] =  pre
        metric_dict["val/recall"] =  recal

        self.metric_logger.update(**metric_dict)
        logging.info(f"Evaluation result:  {metric_dict}")

    def skin_test(self, w_avg):
        test_dataset = SkinDataset(csv_file=self.cfg['dataset']['test'], cfg=self.cfg)
        batch_size = self.cfg["train"]["batch_size"]
        num_workers = 8
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, pin_memory=True)

        model = DenseNet121(self.cfg)
        model.load_state_dict(w_avg)
        device = get_free_device_name()
        model = model.to(device)
        roc_auc, acc, pre, recal, f1 = skin_epoch_val(model, test_dataloader, device)
        metric_dict = {}
        metric_dict["test/auc_avg"] =  roc_auc
        metric_dict["test/f1_avg"] =  f1
        metric_dict["test/acc"] =  acc
        metric_dict["test/precision"] =  pre
        metric_dict["test/recall"] =  recal
        self.metric_logger.update(**metric_dict)
        logging.info(f"Test result:  {metric_dict}")
