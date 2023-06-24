import copy
import torch
import abc
import logging

from src.modules.cxr_trainer import CXRTrainer
from src.modules.skin_trainer import SkinTrainer

class Client(abc.ABC):
    def __init__(self, name, args, cfg):
        self._name = name
        self.cfg = copy.deepcopy(cfg)
        self.setup()
        self.global_discriminator = None

        if self.cfg['task']==0:
            self.trainer = CXRTrainer(args=args, cfg= self.cfg)
        elif self.cfg['task']==1:
            self.trainer = SkinTrainer(args=args, cfg= self.cfg)
        self.round = 0
        self._train_data_num = self.trainer.train_data_num
        self.trainer.client_label = int(name.split('_')[-1]) + 1

    def load_model(self, model_weights):
        self.trainer.model.load_state_dict(model_weights, strict=False)
        # need to construct a new optimizer for the new network
        old_scheduler = copy.deepcopy(self.trainer.lr_scheduler.state_dict())
        old_optimizer = copy.deepcopy(self.trainer.optimizer.state_dict())
        self.trainer.build_optimizer()
        self.trainer.optimizer.load_state_dict(old_optimizer)
        self.trainer.build_schedular(self.trainer.optimizer)
        self.trainer.lr_scheduler.load_state_dict(old_scheduler)


    def setup(self):
        self.cfg["wandb"]["run_name"] = f"{self.cfg['wandb']['run_name']}_rounds_{self.cfg['fl']['rounds']}_{self.name}"
        self.cfg['dataset']['train'] = f"{self.cfg['dataset']['train']}/{self.name}/{self.cfg['fl']['mode']}_train.csv"        

        self.total_rounds = self.cfg['fl']['rounds']
        self.iter_per_round = self.cfg['fl']['local_iter']
        self.max_iter =  self.total_rounds * self.iter_per_round
        self.cfg["train"]["max_iter"] = self.max_iter

    def train(self):
        """train the model """
        logging.info(f"{self.name}: Starting training from round {self.round}")
        self.trainer.train(self.iter_per_round)
        self.round+=1

    @property
    def model(self):
        return self.trainer.model

    @property
    def name(self):
        return self._name

    @property
    def train_data_num(self):
        return self._train_data_num
        
    @property
    def class_nums(self):
        return self.trainer._class_nums


