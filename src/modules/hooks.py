
import logging
import torch 
import wandb
import copy
from datetime import datetime

from src.modules.defaults import HookBase
from src.model.ema import ModelEMA


class BestCheckpointer(HookBase):

    def before_train(self):
        self.best_metric = 0
        self.val_interval = self.trainer.cfg["train"]["eval_interval"]
        logging.info("######## Running best check pointer")

    def after_step(self):        
        metric_name = "val/auc_avg"
        if self.trainer.iter % self.val_interval == 0:
            if metric_name in self.trainer.metric_logger._dict:
                eval_metric = self.trainer.metric_logger._dict[metric_name]
                if self.best_metric < eval_metric:
                    self.best_metric = eval_metric
                    self.trainer.save_model(f"best_{metric_name}")
                    logging.info(f"######## New best {metric_name}: {self.best_metric}")

class ValEval(HookBase):
    
    def before_train(self):
        self.val_interval = self.trainer.cfg["train"]["eval_interval"]
        self.test_interval = self.trainer.cfg["train"]["test_interval"]
        self.test_start = self.trainer.cfg["train"]["test_start"]

    def after_step(self):
        if self.trainer.iter % self.val_interval == 0:
            self.trainer.validate()
        
        if (self.trainer.iter >= self.test_start) and (self.trainer.iter % self.test_interval == 0):
            self.trainer.test()

class ValLoss(HookBase):
    def before_train(self):
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x
        self._data_iter = iter(cycle(self.trainer.val_dataloader))

    def after_step(self):
        self.trainer.model.eval()
        with torch.no_grad():
            _, image, label = next(self._data_iter)

            image, label = image.to(self.trainer.device), label.to(self.trainer.device)
            if self.trainer.cfg["train"]["use_CMSL"]:
                output = self.trainer.model(image, label)
            else:
                output = self.trainer.model(image)
            loss = self.trainer.loss_fn(output, label.clone())
            self.trainer.loss_logger.update(val_loss=loss)
            self.trainer.metric_logger.update(val_loss=loss)

class Timer(HookBase):

    def before_train(self):
        self.tick = datetime.now()
        logging.info("Begin training at: {}".format(self.tick.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("######## Running Timer")

    def after_train(self):
        tock = datetime.now()
        logging.info("\nBegin training at: {}".format(self.tick.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Finish training at: {}".format(tock.strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Time spent: {}\n".format(str(tock - self.tick).split('.')[0]))

class WAndBUploader(HookBase):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        wandb.login(key=self.cfg["wandb"]["key"])
        self.wandb_id = wandb.util.generate_id()

    def before_train(self):
        self.experiment = wandb.init(project=f'{self.cfg["wandb"]["project"]}', resume='allow', id=self.wandb_id, name=self.cfg["wandb"]["run_name"])
        self.val_interval = self.cfg["train"]["eval_interval"]
        self.experiment.config.update(
            dict(steps=self.trainer.max_iter, batch_size=self.trainer.train_dataloader.batch_size,
                 learning_rate = self.cfg["train"]["optimizer"]["lr"]), allow_val_change=True)
       
        logging.info("######## Running wandb logger")

    def after_step(self):
        wandb_dict = {}
        metric_names = {"loss", "val_loss"}
        for metric_name in metric_names:
            if metric_name in self.trainer.metric_logger._dict:
                wandb_dict.update({metric_name: self.trainer.metric_logger._dict[metric_name]})

        if self.trainer.iter % self.val_interval == 0:
            wandb_dict.update(self.trainer.metric_logger._dict)
        self.experiment.log(wandb_dict)


    def after_train(self):
        self.experiment.finish()

class EMA(HookBase):
    def __init__(self, cfg):
        self.decay = cfg['fl']['ema_decay']

    def before_train(self):
        self.trainer.ema_model = ModelEMA(self.trainer.device, self.trainer.model, self.decay)

    def after_step(self):
        self.trainer.ema_model.update(self.trainer.model)