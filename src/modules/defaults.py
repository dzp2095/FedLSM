#
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import weakref
import copy
import os

from shutil import copyfile
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.device_selector import get_free_device_name
from src.utils.metric_logger import MetricLogger, EMAMetricLogger

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}

class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    """
    def __init__(self, args, cfg) -> None:
        self._hooks: List[HookBase] = []
        
        self.args = args
        self.val_interval = self.args.val_interval
        self.cfg = copy.deepcopy(cfg)
        self.init_dataloader()
        self.setup_train()

        self.name = "defaults"
        self.metric_logger = MetricLogger()
        self.loss_logger = EMAMetricLogger()


    def setup_train(self):
        """
        Setup model/optmizer/lr_schedular/loss and etc. for training
        """
        self.iter = 0
        self.start_iter = 0
        self.epoch = 0
        self.max_iter = self.cfg["train"]["max_iter"]
        if self.cfg["train"]["device"] is not None:
            self.device = self.cfg["train"]["device"]
        else:
            self.device = get_free_device_name()
        self.build_model()
        resume_path = self.cfg["train"]["resume_path"]
        self.model = self.model.to(self.device)
        self.auc_best = 0

        Path(self.cfg["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        # Copy config file to checkpoint folder
        cfg_file = os.path.join(self.cfg["train"]["checkpoint_dir"], os.path.basename(self.args.config))
        copyfile(self.args.config, cfg_file)

        self.build_optimizer()
        self.build_schedular(self.optimizer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def build_model(self):
        raise NotImplementedError
            
    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["train"]["optimizer"]['lr'],
            betas=(self.cfg["train"]["optimizer"]['beta1'],self.cfg["train"]["optimizer"]['beta2']), 
            weight_decay=self.cfg["train"]["optimizer"]['weight_decay'])


    def build_schedular(self, optimizer):
        self.lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.cfg["train"]["lr_scheduler"]["factor"], 
        patience=self.cfg["train"]["lr_scheduler"]["patience"], verbose=True, min_lr=self.cfg["train"]["lr_scheduler"]["min_lr"])
    

    def init_dataloader(self):
        raise NotImplementedError

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
        
    def train(self, iter: int):
        """
        Args:
            iter (int): 
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))
        start_iter = self.start_iter
        max_iter = self.max_iter

        with tqdm(total=iter) as pbar:
            try:
                self.before_train()
                for self.iter in range(start_iter, min(max_iter, start_iter + iter)):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    pbar.update(1)
                    pbar.set_postfix(**{'loss (batch)': self.metric_logger.loss})
                    if self.iter % self.iter_per_epoch == 0:
                        self.epoch += 1
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.start_iter = self.iter
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError


                