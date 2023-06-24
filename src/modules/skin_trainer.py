import logging
import torch 
import numpy as np

from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from src.modules.defaults import TrainerBase
from src.modules import hooks
from src.model.net import DenseNet121
from src.datasets.data_pool_isic import SkinDataPool

class SkinTrainer(TrainerBase):

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)
        self.use_data_pool = False
        self._pseudo_iter = None
        self._uncertain_iter = None
        self._confident_iter = None
        self._start_mixup = False
        self._start_pseudo = False
        self.global_pos_weight = None
        self.pseudo_label_p_thresh = self.cfg['fl']['pseudo_thresh']
        self.pseudo_label_n_thresh = self.cfg['fl']['pseudo_negative_thresh']
        self.uncertain_label_p_thresh = self.cfg['fl']['data_pool']['pos_thresh']
        self.uncertain_label_n_thresh = self.cfg['fl']['data_pool']['neg_thresh']
        self.pseudo_label_temp = self.cfg['fl']['pseudo_temperate']
        self.ema = self.cfg['fl']['ema']
        self.lambda_uncertain = self.cfg['fl']['data_pool']['lambda_uncertain']
        self.register_hooks(self.build_hooks())

    def build_model(self):
        self.model = DenseNet121(cfg=self.cfg)
    
    def init_dataloader(self):
        self.data_pool = SkinDataPool(self.cfg)
        self.label_mask = self.data_pool.label_mask
        batch_size = self.cfg["train"]["batch_size"]
        self.iter_per_epoch = len(self.data_pool.train_raw_dataset)//batch_size
    
    def build_hooks(self):
        ret = [hooks.Timer()]
        if self.cfg["hooks"]["eval"]:
            ret.append(hooks.ValEval())
        if self.cfg["hooks"]["wandb"]:
            ret.append(hooks.WAndBUploader(self.cfg))
        if self.cfg["hooks"]["best_saver"]:
            ret.append(hooks.BestCheckpointer())
        if self.cfg["hooks"]["val_loss"]:
            ret.append(hooks.ValLoss())
        if self.cfg["fl"]["ema"]:
            ret.append(hooks.EMA(self.cfg))
        return ret
    
    def before_train(self):
        num_classes = self.cfg['model']['num_classes']
        self._class_nums = np.zeros(num_classes)
        batch_size = self.cfg["train"]["batch_size"]
        uncertain_batch_size = self.cfg['fl']['data_pool']['batch_size']

        pseudo_batch_size = batch_size - uncertain_batch_size

        pseudo_dataloader, uncertain_dataloader, confident_dataloader = self.data_pool.get_dataloader([self.model], \
                                self.device, pseudo_batch_size, uncertain_batch_size)
        self._data_iter = None
        self._pseudo_iter = iter(pseudo_dataloader)
        self._uncertain_iter = iter(uncertain_dataloader)
        self._confident_iter = iter(confident_dataloader)
        
        return super().before_train()

    def after_train(self):
        logging.info(f"# of classes: {self._class_nums}")
        return super().after_train()
    

    def run_step(self):
        self.model.train()
        label_mask = torch.FloatTensor(self.label_mask) 
        label_mask = label_mask.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        
        _, weak, strong, p_label = next(self._pseudo_iter)
        _, u_img, u_label = next(self._uncertain_iter)
        _, c_img, c_label = next(self._confident_iter)
       
        w_img = torch.cat((weak, u_img, c_img), dim=0)

        w_sz = weak.shape[0]
        u_sz = u_img.shape[0]

        w_img, p_label = w_img.to(self.device), p_label.to(self.device)

        # 1. pseudo labeling 
        with torch.no_grad():
            if self.ema:
                logits = self.ema_model.ema(w_img)
            else:
                logits = self.model(w_img)
        # 1.1 pseudo labeling for data with high/medium certainty
        w_out = logits[:w_sz]
        pseudo_label = torch.softmax(w_out.detach()/self.pseudo_label_temp, dim=-1).to(self.device, dtype=p_label.dtype)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.pseudo_label_p_thresh)
        label_missing = ~torch.any(p_label, dim=1) 
        label_missing = torch.logical_and(label_missing, mask)
        p_label[label_missing, targets_u[label_missing]] = 1
        self._class_nums += torch.count_nonzero(p_label, dim=0).detach().cpu().numpy()
        # 2. pseudo labeling for mixup
        u_img, c_img, u_label, c_label = u_img.to(self.device), c_img.to(self.device), u_label.to(self.device), c_label.to(self.device)

        u_out = logits[w_sz:w_sz+u_sz]
        c_out = logits[w_sz+u_sz:]
        pseudo_u_label = torch.softmax(u_out.detach()/self.pseudo_label_temp, dim=-1).to(self.device, dtype=p_label.dtype)
        pseudo_c_label = torch.softmax(c_out.detach()/self.pseudo_label_temp, dim=-1).to(self.device, dtype=p_label.dtype)
        
        # confident data
        max_probs, targets_u = torch.max(pseudo_c_label, dim=-1)
        mask = max_probs.ge(self.pseudo_label_p_thresh)
        label_missing = ~torch.any(pseudo_c_label, dim=1) 
        label_missing = torch.logical_and(label_missing, mask)
        c_label[label_missing, targets_u[label_missing]] = 1
        
        # uncertain data
        max_probs, targets_u = torch.max(pseudo_u_label, dim=-1)
        mask = max_probs.ge(self.uncertain_label_p_thresh)
        label_missing = ~torch.any(pseudo_u_label, dim=1) 
        label_missing = torch.logical_and(label_missing, mask)
        u_label[label_missing, targets_u[label_missing]] = 1

        with torch.no_grad():
            mixup_img, mixup_label = self._mixup(u_img, c_img, u_label, c_label)
        mixup_label[~torch.logical_and(torch.any(u_label, dim=1), torch.any(c_label, dim=1))] = 0 
        strong = strong.to(self.device)
        inp = torch.cat((strong, mixup_img), dim=0)
        s_out = self.model(inp)
        loss = cross_entropy(s_out[:w_sz], p_label) + self.lambda_uncertain*(cross_entropy(s_out[w_sz:], mixup_label))

        self.loss_logger.update(loss=loss)
        self.metric_logger.update(loss=loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _mixup(self, img1, img2, l1, l2):
        alpha = self.cfg['fl']['data_pool']['alpha']
        beta_dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        weight = beta_dist.sample()
        weight = torch.FloatTensor(weight)
        weight = weight.to(device=self.device)
        img =  weight * img1 + (1-weight)*img2
        label = weight * l1 + (1-weight)*l2
        return img, label

    @property
    def train_data_num(self):
        return len(self.data_pool.train_raw_dataset)
    
