import torch 
import logging
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.functional import F
from tqdm import tqdm

from src.datasets.dataset_cxr import ChestDatasetRaw, PesudoSubset, OriginialSubset, SrongAugmentedSubset
from src.datasets.sampler import TrainingSampler

class CXRDataPool:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_raw_dataset = ChestDatasetRaw(csv_file=self.cfg['dataset']['train'], cfg=self.cfg)
        self.train_rank_dataset = OriginialSubset(Subset(self.train_raw_dataset, list(range(len(self.train_raw_dataset)))), self.cfg)

        self.label_mask = self.train_raw_dataset.compute_label_mask()
        self.num_workers = self.cfg["train"]["num_workers"]
        self.uncertain_pool_size = self.cfg['fl']['data_pool']['uncertain_num']
        self.confident_pool_size = self.cfg['fl']['data_pool']['certain_num']

    def _get_data_idx(self, nets, device):
        batch_size = 64
        num_workers = 8
        dataloader = DataLoader(dataset=self.train_rank_dataset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, pin_memory=True)

        num_net = len(nets)
        n_data = len(self.train_raw_dataset)
        label_mask = torch.BoolTensor(self.label_mask) 
        
        label_mask = label_mask.to(device)
        rank_array = torch.empty((0),device=device)

        with tqdm(total=n_data, desc=f'Calculate Entropy. ', unit='img') as pbar:
            for _, images, _ in dataloader:
                images = images.to(device=device, dtype=torch.float32)
                batch_size = images.shape[0]
                # keep the indices of images
                ensemble_entropy = torch.zeros((batch_size), device=device)
                for net in nets:
                    with torch.no_grad():
                        output = F.sigmoid(net(images).detach()) * ~label_mask
                    entr_res = torch.special.entr(output)
                    ensemble_entropy = ensemble_entropy + torch.mean(entr_res, dim = 1)
                ensemble_entropy = ensemble_entropy / num_net
                rank_array = torch.cat((rank_array, ensemble_entropy))
                pbar.update(images.shape[0])

        # rank
        # indices : indices of ndarray rank_array, entropy value of those rows are k largest
        _, uncertain_indices = torch.topk(rank_array, k = self.uncertain_pool_size, largest=True)
        _, confident_indices = torch.topk(rank_array, k = self.confident_pool_size, largest=False)
        return uncertain_indices.tolist(), confident_indices.tolist()

    def get_dataloader(self, nets, device, pseudo_batch_size, uncertain_batch_size):
        logging.info("Rerank Data Pool")
        uncertain_indices, confident_indices = self._get_data_idx(nets, device)
        pseudo_indices = set(list(range(len(self.train_raw_dataset)))).difference(uncertain_indices)
        pseudo_dataset =  PesudoSubset(Subset(self.train_raw_dataset, list(pseudo_indices)),  self.cfg)
        uncertain_dataset = OriginialSubset(Subset(self.train_raw_dataset, uncertain_indices), self.cfg)
        confident_dataset = OriginialSubset(Subset(self.train_raw_dataset, confident_indices), self.cfg)

        pseudo_dataloader = DataLoader(dataset=pseudo_dataset, batch_size=pseudo_batch_size, 
            sampler=TrainingSampler(len(pseudo_dataset), seed=523), 
            shuffle=False, num_workers=self.num_workers, pin_memory=True)

        uncertain_dataloader = DataLoader(dataset=uncertain_dataset, batch_size=uncertain_batch_size, 
            sampler=TrainingSampler(len(uncertain_dataset), seed=523), 
            shuffle=False, num_workers=self.num_workers // 2, pin_memory=True)

        confident_dataloader = DataLoader(dataset=confident_dataset, batch_size=uncertain_batch_size, 
            sampler=TrainingSampler(len(confident_dataset),seed=523), 
            shuffle=False, num_workers=self.num_workers // 2, pin_memory=True)
        return pseudo_dataloader, uncertain_dataloader, confident_dataloader

    def get_regular_dataloader(self, batch_size):
        regular_dataset = SrongAugmentedSubset(Subset(self.train_raw_dataset, list(range(len(self.train_raw_dataset)))),  self.cfg)
        regular_dataloader = DataLoader(dataset=regular_dataset, batch_size=batch_size, 
            sampler=TrainingSampler(len(regular_dataset), seed=523), 
            shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return regular_dataloader
        

