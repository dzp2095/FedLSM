# encoding: utf-8

"""
CXR14
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import logging

import albumentations
from albumentations.pytorch import ToTensorV2

class_names =  [
            # "No Finding",
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Hernia",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pleural_Thickening",
            "Pneumonia",
            "Pneumothorax",           
            ]

class_name_to_id = {}
for i, each_label in enumerate(class_names):
    class_id = i  # starts with 0, ignore No Finding
    class_name = each_label
    class_name_to_id[class_name] = class_id


class ChestDataset(Dataset):
    def __init__(self, csv_file, cfg):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(ChestDataset, self).__init__()
        self.img_names, self.labels = self.__load_imgs__(csv_file)
        self.transform = albumentations.Compose([
            albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
            albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
            ToTensorV2()
        ])


    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        path = self.img_names[index]
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
        label = self.labels[index]
        image = self.transform(image=image)['image']
        return index, image, label

    def __len__(self):
        return len(self.img_names)

    def compute_pos_weight(self):
        l = np.array(self.labels)
        pos_weight = []
        for i in range(len(class_names)):    # labels of shape n_samples x n_classes
            n_pos = len(np.where(l[:,i]==1)[0])
            n_neg = len(np.where(l[:,i]==0)[0])
            if n_pos == 0:
                pos_weight.append(0.)
            else:
                pos_weight.append(n_neg*1. / n_pos)
        logging.info('pos_weight of each class', pos_weight)
        return pos_weight

    def compute_label_mask(self):
        l = np.array(self.labels)
        label_mask = [False] * len(class_names)
        for i in range(len(class_names)):    # labels of shape n_samples x n_classes
            exist = np.any(l[:,i]==1)
            label_mask[i] = exist
        logging.info('exist of each class', label_mask)
        return label_mask

    def __load_imgs__(self, csv_path):
        data = pd.read_csv(csv_path)
        imgs = data['path'].values
        labels = data['Finding Labels'].values
        # convert label to one-hot
        onehots = []
        for label in labels:
            label = label.split('|')
            onehot = np.zeros(len(class_name_to_id), dtype=np.float32)
            for l in label:
                if l != "No Finding":
                    onehot[class_name_to_id[l]] = 1
            onehots.append(onehot)
        labels = onehots
        logging.info(f'Total # images:{len(imgs)}, labels:{len(labels)}')
        return imgs, labels

class ChestDatasetRaw(ChestDataset):
    def __init__(self, csv_file, cfg):
        super(ChestDatasetRaw, self).__init__(csv_file, cfg)
    
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        path = self.img_names[index]
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
        label = self.labels[index]
        return index, image, label

class PesudoSubset(Dataset):
    def __init__(self, subset, cfg):
        super(PesudoSubset, self).__init__()
        self.subset = subset
        self.weak = albumentations.Compose([
            albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
            ToTensorV2()
        ])
        self.strong = albumentations.Compose([
            albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.OneOf([
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=5),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85,),
            albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
            ToTensorV2()
            ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        _, image, label = self.subset[index]
        weak = self.weak(image=image)['image']
        strong = self.strong(image=image)['image']
        return index, weak, strong, label

class OriginialSubset(Dataset):
    def __init__(self, subset, cfg):
        super(OriginialSubset, self).__init__()
        self.subset = subset
        self.orginal = albumentations.Compose([
            albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
            albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        _, image, label = self.subset[index]
        image = self.orginal(image=image)['image']
        return index, image, label

    def __len__(self):
        return len(self.subset)

class SrongAugmentedSubset(Dataset):
    def __init__(self, subset, cfg):
        super(SrongAugmentedSubset, self).__init__()
        self.subset = subset
        self.strong = albumentations.Compose([
            albumentations.Resize(height=cfg['dataset']['resize']['height'], width=cfg['dataset']['resize']['width']),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.OneOf([
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=5),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85,),
            albumentations.Normalize(cfg['dataset']['mean'], cfg['dataset']['std']),
            ToTensorV2()
            ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        _, image, label = self.subset[index]
        image = self.strong(image=image)['image']
        return index, image, label

    def __len__(self):
        return len(self.subset)

