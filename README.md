# FedLSM
1. Download the ISIC dataset from [ISIC 2018 Task3](https://www.kaggle.com/datasets/shonenkov/isic2018) and Chest-Xray14 NIH dataset from [Chest-Xray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)

2. Run the scprits (scripts/isic2018_prepare_fl.py and scripts/chestxray14_prepare_fl.py) to generate FL datasets, the corresponding configure file is in configs/cxr/scripts_conf.yaml and configs/isic/scripts_conf.yaml

3. Run the training code.
python3 train.py --config "/home/project/FedLSM/configs/cxr/run_conf.yaml"
python3 train.py --config "/home/project/FedLSM/configs/isic/run_conf.yaml"

4. Test the result
python3 test.py --config "config file"  \
--resume_path "model path" \
--test_csv_path "test csv file path"

Please cite our paper if you find this code useful for your research.
# citation
@InProceedings{Deng2023,
  author    = {Deng, Zhipeng and Luo, Luyang and Chen, Hao},
  title     = {Scale Federated Learning for Label Set Mismatch in Medical Image Classification},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year      = {2023},
}


