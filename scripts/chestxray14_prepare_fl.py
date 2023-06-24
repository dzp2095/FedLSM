import os
import pandas as pd
from glob import glob
import re

import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/cxr/scripts_conf.yaml")))

data_folder = config["data_folder"]
raw_data_folder = config["raw_data_folder"]
random.seed(config["seed"])
target_folder = f'{data_folder}/fl'
Path(target_folder).mkdir(parents=True, exist_ok=True)

all_xray_df = pd.read_csv(f'{raw_data_folder}/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join(raw_data_folder, 'images*', '*', '*.png'))}

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

# read test set
with open(f'{raw_data_folder}/test_list.txt') as f:
    test_files = [line.rstrip() for line in f]

# read train/val set
with open(f'{raw_data_folder}/train_val_list.txt') as f:
    train_val_files = [line.rstrip() for line in f]

train_ratio = 0.7
val_ratio = 0.1
train_files, val_files = train_test_split(train_val_files,test_size=val_ratio/(train_ratio+val_ratio), random_state=523)

client_num = config["client_num"]
class_num = config["class_num"]

# make fl dataset

class_names =  ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
            "Mass", "Nodule","Pleural_Thickening", "Pneumonia", "Pneumothorax"]

random.shuffle(class_names)
clients = [[] for _ in range(client_num)]

for i in range(0, len(class_names)):
    clients[i%client_num].append(class_names[i])

for client in clients:
    client+= random.sample(class_names, class_num - len(client))
print(clients)

train_fl_files = [train_files[i:i + len(train_files)//client_num] for i in range(0, len(train_files), len(train_files)//client_num)]

if (len(train_fl_files)>client_num):
      val_files = val_files + train_fl_files.pop(client_num)

test_csv = all_xray_df[all_xray_df['Image Index'].isin(test_files)]
val_csv = all_xray_df[all_xray_df['Image Index'].isin(val_files)]

test_csv.to_csv(f'{target_folder}/test.csv', index=False)
val_csv.to_csv(f'{target_folder}/val.csv', index=False)

for i, client_file in enumerate(train_fl_files):
    train_csv = all_xray_df[all_xray_df['Image Index'].isin(client_file)]
    Path(f'{target_folder}/client_{i}').mkdir(parents=True, exist_ok=True)
    # save fully labeled dataset
    train_csv.to_csv(f'{target_folder}/client_{i}/fully_train.csv', index=False)
    # configure partial labeled dataset
    concerned_diseases = ''.join(map(lambda x : str(x)+"|", clients[i]))
    concerned_diseases = concerned_diseases[:-1]

    def remove_not_interested_label(target):
        if re.findall(concerned_diseases, target):
            labels = ""
            for label in target.split("|"):
                if label in clients[i]:
                    labels += label + "|"
            labels = labels[:-1]
            target = labels
        else:
            target = "No Finding"
        return target

    train_csv.loc[:,'Finding Labels'] = train_csv['Finding Labels'].apply(remove_not_interested_label)
    train_csv.to_csv(f'{target_folder}/client_{i}/partial_train.csv', index=False)
