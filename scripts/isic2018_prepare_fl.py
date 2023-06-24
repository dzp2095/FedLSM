import os
import pandas as pd
import random

from glob import glob

import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from pathlib import Path

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/isic/scripts_conf.yaml")))

data_folder = config["data_folder"]
raw_data_folder = config["raw_data_folder"]
client_num = config["client_num"]
class_num = config["class_num"]
random.seed(config["seed"])

target_folder = f'{data_folder}/fl'
Path(target_folder).mkdir(parents=True, exist_ok=True)

all_xray_df = pd.read_csv(f'{raw_data_folder}/ISIC2018_Task3_Training_GroundTruth.csv')
all_image_paths = {Path(x).stem: x for x in glob(os.path.join(raw_data_folder, 'ISIC*', '*', '*.jpg'))}

all_xray_df['path'] = all_xray_df['image'].map(all_image_paths.get)


train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

train, test = train_test_split(all_xray_df,test_size=test_ratio, random_state=523)
train, val = train_test_split(train,test_size=val_ratio/(train_ratio+val_ratio), random_state=523)


class_names =  ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
clients = [[] for _ in range(client_num)]

random.shuffle(class_names)
clients = [[] for _ in range(client_num)]

for i in range(0, len(class_names)):
    clients[i%client_num].append(class_names[i])

for client in clients:
    client+= random.sample(class_names, class_num - len(client))


client_nums = [train.shape[0]//client_num] * client_num
client_nums[-1] += train.shape[0]-sum(client_nums)

for i, client in enumerate(clients):
    train_csv = train.sample(n=client_nums[i],random_state=config["seed"])
    train=train.drop(train_csv.index)
    Path(f'{target_folder}/client_{i}').mkdir(parents=True, exist_ok=True)
    train_csv.to_csv(f'{target_folder}/client_{i}/fully_train.csv', index=False)
    not_concerned_diseases = set(class_names).difference(set(client))
    for disease in not_concerned_diseases:
        train_csv.loc[:,disease] = 0
    train_csv.to_csv(f'{target_folder}/client_{i}/partial_train.csv', index=False)

test.to_csv(f'{target_folder}/test.csv', index=False)
val.to_csv(f'{target_folder}/val.csv', index=False)