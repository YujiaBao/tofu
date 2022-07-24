import os
import json

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Bird(Dataset):
    def __init__(self, file_path, task):
        #######################
        # load all environments
        #######################
        self.envs = []

        # two training environments have different correlation w.r.t. label
        self.envs.append(Bird.load_json(os.path.join(file_path,
                                                     '{}_env_train1.json'.format(task))))
        self.envs.append(Bird.load_json(os.path.join(file_path,
                                                     '{}_env_train2.json'.format(task))))

        # validation has same distribution as train1
        self.envs.append(Bird.load_json(os.path.join(file_path,
                                                     '{}_env_val.json'.format(task))))

        # testing environments is perfectly balanced for evaluation
        self.envs.append(Bird.load_json(os.path.join(file_path,
                                                     '{}_env_test.json'.format(task))))

        self.file_path = file_path

        # not evaluating worst-case performance for this dataset
        self.val_att_idx_dict = None
        self.test_att_idx_dict = None

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            data = {'y': [], 'c': [], 'x': []}

            all_text = []

            for line in f:
                example = json.loads(line)
                data['y'].append(example['y'])
                data['x'].append(example['x'])
                data['c'].append(example['c'])

            data['y'] = torch.tensor(data['y'])
            data['c'] = torch.tensor(data['c'])
            data['idx_list'] = list(range(len(data['y'])))

        return data

    def __len__(self):
        return self.length

    def __getitem__(self, keys):
        idx = []
        for key in keys:
            env_id = int(key[1])
            idx.append(key[0])

        # get labels
        batch = {}
        batch['Y'] = self.envs[env_id]['y'][idx]
        batch['C'] = self.envs[env_id]['c'][idx]
        batch['idx'] = torch.tensor(idx).long()

        img2tensor = transforms.ToTensor()
        # transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = Bird.get_transform_cub()

        x = []
        for i in idx:
            with Image.open(os.path.join(
                self.file_path, self.envs[env_id]['x'][i])) as img:
                img = transform(img)
                x.append(img)

        batch['X'] = torch.stack(x)

        return batch

    def get_all_y(self, env_id):
        return self.envs[env_id]['y'].tolist()

    def get_all_c(self, env_id):
        return self.envs[env_id]['c'].tolist()

    @staticmethod
    def get_transform_cub():
        scale = 256.0/224.0
        target_resolution = (224, 224)  # for resnet 50
        assert target_resolution is not None

        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform
