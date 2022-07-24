import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Celeba(Dataset):
    def __init__(self, file_path, tar_att, env_att, target):
        # load the official train / val / test splits
        self.data = {}
        for split in ['train', 'valid', 'test']:
            self.data[split] = datasets.CelebA(file_path, split=split,
                                               target_type='attr',
                                               transform=None,
                                               target_transform=None,
                                               download=True)

        # get the idx of label and bias attributes
        self.label_idx = self.data['train'].attr_names.index(tar_att)
        self.bias_idx  = self.data['train'].attr_names.index(env_att)

        # Choose a random subset of the data (depending on whether this is the
        # source task or the target task).
        # This ensures that the target and source tasks don't have overlapping
        # data points.
        split_to_idx_list = self.split_to_src_and_tar(self.data, target)

        # define the training, val and testing environments
        self.envs = []

        self.add_env(data=self.data['train'],
                     idx_list=split_to_idx_list['train'], env_att_value=0)

        self.add_env(data=self.data['train'],
                     idx_list=split_to_idx_list['train'],
                     env_att_value=0 if target else 1)

        self.add_env(data=self.data['valid'],
                     idx_list=split_to_idx_list['valid'], env_att_value=0)

        self.add_env(data=self.data['test'],
                     idx_list=split_to_idx_list['test'], env_att_value=1)

        # for env in self.envs:
        #     print('size: ', len(env['idx_list']))

        self.val_att_idx_dict = None

        # compute correlation between each attribute and the target attribute
        # only for the test set
        if target:
            self.test_att_idx_dict = self.get_att_idx_dict(
                data=self.data['test'], idx_list=self.envs[3]['idx_list'])
        else:
            self.test_att_idx_dict = None

        self.length = sum([len(env['idx_list']) for env in self.envs])

    @staticmethod
    def split_to_src_and_tar(data, target):
        '''
          use half of the data for the source task and half of the data for the
          target task
        '''
        random.seed(1)  # need to fix the seed to ensure fixed split
        split_to_idx_list = {}

        for split in data.keys():
            split_len = len(data[split].attr)
            split_to_idx_list[split] = list(range(split_len))
            random.shuffle(split_to_idx_list[split])

            if target:
                split_to_idx_list[split] = split_to_idx_list[split][:split_len//2]
            else:
                split_to_idx_list[split] = split_to_idx_list[split][split_len//2:]

        return split_to_idx_list

    def add_env(self, data, idx_list, env_att_value):
        '''
            Go through the provided idx_list in the given data,
            add data that has the given env_att_value into the new env
        '''
        new_env = {
            'idx_list': [],
            'data': data,
        }
        for idx in idx_list:
            if data.attr[idx, self.bias_idx] == env_att_value:
                new_env['idx_list'].append(idx)
        self.envs.append(new_env)

    def get_att_idx_dict(self, data, idx_list):
        '''
            For each unknown attribute,
            we compute a dictionary that maps the label_attribute_value pair
            into the list of example indicies.
            This will help us evaluate the worst-group performance at test time.
        '''
        att_idx_dict = {}

        attr_names = data.attr_names.copy()

        if attr_names[-1] == '':  # remove the redundant name
            attr_names = attr_names[:-1]

        for idx, att in enumerate(attr_names):
            if idx == self.label_idx:
                continue

            if idx == self.bias_idx:
                continue

            data_dict = {
                '0_0': [],
                '0_1': [],
                '1_0': [],
                '1_1': [],
            }

            # go through only the att label
            for i, attrs in enumerate(data.attr):
                if i not in idx_list:
                    continue

                k = '{}_{}'.format(attrs[self.label_idx], attrs[idx])
                data_dict[k].append(i)

            # # print data stats
            # print('{:>20}'.format(att), end=' ')
            # for k, v in data_dict.items():
            #     print(k, ' ', '{:>8}'.format(len(v)), end=', ')

            # print()

            att_idx_dict[att] = data_dict

        return att_idx_dict

    def __len__(self):
        return self.length

    def __getitem__(self, keys):
        idx = []
        for key in keys:
            env_id = int(key[1])  # this doesn't matter for Pubmed data
            idx.append(key[0])

        batch = {}
        batch['Y'] = self.envs[env_id]['data'].attr[:, self.label_idx][idx]
        batch['C'] = self.envs[env_id]['data'].attr[:, self.bias_idx][idx]
        batch['idx'] = torch.tensor(idx).long()

        img2tensor = transforms.ToTensor()
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        x = []
        for i in idx:
            img = img2tensor(self.envs[env_id]['data'][i][0])
            img = transform(img)
            x.append(img)

        batch['X'] = torch.stack(x)

        return batch

    def get_all_y(self, env_id):
        return self.envs[env_id]['data'].attr[self.envs[env_id]['idx_list'], self.label_idx].tolist()

    def get_all_c(self, env_id):
        return self.envs[env_id]['data'].attr[self.envs[env_id]['idx_list'],
                                              self.bias_idx].tolist()

    def get_all_att(self, env_id):
        return self.envs[env_id]['data'].attr

    def get_att_names(self, i):
        return self.envs[0]['data'].attr_names[i]
