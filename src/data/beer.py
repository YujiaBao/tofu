import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset
import torchtext


class BeerReview(Dataset):
    def __init__(self, file_path, aspect, vocab=None, target=None):
        # load the beer review data
        # iterate through the envs
        self.envs = []
        self.length = 0

        if target:
            # target task only uses env_1 and env_1_val
            idx_to_env_name = [
                'env_1',     # train env
                'env_1',     # train env
                'env_1_val', # val env
                'env_2'      # test env
            ]
        else:
            # source task has access to both train envs
            idx_to_env_name = [
                'env_1',     # train env
                'env_0',     # train env
                'env_1_val', # val env
                'env_2'      # test env
            ]

        for i in range(4):
            data, words = BeerReview.load_json(os.path.join(
                file_path, f'art_aspect_{aspect}_{idx_to_env_name[i]}.json'))

            self.envs.append(data)

            self.length += len(data['y'])

        if vocab is not None:
            self.vocab = vocab
        else:
            # get word embeddings from fasttext
            self.vocab = torchtext.vocab.FastText()
            for special in ['<pad>', '<unk>', '<art_negative>',
                            '<art_positive>']:
                if special not in self.vocab.stoi:
                    self.vocab.stoi[special] = len(self.vocab.itos)
                    self.vocab.itos.append(special)
                    self.vocab.vectors = torch.cat(
                        [self.vocab.vectors, torch.rand(1, 300)], dim=0)

        # not evaluating worst-case performance for beer
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
                data['x'].append(example['text'])
                data['c'].append(example['c'])
                all_text.extend(example['text'].split())

            data['y'] = torch.tensor(data['y'])
            data['c'] = torch.tensor(data['c'])
            data['idx_list'] = list(range(len(data['y'])))

        return data, all_text

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

        # convert text into a dictionary of np arrays
        text_list = []
        for i in idx:
            text_list.append(self.envs[env_id]['x'][i].split())

        text_len = np.array([len(text) for text in text_list])
        max_text_len = max(text_len)

        # initialize the big numpy array by <pad>
        text = self.vocab.stoi['<pad>'] * np.ones(
            [len(text_list), max_text_len], dtype=np.int64)

        # convert each token to its corresponding id
        for i, t in enumerate(text_list):
            text[i, :len(t)] = [
                self.vocab.stoi[x] if x in self.vocab.stoi \
                else self.vocab.stoi['<unk>'] for x in t]

        batch['X'] = torch.tensor(text)
        batch['X_len'] = torch.tensor(text_len).long()

        return batch

    def get_all_y(self, env_id):
        return self.envs[env_id]['y'].tolist()

    def get_all_c(self, env_id):
        return self.envs[env_id]['c'].tolist()
