import random
import os

from torch.utils.data import Sampler

from data.mnist import ColoredMNIST
from data.beer import BeerReview
from data.bird import Bird
from data.ask2me import ASK2ME
from data.celeba import Celeba


class EnvSampler(Sampler):
    def __init__(self, num_batches, batch_size, env_id, idx_list, seed=0):
        '''
            Sample @num_episodes episodes for each epoch. If set to -1, iterate
            through the entire dataet (test mode)

            env_id specifies the env that we are sampling from
                0: train_env_a
                1: train_env_b
                2: val_env
                3: test_env

            idx_list is the list of data index
        '''
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.env_id = env_id
        self.idx_list = list(idx_list)

        if self.num_batches == -1:
            self.length = ((len(self.idx_list) + self.batch_size - 1) // self.batch_size)

        else:
            self.length = self.num_batches

    def __iter__(self):
        '''
            Return a list of keys
        '''
        if self.num_batches == -1:
            # iterate through the dataset sequentially
            # for testing
            random.shuffle(self.idx_list)

            # sample through the data
            for i in range(self.length):
                start = i * self.batch_size
                end = min((i+1) * self.batch_size, len(self.idx_list))

                # provide the idx and the env information to the dataset
                yield [(idx, self.env_id) for idx in self.idx_list[start:end]]

        else:
            for _ in range(self.num_batches):
                if self.batch_size < len(self.idx_list):
                    yield [(idx, self.env_id) for idx in
                           random.sample(self.idx_list, self.batch_size)]
                else:
                    # if the number of examples is less than a batch
                    yield [(idx, self.env_id) for idx in self.idx_list]

    def __len__(self):
        return self.length


def is_textdata(dataset):
    return (dataset[:4] == 'beer' or dataset[:6] == 'ask2me')


def get_dataset(data_name, is_target=None, vocab=None):
    os.makedirs('./datasets', exist_ok=True)
    if 'MNIST' in data_name:
        data = ColoredMNIST('./datasets/mnist', data_config=data_name[6:],
                            target=is_target)
        return data

    if data_name[:4] == 'beer':
        # look: beer_0
        # aroma: beer_1
        # palate: beer_2
        data = BeerReview('./datasets/beer', aspect=data_name[5:], vocab=vocab,
                          target=is_target)
        return data

    if data_name[:4] == 'bird':
        data = Bird('./datasets/bird/waterbird_complete95_forest2water2/',
                    task=data_name[5:])

        return data

    if data_name[:6] == 'ask2me':
        data = ASK2ME('./datasets/ask2me', task=data_name[7:], vocab=vocab,
                      target=is_target)

        return data

    if data_name[:6] == 'celeba':
        _, tar_att, env_att = tuple(data_name.split(':'))
        data = Celeba('./datasets/celeba', tar_att, env_att, is_target)

        return data

    raise ValueError('dataset {} is not supported'.format(data_name))
