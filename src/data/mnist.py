import random
from collections import defaultdict

import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical


class ColoredMNIST(Dataset):
    def __init__(self, file_path, data_config, target=False):

        self.y_to_c, self.max_c, self.label_to_y = self.load_data_config(data_config)

        self.data = {
            'train': self.load_data(file_path=file_path, train=True,
                                    label_to_y=self.label_to_y),
            'test': self.load_data(file_path=file_path, train=False,
                                   label_to_y=self.label_to_y),
        }

        # define the training, val and testing environments
        self.envs = []

        # use the first half for train env 0
        self.envs.append(self.create_env(
            data=self.data['train'], bias_ratio=0.1, start_ratio=0, end_ratio=0.5))

        # use the second half for train env 1
        self.envs.append(self.create_env(
            data=self.data['train'], bias_ratio=0.2 if target else 0.1,
            start_ratio=0.5, end_ratio=1))

        # use the first half for val env
        self.envs.append(self.create_env(
            data=self.data['test'], bias_ratio=0.1, start_ratio=0, end_ratio=0.5))

        # use the second half for test env
        self.envs.append(self.create_env(
            data=self.data['test'], bias_ratio=0.9, start_ratio=0.5, end_ratio=1))

        self.length = sum([len(env['idx_list']) for env in self.envs])

        # not evaluating worst-case performance of mnist
        self.val_att_idx_dict = None
        self.test_att_idx_dict = None


    def load_data_config(self, data_config):
        '''
            The first segment represent the digits
            The second segment represent the colors that are corerlated with each
            of the digits
            The last segment represent the maximum number of colors

            Examples:
            EVEN: MNIST_02468_01234_5
            ODD: MNIST_13579_01234_5
        '''
        label, color, max_c = tuple(data_config.split('_'))
        y_list = [int(y) for y in label]
        c_list = [int(c) for c in color]
        max_c = int(max_c)

        if y_list is not None:
            # We have specified the digits that we want to classify in y_list
            # Here we map each digit into a 0-based index list
            label_to_y = {}
            for i, y in enumerate(y_list):
                label_to_y[y] = i

            # each digit is correlated with a specified color id
            y_to_c = {}
            for i, c in enumerate(c_list):
                y_to_c[i] = c
        else:
            # use all ten digits if not specified
            label_to_y = dict(zip(list(range(10)), list(range(10))))
            y_to_c = dict(zip(list(range(10)), list(range(10))))

        return y_to_c, max_c, label_to_y


    def load_data(self, file_path, train, label_to_y):
        # load the data based on the current label_to_dict
        mnist = datasets.MNIST(file_path, train=train, download=True)

        data = defaultdict(list)

        for x, y in zip(mnist.data, mnist.targets):
            if int(y) in label_to_y:
                data[label_to_y[int(y)]].append(x)

        # shuffle data
        random.seed(0)
        for k, v in data.items():
            random.shuffle(v)
            data[k] = torch.stack(v, dim=0)

        return data


    def create_env(self, data, bias_ratio, start_ratio, end_ratio):
        '''
            Create an environment using data from the start_ratio to the end_ratio
        '''
        images = []
        labels = []

        for cur_label, cur_images in data.items():
            start = int(start_ratio * len(cur_images))
            end = int(end_ratio * len(cur_images))

            images.append(cur_images[start:end])
            labels.append((torch.ones(end-start) * cur_label).long())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        return self.make_environment(images, labels, bias_ratio)


    def __len__(self):
        return self.length


    def make_environment(self, images, labels, e):
        '''
            https://github.com/facebookresearch/InvariantRiskMinimization
        '''
        # different from the IRM repo, here the labels are already binarized
        images = images.reshape((-1, 28, 28))

        # change label with prob 0.25
        n_labels = len(torch.unique(labels))
        prob_label = torch.ones((n_labels, n_labels)).float() * (0.25 /
                                                                 (n_labels - 1))
        for i in range(n_labels):
            prob_label[i, i] = 0.75

        labels_prob = torch.index_select(prob_label, dim=0, index=labels)
        labels = Categorical(probs=labels_prob).sample()

        # assign the color variable
        prob_color = torch.ones((n_labels, n_labels)).float() * (e /
                                                                 (n_labels - 1))
        for i in range(n_labels):
            prob_color[i, i] = 1 - e

        color_prob = torch.index_select(prob_color, dim=0, index=labels)
        color = Categorical(probs=color_prob).sample()

        # Apply the color to the image by zeroing out the other color channel
        output_images = torch.zeros((len(images), self.max_c, 28, 28))

        idx_dict = defaultdict(list)
        for i in range(len(images)):
            idx_dict[int(labels[i])].append(i)
            output_images[i, self.y_to_c[color[i].item()], :, :] = images[i]

        cor = color.float()

        idx_list = list(range(len(images)))

        return {
            'images': (output_images.float() / 255.),
            'labels': labels.long(),
            'idx_dict': idx_dict,
            'idx_list': idx_list,
            'cor': cor,
        }


    def __getitem__(self, keys):
        '''
            @params [support, query]
            support=[(label, y, idx, env)]
            query=[(label, y, idx, env)]
        '''
        idx = []

        # without reindexing y
        idx = []
        for key in keys:
            env_id = int(key[1])
            idx.append(key[0])

        return {
            'X': self.envs[env_id]['images'][idx],
            'Y': self.envs[env_id]['labels'][idx],
            'C': self.envs[env_id]['cor'][idx],
            'idx': torch.tensor(idx).long(),
        }


    def get_all_y(self, env_id):
        return self.envs[env_id]['labels'].tolist()


    def get_all_c(self, env_id):
        return self.envs[env_id]['cor'].tolist()
