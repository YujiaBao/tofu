'''
    Contrast differnt source enviornments using the PI algorithm.
    http://arxiv.org/abs/2105.12628
'''
import copy
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from termcolor import colored

from data import EnvSampler
from model import get_model
from tofu.utils import print_res, train_dro_loop, test_loop, to_cuda, \
    squeeze_batch


def train_loop(train_loader, model, opt, ep, args):
    stats = {}
    for k in ['acc', 'loss']:
        stats[k] = []

    step = 0
    for batch in train_loader:
        # work on each batch
        model['ebd'].train()
        model['clf'].train()

        batch = to_cuda(squeeze_batch(batch))

        x = model['ebd'](batch['X'])
        y = batch['Y']

        acc, loss = model['clf'](x, y, return_pred=False, grad_penalty=False)

        opt.zero_grad()
        loss.backward()
        opt.step()

        stats['acc'].append(acc)
        stats['loss'].append(loss.item())

    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v).float()).item()

    return stats


def print_pretrain_res(train_res, test_res, ep, env_id):
    print(f'PI env {env_id} epoch {ep:>3} '
          f'train {colored("acc", "blue")} {train_res["acc"]:>6.4f} '
          f'{colored("loss", "yellow")} {train_res["loss"]:>6.4f} '
          f'val {colored("acc", "blue")} {test_res["acc"]:>6.4f} '
          f'{colored("loss", "yellow")} {test_res["loss"]:>6.4f}',
          flush=True)


def train_env_specific_model(data, train_env_id, val_env_id, args):
    '''
        Train an environment specific classifier on one environment.
        Return the learned model.
    '''
    train_loader = DataLoader(
        data,
        sampler=EnvSampler(args.num_batches, args.batch_size, train_env_id,
                           data.envs[train_env_id]['idx_list']),
        num_workers=10)

    # here we still load it in random sampling mode
    # because inference through all data can be slow
    # and we don't need that
    val_loader = DataLoader(
        data,
        sampler=EnvSampler(args.num_batches, args.batch_size, val_env_id,
                           data.envs[val_env_id]['idx_list']),
        num_workers=10)

    cur_model, cur_opt = get_model(args, data)

    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Start training classifier on train env {train_env_id}", flush=True)

    best_acc = -1
    best_model = {}
    cycle = 0

    # start training the env specific model
    for ep in range(args.num_epochs):
        train_res = train_loop(train_loader, cur_model, cur_opt, ep, args)

        with torch.no_grad():
            # evaluate on the other training environment
            val_res = test_loop(val_loader, cur_model, ep, args)

        print_pretrain_res(train_res, val_res, ep, train_env_id)

        if val_res['acc'] > best_acc:
            best_acc = val_res['acc']
            cycle = 0
            # save best ebd
            for k in 'ebd', 'clf':
                best_model[k] = copy.deepcopy(cur_model[k].state_dict())
        else:
            cycle += 1

        if cycle == args.patience:
            break

    # load best model
    for k in 'ebd', 'clf':
        cur_model[k].load_state_dict(best_model[k])

    return cur_model


def train_robust_source_model(data, pretrain_res, model, opt, args):
    '''
        Use dro to learn a robust source model
    '''
    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Use DRO to learn a robust source classifier", flush=True)

    # prepare the training data from the prediction results
    train_loaders = []

    for env_id in range(len(pretrain_res)):
        train_loaders.append(DataLoader(
            data,
            sampler=EnvSampler(args.num_batches, args.batch_size, env_id,
                               pretrain_res[env_id]['correct_idx']),
            num_workers=10))

        train_loaders.append(DataLoader(
            data,
            sampler=EnvSampler(args.num_batches, args.batch_size, env_id,
                               pretrain_res[env_id]['wrong_idx']),
            num_workers=10))

    val_env = 2
    val_loader = DataLoader(data,
                            sampler=EnvSampler(-1, args.batch_size, val_env,
                                               data.envs[val_env]['idx_list']),
                            num_workers=10)

    best_acc = -1
    best_val_res = None
    best_model = {}
    cycle = 0
    for ep in range(args.num_epochs):
        train_res = train_dro_loop(train_loaders, model, opt, ep, args)

        with torch.no_grad():
            # validation
            val_res = test_loop(val_loader, model, ep, args)

        print_res(train_res, val_res, ep)

        if min(train_res['worst_acc'], val_res['acc']) > best_acc:
            best_acc = min(train_res['worst_acc'], val_res['acc'])
            best_val_res = val_res
            best_train_res = train_res
            cycle = 0
            # save best ebd
            for k in 'ebd', 'clf':
                best_model[k] = copy.deepcopy(model[k].state_dict())
        else:
            cycle += 1

        if cycle == args.patience:
            break


    # load best model
    for k in 'ebd', 'clf':
        model[k].load_state_dict(best_model[k])

    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Finished DRO on the source task", flush=True)


def get_partition_loaders(train_data, pretrain_res, args):
    '''
        Given the dataset and the pretrain_res,
        create the data loader for training the partition model.
        The partition loaders will be organized in the following order:
        ENV_e0_LABEL_y0_correct
        ENV_e0_LABEL_y0_mistake
        ENV_e0_LABEL_y1_correct
        ENV_e0_LABEL_y1_mistake
        ENV_e1_LABEL_y0_correct
        ENV_e1_LABEL_y0_mistake
        ENV_e1_LABEL_y1_correct
        ENV_e1_LABEL_y1_mistake
    '''
    # create data loader from each partition x label
    train_partition_loaders, val_partition_loaders = [], []

    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Create groups (based on the prediction correctness)"
          f" for training the partition model", flush=True)

    for env in range(len(pretrain_res)):
        # look at each environment, each label
        groups = {}
        label_list = train_data.get_all_y(env)
        for idx, label in zip(train_data.envs[env]['idx_list'], label_list):
            if label not in groups:
                groups[label] = {
                    'correct': [],
                    'mistake': [],
                }

            if idx in pretrain_res[env]['correct_idx']:
                groups[label]['correct'].append(idx)
            elif idx in pretrain_res[env]['wrong_idx']:
                groups[label]['mistake'].append(idx)
            else:
                raise ValueError('unknown idx')

        for group in groups.values():
            # each group contains pos (yes) and neg (no)
            # each group corresponds to examples from one label value
            # use 70% for training, 30% for validation
            if min(len(group['correct']), len(group['mistake'])) * 0.3 < 1:
                continue

            for k in ['correct', 'mistake']:
                train_partition_loaders.append(
                    DataLoader(train_data,
                               sampler=EnvSampler(args.num_batches, args.batch_size,
                                                  env,
                                                  group[k][:int(len(group[k])*0.7)]),
                               num_workers=2),
                )

                val_partition_loaders.append(
                    DataLoader(train_data,
                               sampler=EnvSampler(args.num_batches, args.batch_size,
                                                  env,
                                                  group[k][int(len(group[k])*0.7):]),
                               num_workers=2),
                )

    return train_partition_loaders, val_partition_loaders


def contrast_source_envs(data, model, opt, args):
    '''
        Use the PI algorithm to contrast the training environments
    '''
    print('Predict then Interpolate (PI)')
    # training the environment-specific classifier
    # assuming two training enviornments
    train_env = [0, 1]
    models = []
    for env_id in train_env:
        # pick a random other train env for validation (early stopping)
        rest = train_env.copy()
        rest.remove(env_id)
        val_env_id = random.choice(rest)

        models.append(train_env_specific_model(
            data=data,
            train_env_id=env_id,
            val_env_id=val_env_id,
            args=args
        ))

    # load the training environments in inference loaders
    train_loaders = []
    for env_id in train_env:
        train_loaders.append(DataLoader(
            data,
            sampler=EnvSampler(-1, args.batch_size, env_id,
                               data.envs[env_id]['idx_list']),
            num_workers=10))

    # split the dataset based on the model predictions
    pretrain_res = []
    for i in train_env:
        for j in train_env:
            if i == j:
                continue
            pretrain_res.append(
                test_loop(train_loaders[i], models[j], args, True)
            )

    train_robust_source_model(data, pretrain_res, model, opt, args)

    return get_partition_loaders(data, pretrain_res, args)

