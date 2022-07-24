'''
    Learning the unstable feature representation
'''

import copy
from datetime import datetime

import torch
from termcolor import colored

from tofu.utils import to_cuda, squeeze_batch


def compute_l2(XS, XQ):
    '''
        Compute the pairwise l2 distance
        @param XS (support x): support_size x ebd_dim
        @param XQ (support x): query_size x ebd_dim

        @return dist: query_size x support_size

    '''
    diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
    dist = torch.norm(diff, dim=2)

    return dist ** 2


def train_partition_loop(train_loaders, model, opt, ep, args):
    '''
        The train_loaders is a list of data loaders.
        It is organized as
        [
            ENV_e0_LABEL_y0_correct
            ENV_e0_LABEL_y0_mistake
            ENV_e0_LABEL_y1_correct
            ENV_e0_LABEL_y1_mistake
            ENV_e1_LABEL_y0_correct
            ENV_e1_LABEL_y0_mistake
            ENV_e1_LABEL_y1_correct
            ENV_e1_LABEL_y1_mistake
        ]
        We want examples from the same prediction outcome to have similar
        feature representation.
    '''
    stats = {}
    for k in ['loss', 'dis_pos', 'dis_neg', 'dis_cross']:
        stats[k] = []

    step = 0

    n_group_pairs = len(train_loaders)
    # every two loaders are paired together

    for batches in zip(*train_loaders):
        # work on each batch
        model['ebd'].train()

        cur_loss = []
        dis_pos = []
        dis_neg = []
        dis_cross = []
        for i in range(n_group_pairs//2):

            x_pos = to_cuda(squeeze_batch(batches[i*2]))['X']
            x_neg = to_cuda(squeeze_batch(batches[i*2+1]))['X']

            min_size = min(len(x_pos), len(x_neg))
            x_pos = x_pos[:min_size]
            x_neg = x_neg[:min_size]

            ebd_pos = model['ebd'](x_pos)
            ebd_neg = model['ebd'](x_neg)

            diff_pos_pos = compute_l2(ebd_pos, ebd_pos)
            diff_pos_neg = compute_l2(ebd_pos, ebd_neg)
            diff_neg_neg = compute_l2(ebd_neg, ebd_neg)

            dis_pos.append(torch.mean(diff_pos_pos.detach()).item())
            dis_neg.append(torch.mean(diff_neg_neg.detach()).item())
            dis_cross.append(torch.mean(diff_pos_neg.detach()).item())

            loss = (
                torch.mean(torch.max(torch.zeros_like(diff_pos_pos),
                                    diff_pos_pos - diff_pos_neg +
                                    torch.ones_like(diff_pos_pos) *
                                     args.thres)))

            loss /= n_group_pairs

            cur_loss.append(loss.item())
            loss.backward()

        opt.step()
        opt.zero_grad()
        loss = sum(cur_loss)

        stats['loss'].append(loss)
        stats['dis_pos'].append(sum(dis_pos) / len(dis_pos))
        stats['dis_neg'].append(sum(dis_neg) / len(dis_neg))
        stats['dis_cross'].append(sum(dis_cross) / len(dis_cross))

    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v).float()).item()

    return stats


def test_partition_loop(test_loaders, model, ep, args):
    stats = {}
    for k in ['loss']:
        stats[k] = []

    step = 0

    n_group_pairs = len(test_loaders)
    # every two loaders are paired together

    for batches in zip(*test_loaders):
        # work on each batch
        model['ebd'].eval()

        cur_loss = []
        for i in range(n_group_pairs//2):

            x_pos = to_cuda(squeeze_batch(batches[i*2]))['X']
            x_neg = to_cuda(squeeze_batch(batches[i*2+1]))['X']

            min_size = min(len(x_pos), len(x_neg))
            x_pos = x_pos[:min_size]
            x_neg = x_neg[:min_size]

            ebd_pos = model['ebd'](x_pos)
            ebd_neg = model['ebd'](x_neg)

            diff_pos_pos = compute_l2(ebd_pos, ebd_pos)
            diff_pos_neg = compute_l2(ebd_pos, ebd_neg)
            diff_neg_neg = compute_l2(ebd_neg, ebd_neg)

            loss = (
                torch.mean(torch.max(torch.zeros_like(diff_pos_pos),
                                    diff_pos_pos - diff_pos_neg +
                                    torch.ones_like(diff_pos_pos) * args.thres)) +
                torch.mean(torch.max(torch.zeros_like(diff_neg_neg),
                                    diff_neg_neg - diff_pos_neg +
                                    torch.ones_like(diff_neg_neg) * args.thres))
            )

            loss /= n_group_pairs
            cur_loss.append(loss.item())


        loss = sum(cur_loss)

        stats['loss'].append(loss)

    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v).float()).item()

    return stats


def print_partition_res(train_res, val_res, ep):
    print(f'epoch {ep:>3}, train '
          f'{colored("pos_d", "blue")} {train_res["dis_pos"]:>10.4f} '
          f'{colored("neg_d", "blue")} {train_res["dis_neg"]:>10.4f} '
          f'{colored("cross", "blue")} {train_res["dis_cross"]:>10.4f} '
          f'{colored("loss", "yellow")} {train_res["loss"]:8.6f} '
          f'val {colored("loss", "yellow")} {val_res["loss"]:>8.6f}',
          flush=True)


def train_partition(train_partition_loaders,
                    val_partition_loaders,
                    partition_model, opt, args):
    '''
        Given the train_partition_loaders and val_partition_loaders,
        train an unstable feature representation such that examples with the
        same prediction outcome are clustered together.
    '''
    best_loss = float('inf')
    best_val_res = None
    best_model = {}
    cycle = 0

    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Learning the unstable feature representaiton",
          flush=True)

    for ep in range(args.num_epochs):

        train_res = train_partition_loop(
            train_partition_loaders, partition_model, opt, ep, args)

        with torch.no_grad():
            # validation
            val_res = test_partition_loop(
                val_partition_loaders, partition_model, ep, args)

        print_partition_res(train_res, val_res, ep)

        if val_res['loss'] < best_loss:
            best_loss = val_res['loss']
            best_val_res = val_res
            best_train_res = train_res
            cycle = 0
            # save best ebd
            for k in 'ebd', 'clf':
                best_model[k] = copy.deepcopy(partition_model[k].state_dict())
        else:
            cycle += 1

        if cycle == args.patience:
            break

    for k in 'ebd', 'clf':
        partition_model[k].load_state_dict(best_model[k])

    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Finished training the unstable feature representaiton",
          flush=True)
