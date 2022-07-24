'''
    Learning a robust target classifier.
'''
import copy
from datetime import datetime
from collections import Counter

import torch
import numpy as np
from sklearn import metrics
from sklearn.cluster import k_means
from torch.utils.data import DataLoader
from termcolor import colored

from data import EnvSampler
from tofu.utils import train_dro_loop, test_loop, to_cuda, squeeze_batch, print_res


def cluster_loop(test_loader, model, args):
    '''
        Given a test loader,
        1. Convert all input examples into the unstable feature representation
        space.
        2. Cluster the feature representation
    '''
    model['ebd'].eval()
    groups = {}

    for batch in test_loader:
        batch = to_cuda(squeeze_batch(batch))

        x_s = model['ebd'](batch['X']).cpu().numpy()
        y_s = batch['Y'].cpu().numpy()
        c_s = batch['C'].cpu().numpy()
        idx_s = batch['idx'].cpu().numpy()

        for x, y, c, idx in zip(x_s, y_s, c_s, idx_s):
            if int(y) not in groups:
                groups[int(y)] = {
                    'x': [],
                    'c': [],
                    'idx': [],
                }
            groups[int(y)]['x'].append(x)
            groups[int(y)]['c'].append(c)
            groups[int(y)]['idx'].append(idx)

    clusters = []
    clustering_metrics = [metrics.homogeneity_score,
                          metrics.completeness_score,
                          metrics.v_measure_score,
                          metrics.adjusted_rand_score,
                          metrics.adjusted_mutual_info_score]
    clustering_results = []

    for k, v in groups.items():
        print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
              f' Generate {args.num_clusters} clusters on examples with label value {k}')

        x = np.stack(v['x'], axis=0)

        cur_clusters = {}
        cur_cs = {}

        centroid, label, inertia = k_means(x, args.num_clusters)

        metric_c = np.stack(v['c'], axis=0)
        clustering_results.append([m(metric_c, label) for m in
                                   clustering_metrics])

        for cluster_id, idx, c in zip(label, v['idx'], v['c']):
            if cluster_id not in cur_clusters:
                cur_clusters[cluster_id] = []
                cur_cs[cluster_id] = []

            cur_clusters[cluster_id].append(idx)
            cur_cs[cluster_id].append(c)

        for cluster_id, cluster in cur_clusters.items():
            clusters.append(cluster)
            cnt = Counter(cur_cs[cluster_id])

            print(f'cluster {cluster_id} '
                  f'{colored("total_size", "yellow")} '
                  f'{len(cur_cs[cluster_id]):>6d}',
                  end=' (')

            for c, cur_cnt in sorted(cnt.items()):
                print(f'{int(100*cur_cnt/len(cur_cs[cluster_id])):>3}% '
                      f'has unstable_feat={c} ',
                      end='')
            print(')')

    clustering_results = np.mean(np.array(clustering_results), axis=0)
    print(f'Clustering metrics '
          f'{colored("homogeneity", "blue")} '
          f'{float(clustering_results[0]):.4f} '
          f'{colored("completeness", "blue")} '
          f'{float(clustering_results[1]):.4f} '
          f'{colored("v-score", "blue")} '
          f'{float(clustering_results[2]):.4f}'
          )

    return clusters


def train_target_model(data, model, partition_model, opt, args):
    '''
        Train a robust target model.
        1. Use the partition model to cluster the target data into different
        groups
        2. Apply group-DRO to learn a robust target model
    '''
    # env id
    train_env = 0
    val_env   = 2

    #########################################################
    #
    # 1. partition the target data into different groups

    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          ' Partition the train data for the target task.', flush=True)

    train_loader = DataLoader(
        data, sampler=EnvSampler(
            -1, args.batch_size, train_env, data.envs[train_env]['idx_list']),
        num_workers=10)

    with torch.no_grad():
        partition_res = cluster_loop(train_loader, partition_model, args)

    del train_loader

    # use the partition results to create training groups
    train_loaders = []
    for group in partition_res:
        train_loaders.append(DataLoader(
            data, sampler=EnvSampler(args.num_batches, args.batch_size,
                                     train_env, group),
            num_workers=int(10 / args.num_clusters * 2)))

    # use partition model to cluster the valdiation input and create the
    # validation groups
    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          ' Partition the validation data for the target task.', flush=True)

    val_loaders = []
    val_loader = DataLoader(
        data,
        sampler=EnvSampler(-1, args.batch_size, val_env,
                           data.envs[val_env]['idx_list'])
    )

    with torch.no_grad():
        partition_res = cluster_loop(val_loader, partition_model, args)

    for group in partition_res:
        val_loaders.append(DataLoader(
            data, sampler=EnvSampler(args.num_batches, args.batch_size,
                                     val_env, group),
            num_workers=int(10 / args.num_clusters * 2)))
    #########################################################
    #
    #
    #
    #########################################################
    # 2. train a stable classifier by minimizing the worst case risk across all
    # groups
    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f" Use DRO to learn a robust target classifier", flush=True)

    best_acc = -1
    best_val_res = None
    best_model = {}
    cycle = 0
    for ep in range(args.num_epochs):
        train_res = train_dro_loop(train_loaders, model, opt, ep, args)

        with torch.no_grad():
            # validation
            # use the worst-cluster acc in the validation data for early
            # stopping
            val_res = {'acc': [], 'loss': []}
            for val_loader in val_loaders:
                cur_val_res = test_loop(val_loader, model, args)
                val_res['acc'].append(cur_val_res['acc'])
                val_res['loss'].append(cur_val_res['loss'])
            val_res['acc'] = min(val_res['acc'])
            val_res['loss'] = max(val_res['loss'])

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


def evaluate_target_model(data, model, args):
    test_env = 3

    print()
    print("{} Evaluating on the test environment for {}".format(
        datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
        args.dataset), flush=True)

    test_loader = DataLoader(
        data,
        sampler=EnvSampler(-1, args.batch_size, test_env,
                           data.envs[test_env]['idx_list']),
        num_workers=10)

    test_res = test_loop(test_loader, model, args,
                         att_idx_dict=data.test_att_idx_dict)

    if 'avg_acc' not in test_res:
        print('Test results: '
              f'{colored("acc", "yellow")} {test_res["acc"]:.4f} '
              f'{colored("loss", "blue")} {test_res["loss"]:.4f} ')
    else:
        print('Test results: '
              f'{colored("worst acc", "yellow")} {test_res["acc"]:.4f} '
              f'{colored("avg acc", "yellow")} {test_res["acc"]:.4f} '
              f'{colored("loss", "blue")} {test_res["loss"]:.4f} ')
