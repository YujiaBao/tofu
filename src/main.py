import os
import copy
import argparse
from datetime import datetime
import random

import torch
import numpy as np

from data import get_dataset, is_textdata
from model import get_model
import tofu


def get_parser():
    parser = argparse.ArgumentParser(description='TOFU: transfer of unstable features')
    parser.add_argument('--cuda', type=int, default=0)

    # data sample
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='max number of epochs to run')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='sample num_batches batches for each epoch')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)

    # model
    parser.add_argument('--hidden_dim', type=int, default=300)

    #dataset
    parser.add_argument('--src_dataset', type=str, default='')
    parser.add_argument('--tar_dataset', type=str, default='')
    parser.add_argument('--dataset', type=str, default='', help='placeholder')

    # method specification
    parser.add_argument('--num_clusters', type=int, default=2)
    parser.add_argument('--transfer_ebd', action='store_true', default=False,
        help='whether to transfer the ebd function learned from the source task')

    #optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--thres', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)

    return parser


def print_args(args):
    '''
        Print arguments (only show the relevant arguments)
    '''
    print(
      '''                                                                            \n'''
      '''TTTTTTTTTTTTTTTTTTTTTTT                 ffffffffffffffff                    \n'''
      '''T:::::::::::::::::::::T                f::::::::::::::::f                   \n'''
      '''T:::::::::::::::::::::T               f::::::::::::::::::f                  \n'''
      '''T:::::TT:::::::TT:::::T               f::::::fffffff:::::f                  \n'''
      '''TTTTTT  T:::::T  TTTTTTooooooooooo    f:::::f       ffffffuuuuuu    uuuuuu  \n'''
      '''        T:::::T      oo:::::::::::oo  f:::::f             u::::u    u::::u  \n'''
      '''        T:::::T     o:::::::::::::::of:::::::ffffff       u::::u    u::::u  \n'''
      '''        T:::::T     o:::::ooooo:::::of::::::::::::f       u::::u    u::::u  \n'''
      '''        T:::::T     o::::o     o::::of::::::::::::f       u::::u    u::::u  \n'''
      '''        T:::::T     o::::o     o::::of:::::::ffffff       u::::u    u::::u  \n'''
      '''        T:::::T     o::::o     o::::o f:::::f             u::::u    u::::u  \n'''
      '''        T:::::T     o::::o     o::::o f:::::f             u:::::uuuu:::::u  \n'''
      '''      TT:::::::TT   o:::::ooooo:::::of:::::::f            u:::::::::::::::uu\n'''
      '''      T:::::::::T   o:::::::::::::::of:::::::f             u:::::::::::::::u\n'''
      '''      T:::::::::T    oo:::::::::::oo f:::::::f              uu::::::::uu:::u\n'''
      '''      TTTTTTTTTTT      ooooooooooo   fffffffff                uuuuuuuu  uuuu\n'''
      '''\n'''
      '''In memory of White Nebula Cotton Candy Tofu Cutie CGC TKI                 \n'''
    )

    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("  {}={}".format(attr.upper(), value))


def set_seed(seed):
    '''
        Setting random seeds
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    set_seed(args.seed)

    print_args(args)

    #################################################################
    #
    # Step 1: Identify spurious correlations from the source tasks
    #
    #################################################################

    best_model_ebd = None # store the feature extractor learned on the source task

    train_partition_loaders = []
    val_partition_loaders = []

    for source_task in args.src_dataset.split(','):
        # set the current dataset to source task
        args.dataset = source_task
        print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
              f' Loading source task {args.dataset}',
              flush=True)

        src_data = get_dataset(args.dataset, is_target=False)

        # initialize model and optimizer based on the dataset
        model, opt = get_model(args, src_data)

        # load encoder from the previous source tasks (if exists)
        if best_model_ebd is not None:
            model['ebd'].load_state_dict(best_model_ebd)

        # start training

        # contrast different source environments
        # save the partition results on the source task for learning the
        # unstable feature space
        cur_train_partition_loaders, cur_val_partition_loaders = \
            tofu.contrast_source_envs(src_data, model, opt, args)

        train_partition_loaders.extend(cur_train_partition_loaders)
        val_partition_loaders.extend(cur_val_partition_loaders)

        # transfer the source robust model's feature extractor
        best_model_ebd = copy.deepcopy(model['ebd'].state_dict())

    #################################################################
    #
    # Step 2: Learn an unstable feature representation
    #
    #################################################################

    partition_model, opt = get_model(args, src_data)
    tofu.train_partition(train_partition_loaders,
                         val_partition_loaders,
                         partition_model,
                         opt, args)

    #################################################################
    #
    # Step 3: Transfer the unstable feature to the target task
    #
    #################################################################

    # set the current dataset to the target task
    args.dataset = args.tar_dataset
    print()
    print(datetime.now().strftime('%02y/%02m/%02d %H:%M:%S') +
          f' Loading target task {args.dataset}',
          flush=True)

    if args.transfer_ebd and is_textdata(args.dataset):
        tar_data = get_dataset(
            args.dataset, is_target=True, vocab=src_data.vocab)
    else:
        tar_data = get_dataset(args.dataset, is_target=True)

    # initialize model and optimizer based on the dataset
    model, opt = get_model(args, tar_data)
    if args.transfer_ebd:
        model['ebd'].load_state_dict(best_model_ebd)

    tofu.train_target_model(
        tar_data, model, partition_model, opt, args
    )

    # evaluate the robust performance on the test data
    tofu.evaluate_target_model(tar_data, model, args)
