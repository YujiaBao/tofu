import torch
import torch.nn as nn

from model.embedding.cnn import CNN
from model.embedding.textcnn import TextCNN
from model.classifier.mlp import MLP
from model.embedding.resnet import Resnet50
from data.utils import is_textdata


def get_model(args, data=None):
    '''
        All model has two keys: ebd and clf
        ebd is the feature extractor (different for different datasets)
        clf uses a MLP for the final classification
    '''
    model = {}
    if args.dataset[:5] == 'MNIST':
        if args.dataset == 'MNIST':
            model['ebd'] = CNN(include_fc=True, hidden_dim=args.hidden_dim).cuda()
        else:
            _, _, _, max_c = args.dataset.split('_')
            model['ebd'] = CNN(include_fc=True,
                               hidden_dim=args.hidden_dim,
                               input_channels=int(max_c)).cuda()
        out_dim=args.hidden_dim
        num_classes = 10

    if args.dataset[:4] == 'bird':
        model['ebd'] = Resnet50().cuda()
        out_dim=model['ebd'].out_dim
        num_classes = 2

    if args.dataset[:6] == 'celeba':
        model['ebd'] = Resnet50().cuda()
        out_dim=model['ebd'].out_dim
        num_classes = 2

    if is_textdata(args.dataset):
        model['ebd'] = TextCNN(data.vocab, num_filters=args.hidden_dim,
                               dropout=args.dropout).cuda()
        out_dim = args.hidden_dim * 3  # 3 different filters
        num_classes = 2

    model['clf'] = MLP(out_dim, args.hidden_dim, num_classes, args.dropout,
                           depth=1).cuda()

    opt = torch.optim.Adam(
        list(model['ebd'].parameters()) + list(model['clf'].parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    return model, opt
