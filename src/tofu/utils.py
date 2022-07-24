import torch
import torch.nn.functional as F
from termcolor import colored

from data import is_textdata


def to_cuda(d):
    '''
        convert the input dict to cuda
    '''
    for k, v in d.items():
        d[k] = v.cuda()

    return d


def squeeze_batch(batch):
    '''
        squeeze the first dim in a batch
    '''
    res = {}
    for k, v in batch.items():
        assert len(v) == 1
        res[k] = v[0]

    return res


def print_res(train_res, val_res, ep):
    print(f'epoch {ep:>3} '
          f'train {colored("avg", "blue")} {train_res["avg_acc"]:>6.4f} '
          f'{colored("worst", "blue")} {train_res["worst_acc"]:>6.4f} '
          f'{colored("loss", "yellow")} {train_res["avg_loss"]:>6.4f} '
          f'{train_res["worst_loss"]:>6.4f} '
          f'val {colored("acc", "blue")} {val_res["acc"]:>6.4f} '
          f'{colored("loss", "yellow")} {val_res["loss"]:>6.4f}',
          flush=True)


def train_dro_loop(train_loaders, model, opt, ep, args):
    stats = {}
    for k in ['worst_loss', 'avg_loss', 'worst_acc', 'avg_acc']:
        stats[k] = []

    step = 0
    for batches in zip(*train_loaders):
        # work on each batch
        model['ebd'].train()
        model['clf'].train()

        x, y = [], []

        for batch in batches:
            batch = to_cuda(squeeze_batch(batch))
            x.append(batch['X'])
            y.append(batch['Y'])

        if is_textdata(args.dataset):
            # text models have varying length between batches
            pred = []
            for cur_x in x:
                pred.append(model['clf'](model['ebd'](cur_x)))
            pred = torch.cat(pred, dim=0)
        else:
            pred = model['clf'](model['ebd'](torch.cat(x, dim=0)))

        cur_idx = 0

        avg_loss = 0
        avg_acc = 0
        worst_loss = 0
        worst_acc = 0

        for cur_true in y:
            cur_pred = pred[cur_idx:cur_idx+len(cur_true)]
            cur_idx += len(cur_true)

            loss = F.cross_entropy(cur_pred, cur_true)
            acc = torch.mean((torch.argmax(cur_pred, dim=1) == cur_true).float()).item()

            avg_loss += loss.item()
            avg_acc += acc

            if loss.item() > worst_loss:
                worst_loss = loss
                worst_acc = acc

        opt.zero_grad()
        worst_loss.backward()
        opt.step()

        avg_loss /= len(y)
        avg_acc /= len(y)

        stats['avg_acc'].append(avg_acc)
        stats['avg_loss'].append(avg_loss)
        stats['worst_acc'].append(worst_acc)
        stats['worst_loss'].append(worst_loss.item())

    for k, v in stats.items():
        stats[k] = torch.mean(torch.tensor(v).float()).item()

    return stats


def get_worst_acc(true, pred, idx, loss, att_idx_dict):
    acc_list = (true == pred).float().tolist()
    idx = torch.cat(idx).tolist()
    idx_origin2new = dict(zip(idx, range(len(idx))))

    verbose = True
    if len(att_idx_dict) == 1:  # validation
        verbose = False

    worst_acc_list = []
    avg_acc_list = []

    for att, data_dict in att_idx_dict.items():
        if verbose:
            print('{:>20}'.format(att), end=' ')

        worst_acc = 1
        avg_acc = []
        for k, v in data_dict.items():
            # value to index mapping
            acc = []
            for origin in v:
                acc.append(acc_list[idx_origin2new[origin]])

            if len(acc) > 0:
                cur_acc = torch.mean(torch.tensor(acc)).item()

                if verbose:
                    print(k, ' {:>5}'.format(len(v)),
                          ' {:>7.4f}'.format(cur_acc), end=', ')
                if cur_acc < worst_acc:
                    worst_acc = cur_acc
                avg_acc.append(cur_acc)
            else:
                if verbose:
                    print(k, f' {len(v):>5}         ', end=', ')

        if verbose:
            print(' worst: {:>7.4f}'.format(worst_acc))

        avg_acc = torch.mean(torch.tensor(avg_acc)).item()
        worst_acc_list.append(worst_acc)
        avg_acc_list.append(avg_acc)

    return {
        'acc': torch.mean(torch.tensor(worst_acc_list)).item(),
        'avg_acc': torch.mean(torch.tensor(avg_acc_list)).item(),
        'loss': loss,
    }


def test_loop(test_loader, model, args, return_idx=False, att_idx_dict=None):
    loss_list = []
    true, pred, cor = [], [], []
    if (att_idx_dict is not None) or return_idx:
        idx = []

    for batch in test_loader:
        # work on each batch
        model['ebd'].eval()
        model['clf'].eval()

        batch = to_cuda(squeeze_batch(batch))

        x = model['ebd'](batch['X'])
        y = batch['Y']
        c = batch['C']

        y_hat, loss = model['clf'](x, y, return_pred=True)

        true.append(y)
        pred.append(y_hat)
        cor.append(c)

        if (att_idx_dict is not None) or return_idx:
            idx.append(batch['idx'])

        loss_list.append(loss.item())

    true = torch.cat(true)
    pred = torch.cat(pred)

    acc = torch.mean((true == pred).float()).item()
    loss = torch.mean(torch.tensor(loss_list).float()).item()

    if return_idx:
        cor = torch.cat(cor).tolist()
        true = true.tolist()
        pred = pred.tolist()
        idx = torch.cat(idx).tolist()

        # split correct and wrong idx
        correct_idx, wrong_idx = [], []

        # compute correlation between cor and y for analysis
        correct_cor, wrong_cor = [], []
        correct_y, wrong_y = [], []

        for i, y, y_hat, c in zip(idx, true, pred, cor):
            if y == y_hat:
                correct_idx.append(i)
                correct_cor.append(c)
                correct_y.append(y)
            else:
                wrong_idx.append(i)
                wrong_cor.append(c)
                wrong_y.append(y)

        return {
            'acc': acc,
            'loss': loss,
            'correct_idx': correct_idx,
            'correct_cor': correct_cor,
            'correct_y': correct_y,
            'wrong_idx': wrong_idx,
            'wrong_cor': wrong_cor,
            'wrong_y': wrong_y,
        }

    if att_idx_dict is not None:
        return get_worst_acc(true, pred, idx, loss, att_idx_dict)

    return {
        'acc': acc,
        'loss': loss,
    }
