
import pickle
from matplotlib import style
import torch
import torch.nn as nn
import argparse
import math
import json
from scipy.stats import ttest_ind, pearsonr, spearmanr, kendalltau
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net
from scipy import stats
from utils import *

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--save_dir', type=str, default='./', help='dataset dir')
    parser.add_argument('--api_loc', default='data/NAS-Bench-Data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--datadir', type=str, default='dataset/cifar10', help='dataset dir')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=100, help='end index')
    parser.add_argument('--init_channels', default=16, type=int)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args

def compute_effective_capacity2(net, inputs, targets, loss_fn, THR, THR2):
    net.zero_grad()

    outputs = net(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    result = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            p = m.weight
            if p.grad is not None:
                v = (p.grad).abs()
                result += ((v >= THR).sum() - (v >= THR2).sum())
    
    score = result.item()

    return score


def cal_best_bins(nets, x, y, loss_fn, bins):
    def cal_score(r_list):
        score = 0
        l = len(r_list) - 1
        for i in range(l):
            score += np.sign(r_list[i+1] - r_list[i])
        score /= max(1, l)
        return score

    '''
    all_ratio_info is as follow: 
        net  bin1                               bin2                              ...   bin_n
        1    [d1_ratio d2_ratio ... dL_ratio]   [d1_ratio d2_ratio ... dL_ratio]  ...   [d1_ratio d2_ratio ... dL_ratio]
        2    [d1_ratio d2_ratio ... dL_ratio]   [d1_ratio d2_ratio ... dL_ratio]  ...   [d1_ratio d2_ratio ... dL_ratio]
        ...
        m    [d1_ratio d2_ratio ... dL_ratio]   [d1_ratio d2_ratio ... dL_ratio]  ...   [d1_ratio d2_ratio ... dL_ratio]
    '''
    all_ratio_info = [[[] for _ in bins] for _ in nets]
    for k, net in enumerate(nets):
        logging.info(f"{k}/{len(nets)}")
        net.zero_grad()
        outputs = net(x)
        loss = loss_fn(outputs, y)
        loss.backward()

        for m in net.modules():
            if m.__class__.__name__ in ['Cell', 'InferCell', 'ResNetBasicblock']:
                total_params = 0
                params_in_bins = [0 for _ in bins]
                for cell_m in m.modules():
                    if isinstance(cell_m, nn.Conv2d) or isinstance(cell_m, nn.Linear):
                        p = cell_m.weight
                        if p.grad is not None:
                            v = (p.grad).abs()
                            total_params += v.numel()
                            for i, (THR1, THR2) in enumerate(bins):
                                params_in_bins[i] += ((v >= THR1).sum() - (v >= THR2).sum()).item()
                for i in range(len(bins)):
                    all_ratio_info[k][i].append(params_in_bins[i]/max(1, total_params))

    score_matrix = np.zeros((len(nets), len(bins)))
    for i in range(len(nets)):
        for j in range(len(bins)):
            score_matrix[i][j] = cal_score(all_ratio_info[i][j])
    score_bins = np.mean(score_matrix, 0)
    
    best_idx = 0
    best_value = 0
    for i in range(len(bins)):
        value = score_bins[i]
        if value > best_value:
            best_idx = i
            best_value = value

    return best_idx, score_bins



def plot_bin_score_bar(scores, bins, save_file, xlabel="bins", ylabel="score"):
    plt.figure(figsize=(8,6))

    # bar
    my_cmap = plt.cm.get_cmap('YlGnBu')
    colors = my_cmap((scores - np.min(scores) + 0.05) / (np.max(scores) - np.min(scores)))

    bar_width = 0.98
    fake_x = list(range(len(bins)))
    plt.bar(x=fake_x, height=scores, width=0.98, align='edge', color=colors)

    # line
    fake_x_line = np.array(list(range(len(bins)))) + bar_width / 2
    plt.plot(fake_x_line, scores, color = 'orange', linewidth = 1.0, linestyle="--", marker='*')

    xticks = [bins[0][0]]
    for bin_ in bins:
        xticks.append(bin_[1])

    plt.grid(axis='y')
    plt.xticks(list(range(len(bins) + 1)), xticks, rotation=45)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_file)
    plt.close()

if __name__ == '__main__':
    init_log()
    args = parse_arguments()
    logging.info(args)
    set_seed(args.seed)

    from nas_201_api import NASBench201API as API
    api = API(args.api_loc, verbose=False)

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, datadir=args.datadir)

    for _, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        break
    
    nets = []
    for i, arch_str in enumerate(api):
        if i < args.start:
            continue
        if i >= args.end:
            break 
        # print(arch_str)
        net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args), init_channels=args.init_channels)
        net.to(args.device)
        init_net(net, args.init_w_type, args.init_b_type)
        nets.append(net)


    THR_list = [0, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0, 5e-0]

    bins = []
    for i in range(len(THR_list) - 1):
        bins.append([THR_list[i], THR_list[i+1]])
    
    best_idx, score_list = cal_best_bins(nets, x, y, nn.CrossEntropyLoss().cuda(), bins)

    for i, (bin_, score) in enumerate(zip(bins, score_list)):
        logging.info(f'{i} bin: {bin_}, score: {score_list[i]}')
    logging.info(f'best_bin: {bins[best_idx]}, Score_list: {score_list}')

    plot_bin_score_bar(
        scores=score_list, 
        bins=bins, 
        save_file=os.path.join(args.save_dir, f'ScoreBar_201_{args.dataset}_THR1~THR2.pdf'), 
        xlabel="bins", 
        ylabel="score")







