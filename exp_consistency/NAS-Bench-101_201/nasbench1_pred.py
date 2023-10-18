# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pickle
import torch
import argparse
import json
import numpy as np
from thop import profile
import logging

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from utils import *

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-101')
    parser.add_argument('--api_loc', default='data/NAS-Bench-Data/nasbench_only108.tfrecord',
                        type=str, help='path to API')
    parser.add_argument('--json_loc', default='data/NAS-Bench-Data/all_graphs.json',
                        type=str, help='path to JSON database')
    parser.add_argument('--outdir', default='./101_results',
                        type=str, help='output directory')
    parser.add_argument('--outfname', default='test',
                        type=str, help='output filename')
    parser.add_argument('--measure_names', default="all",
                        type=str, help='measure_names, like "synflow params"')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--datadir', type=str, default='dataset/cifar10', help='dataset dir')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=423624, help='end index')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--THR1', default=0, type=float)
    parser.add_argument('--THR2', default=5, type=float)
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args

def get_op_names(v):
    o = []
    for op in v:
        if op == -1:
            o.append('input')
        elif op == -2:
            o.append('output')
        elif op == 0:
            o.append('conv3x3-bn-relu')
        elif op == 1:
            o.append('conv1x1-bn-relu')
        elif op == 2:
            o.append('maxpool3x3')
    return o
    
if __name__ == '__main__':
    init_log()
    args = parse_arguments()
    logging.info(args)
    set_seed(args.seed)

    #nasbench = api.NASBench(args.api_loc)
    models = json.load(open(args.json_loc))

    logging.info(f'Running models {args.start} to {args.end} out of {len(models.keys())}')

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers, datadir=args.datadir)

    for _, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        break

    all_points = []
    pre='cf' if 'cifar' in args.dataset else 'im'

    if args.outfname == 'test':
        fn = f'nb1_{pre}{get_num_classes(args)}.p'
    else:
        fn = f'{args.outfname}.p'
    op = os.path.join(args.outdir,fn)

    logging.info(f'outfile ={op}')
    first = True

    #loop over nasbench1 archs (k=hash, v=[adj_matrix, ops])
    idx = 0
    cached_res = []
    for k,v in models.items():

        if idx < args.start:
            idx += 1
            continue
        if idx >= args.end:
            break 
        logging.info(f'idx = {idx}')
        idx += 1

        res = {}
        res['hash']=k

        # model
        spec = nasbench1_spec._ToModelSpec(v[0], get_op_names(v[1]))
        net = nasbench1.Network(spec, stem_out=128, num_stacks=3, num_mods=3, num_classes=get_num_classes(args))
        net.to(args.device)

        measures = predictive.find_measures(net, 
                                            train_loader,
                                            (args.dataload, args.dataload_info, get_num_classes(args)),
                                            args.device,
                                            pre_data_pair=None, # (x, y),
                                            measure_names=None if args.measure_names == "all" else args.measure_names.split('+'),
                                            THR1=args.THR1,
                                            THR2=args.THR2)
        res['logmeasures']= measures

        logging.info(res)
        cached_res.append(res)

        #write to file
        if idx % args.write_freq == 0:
            logging.info(f'writing {len(cached_res)} results to {op}')
            pf=open(op, 'ab')
            for cr in cached_res:
                pickle.dump(cr, pf)
            pf.close()
            cached_res = []
    