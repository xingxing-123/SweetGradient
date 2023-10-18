import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from tqdm import tqdm
from prettytable import PrettyTable

d = 'logs/Consistency-NB-201'
runs = []
accs = []
processed = {}

dataset='cifar10' # cifar10 cifar100 ImageNet16-120

file_list = None
# [
#     'cifar10_0_15625_measure[effective_capacity].p',
# ]
if file_list is None:
    file_list = os.listdir(d)

for f in tqdm(file_list):
    if dataset + '_' not in f or '.p' not in f:
        continue
    else:
        pass
        # print(f)
    pf = open(os.path.join(d,f),'rb')
    # print(os.path.join(d,f))
    while 1:
        try:
            p = pickle.load(pf)
            if p['i'] in processed:
                idx = processed[p['i']]
                runs[idx]['logmeasures'] = {**runs[idx]['logmeasures'], **p['logmeasures']}
                continue
            processed[p['i']] = len(runs)
            runs.append(p)
            accs.append(p['testacc'])
        except:
            break
    pf.close()

t = None

print(d, len(runs))
metrics = {}
for k in runs[0]['logmeasures'].keys():
    metrics[k] = []
acc = accs

if t is None:
    hl = ['Dataset']
    # hl.extend(['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacob_cov', 'te_score', 'zen_score', 'grad_angle', 'grad_conflict', 'flops', 'params', 'effective_capacity'])
    # hl.extend(['params', 'effective_capacity', 'grad_norm', 'effective_gradnorm', 'snip', 'effective_snip', 'grasp', 'effective_grasp', 'ntktrace', 'effective_ntktrace'])
    # hl.extend(['ntktrace', 'zico', 'effective_zico', 'grad_conflict', 'effective_gradsign', 'effective_gradnorm', 'effective_snip', 'effective_ntktrace', 'effective_capacity'])
    # hl.extend(['te_score'])
    # hl.extend(['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacob_cov', 'te_score', 'zen_score'])
    hl.extend(['effective_capacity'])
    t = PrettyTable(hl)

for r in runs:
    for k, v in r['logmeasures'].items():
        metrics[k].append(v)

print(hl)

res = []
for k in hl:
    if k == 'Dataset':
        continue
    v = metrics[k]
    cr = stats.spearmanr(acc, v, nan_policy='omit').correlation
    # cr = stats.kendalltau(acc, v, nan_policy='omit').correlation
    # print(f'{k} = {cr}')
    res.append(round(cr, 3))

ds = dataset # 'CIFAR10' CIFAR100, ImageNet16-120
t.add_row([ds] + res)

print(t)