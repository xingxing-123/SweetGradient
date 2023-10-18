import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from tqdm import tqdm
from prettytable import PrettyTable

d = 'logs/Consistency-NB-101'
runs = []
processed = {}

file_list = None

if file_list is None:
    file_list = os.listdir(d)

for f in tqdm(file_list):
    pf = open(os.path.join(d,f),'rb')
    print(os.path.join(d,f))
    while 1:
        try:
            p = pickle.load(pf)
            if p['hash'] in processed:
                idx = processed[p['hash']]
                runs[idx]['logmeasures'] = {**runs[idx]['logmeasures'], **p['logmeasures']}
                continue
            processed[p['hash']] = len(runs)
            runs.append(p)
        except:
            break
    pf.close()
with open('data/NAS-Bench-Data/nasbench1_accuracy.p','rb') as f:
    all_accur = pickle.load(f)


t = None

print(d, len(runs))
metrics = {}
for k in runs[0]['logmeasures'].keys():
    metrics[k] = []
acc = []
hashes = []

if t is None:
    hl = ['Dataset']
    # hl.extend(['params', 'effective_capacity', 'grad_norm', 'effective_gradnorm', 'snip', 'effective_snip', 'grasp', 'effective_grasp', 'ntktrace', 'effective_ntktrace'])
    # hl.extend(['ntktrace', 'zico', 'effective_zico', 'grad_conflict', 'effective_gradsign', 'effective_gradnorm', 'effective_snip', 'effective_ntktrace', 'effective_capacity'])
    hl.extend(['effective_capacity'])
    t = PrettyTable(hl)

for r in runs:
    for k, v in r['logmeasures'].items():
        metrics[k].append(v)

    acc.append(all_accur[r['hash']][0])
    hashes.append(r['hash'])

res = []
for k in hl:
    if k == 'Dataset':
        continue
    v = metrics[k]
    cr = abs(stats.spearmanr(acc, v, nan_policy='omit').correlation)
    # cr = abs(stats.kendalltau(acc, v, nan_policy='omit').correlation)
    # print(f'{k} = {cr}')
    res.append(round(cr, 3))

ds = 'CIFAR10'
t.add_row([ds] + res)

print(t)