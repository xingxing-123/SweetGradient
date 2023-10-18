'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/VITA-Group/TENAS
'''

import os, sys

import torch
from torch import nn
from . import measure
from ..p_utils import get_layer_metric_array

class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples, gpu=None):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None
        self.gpu = gpu


    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron)
            if self.gpu is not None:
                self.activations = self.activations.cuda(self.gpu)
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half())
        res += res.T
        res = 1 - torch.sign(res)
        res = res.sum(1)
        res = 1. / res.float()
        self.n_LR = res.sum().item()
        del self.activations, res
        self.activations = None
        if self.gpu is not None:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR

class Linear_Region_Collector:
    def __init__(self, models=[], input_size=(64, 3, 32, 32), gpu=None,
                 sample_batch=1, dataset=None, data_path=None, seed=0):
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        # self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.dataset = dataset
        self.data_path = data_path
        self.seed = seed
        self.gpu = gpu
        self.device = torch.device('cuda:{}'.format(self.gpu)) if self.gpu is not None else torch.device('cpu')
        # print('Using device:{}'.format(self.device))

        self.reinit(models, input_size, sample_batch, seed)


    def reinit(self, models=None, input_size=None, sample_batch=None, seed=None):
        if models is not None:
            assert isinstance(models, list)
            del self.models
            self.models = models
            for model in self.models:
                self.register_hook(model)
            self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch, gpu=self.gpu) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                # self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch
            # if self.data_path is not None:
            #     self.train_data, _, class_num = get_datasets(self.dataset, self.data_path, self.input_size, -1)
            #     self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.input_size[0], num_workers=16, pin_memory=True, drop_last=True, shuffle=True)
            #     self.loader = iter(self.train_loader)
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            if self.gpu is not None:
                torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        if self.gpu is not None:
            torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        if self.gpu is not None:
            torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            # try:
            #     inputs, targets = self.loader.next()
            # except Exception:
            #     del self.loader
            #     self.loader = iter(self.train_loader)
            #     inputs, targets = self.loader.next()
            inputs = torch.randn(self.input_size, device=self.device)

            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            # model.forward(input_data.cuda())
            model.forward(input_data)
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)


def compute_RN_score(model: nn.Module,  batch_size=None, image_size=None, num_batch=None, gpu=None):
    # # just debug
    # gpu = 0
    # import ModelLoader
    # model = ModelLoader._get_model_(arch='resnet18', num_classes=1000, pretrained=False, opt=None, argv=None)
    #
    # if gpu is not None:
    #     model = model.cuda(gpu)
    lrc_model = Linear_Region_Collector(models=[model], input_size=(batch_size, 3, image_size, image_size),
                                        gpu=gpu, sample_batch=num_batch)
    num_linear_regions = float(lrc_model.forward_batch_sample()[0])
    del lrc_model
    torch.cuda.empty_cache()
    return num_linear_regions



import numpy as np
import torch


def get_ntk_n(networks, inputs, train_mode=False, num_batch=None, gpu=None):
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]

    # for i, (inputs, targets) in enumerate(xloader):
    #     if num_batch > 0 and i >= num_batch: break
    for i in range(num_batch):
        inputs = inputs.to(device)
        # inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            if gpu is not None:
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            else:
                inputs_ = inputs.clone()

            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                if gpu is not None:
                    torch.cuda.empty_cache()

    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        # conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True))
    return conds



def compute_NTK_score(gpu, model, inputs):
    ntk_score = get_ntk_n([model], inputs, train_mode=True, num_batch=1, gpu=gpu)[0]
    return -1 * ntk_score


@measure('te_score', bn=False, mode='param')
def compute_TE_score(net, inputs, targets, mode, split_data=1, loss_fn=None):
    ntk = compute_NTK_score(gpu=0, model=net, inputs=inputs)
    RN = compute_RN_score(model=net, batch_size=inputs.size(0), image_size=inputs.size(3),
                              num_batch=1, gpu=0)
    score = RN + ntk

    return score