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

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nasbench2_ops import *
from .nasbench2_utils import topology_str2structure


def gen_searchcell_mask_from_arch_str(arch_str):
    nodes = arch_str.split('+') 
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')  for op_and_input in node] for node in nodes]

    keep_mask = []
    for curr_node_idx in range(len(nodes)):
            for prev_node_idx in range(curr_node_idx+1): 
                _op = [edge[0] for edge in nodes[curr_node_idx] if int(edge[1]) == prev_node_idx]
                assert len(_op) == 1, 'The arch string does not follow the assumption of 1 connection between two nodes.'
                for _op_name in OPS.keys():
                    keep_mask.append(_op[0] == _op_name)
    return keep_mask


def get_model_from_arch_str(arch_str, num_classes, use_bn=True, init_channels=16):
    # keep_mask = gen_searchcell_mask_from_arch_str(arch_str)
    # net = NAS201Model(arch_str=arch_str, num_classes=num_classes, use_bn=use_bn, keep_mask=keep_mask, stem_ch=init_channels)
    genotype = topology_str2structure(arch_str).nodes
    net = NAS201Model(C=init_channels, N=5, genotype=genotype, num_classes=num_classes)
    net.arch_str = arch_str
    return net



class NAS201Model(nn.Module):
    def __init__(self, C, N, genotype, num_classes):
        super(NAS201Model, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
    
    def forward_pre_GAP(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        out = self.lastact(feature)

        return out


# class NAS201Model(nn.Module):

#     def __init__(self, arch_str, num_classes, use_bn=True, keep_mask=None, stem_ch=16):
#         super(NAS201Model, self).__init__()
#         self.arch_str=arch_str
#         self.num_classes=num_classes
#         self.use_bn= use_bn

#         self.stem = stem(out_channels=stem_ch, use_bn=use_bn)
#         self.stack_cell1 = nn.Sequential(*[SearchCell(in_channels=stem_ch, out_channels=stem_ch, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
#         self.reduction1 = reduction(in_channels=stem_ch, out_channels=stem_ch*2)
#         self.stack_cell2 = nn.Sequential(*[SearchCell(in_channels=stem_ch*2, out_channels=stem_ch*2, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
#         self.reduction2 = reduction(in_channels=stem_ch*2, out_channels=stem_ch*4)
#         self.stack_cell3 = nn.Sequential(*[SearchCell(in_channels=stem_ch*4, out_channels=stem_ch*4, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
#         self.top = top(in_dims=stem_ch*4, num_classes=num_classes, use_bn=use_bn)

#     def forward(self, x):
#         x = self.stem(x)        

#         x = self.stack_cell1(x)
#         x = self.reduction1(x)

#         x = self.stack_cell2(x)
#         x = self.reduction2(x)

#         x = self.stack_cell3(x)

#         x = self.top(x)
#         return x
    
#     def get_prunable_copy(self, bn=False):
#         model_new = get_model_from_arch_str(self.arch_str, self.num_classes, use_bn=bn)

#         #TODO this is quite brittle and doesn't work with nn.Sequential when bn is different
#         # it is only required to maintain initialization -- maybe init after get_punable_copy?
#         model_new.load_state_dict(self.state_dict(), strict=False)
#         model_new.train()

#         return model_new
    

# def get_arch_str_from_model(net):
#     search_cell = net.stack_cell1[0].options
#     keep_mask = net.stack_cell1[0].keep_mask
#     num_nodes = net.stack_cell1[0].num_nodes

#     nodes = []
#     idx = 0
#     for curr_node in range(num_nodes -1):
#         edges = []
#         for prev_node in range(curr_node+1): # n-1 prev nodes
#             for _op_name in OPS.keys():
#                 if keep_mask[idx]:
#                     edges.append(f'{_op_name}~{prev_node}')
#                 idx += 1
#         node_str = '|'.join(edges)
#         node_str = f'|{node_str}|'
#         nodes.append(node_str) 
#     arch_str = '+'.join(nodes)
#     return arch_str

def get_arch_str_from_model(net):
    return net.arch_str


if __name__ == "__main__":
    arch_str = '|nor_conv_3x3~0|+|none~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    
    n = get_model_from_arch_str(arch_str=arch_str, num_classes=10)
    # print(n.stack_cell1[0])
    
    arch_str2 = get_arch_str_from_model(n)
    print(arch_str)
    print(arch_str2)
    print(f'Are the two arch strings same? {arch_str == arch_str2}')

    inputs = torch.randn(2, 3, 32,32)
    n.zero_grad()
    ouput = n(inputs)
    ouput.sum().backward()
    with torch.no_grad():
        for name, layer in n.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                p = layer.weight
                if p.grad is None:
                    print(name, layer.__class__.__name__)
                else:
                    print(p.grad.data.mean(), name, layer.__class__.__name__)
    
