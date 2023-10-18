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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import measure
from ..p_utils import get_layer_metric_array

flag = True
@measure('effective_gradnorm', bn=True)
def compute_effective_gradnorm(net, inputs, targets, loss_fn, THR1, THR2):
    global flag
    if flag:
        print(f"!NOTE: THR1: {THR1}, THR2: {THR2}")
        flag = False
    # def forward_hook(module, input, output):
    #     module.input_mean = (input[0] * input[0]).abs().mean().item()
    #     module.input_std = input[0].abs().std().item()
    
    # # register hook
    # hooks = []
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         hooks.append(m.register_forward_hook(forward_hook))

    net.zero_grad()

    # without data and label
    # _, logits = model(torch.ones(1, x.size(1), x.size(2), x.size(3)).cuda())
    # output = logits.sum()
    # output.backward()

    # with data, without label
    # outputs = net.forward(inputs)
    # output = outputs.sum()
    # output.backward()

    # with data, with label
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    # for h in hooks:
    #     h.remove()

    # 101 THR = 5e-5 THR2=0.0001
    # 201 THR = 0.0005, 0.001
    result = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.weight.grad is not None:
                # THR = 0.0005
                # THR2 = 0.001
                p = m.weight
                v = (p.grad.data).abs()
                result += (((v >= THR1) * v).sum() - ((v >= THR2) * v).sum())
    score = result.item()

    return score
