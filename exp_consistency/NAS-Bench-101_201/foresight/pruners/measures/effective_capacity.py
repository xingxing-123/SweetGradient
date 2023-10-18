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
@measure('effective_capacity', bn=True)
def compute_effective_capacity(net, inputs, targets, loss_fn, THR1, THR2):
    global flag
    if flag:
        print(f"!NOTE: THR1: {THR1}, THR2: {THR2}")
        flag = False

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

    result = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.weight.grad is not None:
                p = m.weight
                v = (p.grad.data).abs()
                result += ((v >= THR1).sum() - (v >= THR2).sum())
    score = result.item()

    return score
