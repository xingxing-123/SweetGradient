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

@measure('ntktrace', bn=True)
def get_ntktrace(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    result = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.weight.grad is not None:
                v = m.weight.grad ** 2
                result += v.sum()
    score = result.item()

    return score
