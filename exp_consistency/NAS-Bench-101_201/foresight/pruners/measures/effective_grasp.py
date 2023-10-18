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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from . import measure
from ..p_utils import get_layer_metric_array

flag = True
@measure('effective_grasp', bn=True, mode='param')
def compute_grasp_per_weight(net, inputs, targets, mode, loss_fn, THR1, THR2, T=1, num_iters=1, split_data=1):
    global flag
    if flag:
        print(f"!NOTE: THR1: {THR1}, THR2: {THR2}")
        flag = False

    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True) # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        #forward/grad pass #1
        grad_w = None
        for _ in range(num_iters):
            #TODO get new data, otherwise num_iters is useless!
            outputs = net.forward(inputs[st:en])/T
            loss = loss_fn(outputs, targets[st:en])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        # forward/grad pass #2
        outputs = net.forward(inputs[st:en])/T
        loss = loss_fn(outputs, targets[st:en])
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
        
        # accumulate gradients computed in previous step and call backwards
        z, count = 0,0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    with torch.no_grad():
                        grad_w[count].data *= (grad_w[count] >= THR1) * (grad_w[count] < THR2)
                        # grad_f[count].data *= (grad_f[count] >= THR1) * (grad_f[count] < THR2)
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # result = 0
    # count = 0
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         if m.weight.grad is not None and grad_w[count] is not None:
    #             v = (grad_w[count].data).abs()
    #             hg = -m.weight.data * m.weight.grad
    #             result += (((v >= THR1) * hg).sum() - ((v >= THR2) * hg).sum())
    #         count += 1
    # score = result.item()

    result = 0
    count = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.weight.grad is not None:
                hg = -m.weight.data * m.weight.grad * (grad_w[count] >= THR1) * (grad_w[count] < THR2)
                result += hg.abs().sum()
            count += 1
    score = result.item()

    return score