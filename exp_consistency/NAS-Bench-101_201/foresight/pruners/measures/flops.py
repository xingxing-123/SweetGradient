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
import torch.nn.functional as F
import numpy as np
from thop import profile

import copy

from . import measure
from ..p_utils import get_flattened_metric, count_parameters


@measure('flops', bn=True)
def get_size(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    macs, params = profile(net, inputs=(inputs[[0]], ), verbose=False)
    score = macs
    return score
