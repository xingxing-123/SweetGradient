import os, sys
import torch
from torch import nn
import numpy as np
from . import measure
from ..p_utils import get_layer_metric_array

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output = model.forward_pre_GAP(input)
            mixup_output = model.forward_pre_GAP(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)
            nas_score = torch.where(torch.isnan(nas_score), torch.full_like(nas_score, 0), nas_score)
            nas_score = torch.where(torch.isinf(nas_score), torch.full_like(nas_score, 1), nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var).clamp(min=1e-10))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor.clamp(min=1e-10))
                pass
            pass
            
            nas_score = torch.log(nas_score.clamp(min=1e-10)) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))


    # std_nas_score = np.std(nas_score_list)
    # avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    # info['std_nas_score'] = float(std_nas_score)
    # info['avg_precision'] = float(avg_precision)
    return info


@measure('zen_score', bn=False, mode='param')
def compute_Zen_score(net, inputs, targets, mode, split_data=1, loss_fn=None):
    info = compute_nas_score(gpu=0, model=net, mixup_gamma=1e-2,
                             resolution=inputs.size(3), batch_size=inputs.size(0), repeat=1, fp16=False)
    zen_score = info['avg_nas_score']

    return zen_score
