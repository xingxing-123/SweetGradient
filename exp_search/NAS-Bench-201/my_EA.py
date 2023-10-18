##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################################
# Regularized Evolution for Image Classifier Architecture Search #
##################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
from statistics import mean, stdev

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, get_search_spaces
from xautodl.models.cell_infers import TinyNetwork
from nas_201_api import NASBench201API as API
import torch.nn.functional as F
import types
import torch.autograd as autograd

class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)

class Estimator(object):
    def __init__(self, score, loader, loss_fn, THR1, THR2):
        self.score = score
        self.loader = loader
        self.loss_fn = loss_fn
        self.THR1 = THR1
        self.THR2 = THR2
    
    def get_net(self, genotype):
        return TinyNetwork(16, 5, genotype, 120) # only for cifar10
    
    def get_batch_data(self):
        try:
            inputs, targets = next(self.tmploader)
            return inputs, targets
        except:
            self.tmploader = iter(self.loader)
            inputs, targets = next(self.tmploader)
            return inputs, targets
    
    def get_score(self, model):
        score = self.score
        if score == 'effective_capacity':
            score_value = self.get_effective_capacity(model)
        elif score == 'effective_gradnorm':
            score_value = self.get_effective_gradnorm(model)
        elif score == 'effective_snip':
            score_value = self.get_effective_snip(model)
        elif score == 'effective_grasp':
            score_value = self.get_effective_grasp(model)
        elif score == 'effective_ntktrace':
            score_value = self.get_effective_ntktrace(model)
        elif score == 'params':
            score_value = self.get_params(model)
        elif score == 'gradnorm':
            score_value = self.get_gradnorm(model)
        elif score == 'snip':
            score_value = self.get_snip(model)
        elif score == 'grasp':
            score_value = self.get_grasp(model)
        elif score == 'ntktrace':
            score_value = self.get_ntktrace(model)
        else:
            raise NotImplementedError
        
        return score_value

    def get_params(self, model):
        net = self.get_net(model.arch).cuda()
        score = 0
        for p in net.parameters():
            score += p.numel()
        return score

    def get_gradnorm(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        net.zero_grad()
        _, outputs = net.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    v = m.weight.grad.norm()
                    result += v.sum()
        score = result.item()

        return score

    def get_snip(self, model):
        def snip_forward_conv2d(self, x):
            return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

        def snip_forward_linear(self, x):
            return F.linear(x, self.weight * self.weight_mask, self.bias)

        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        # Compute gradients (but don't apply them)
        split_data = 0
        net.zero_grad()
        N = inputs.shape[0]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data
        
            _, outputs = net.forward(inputs[st:en])
            loss = self.loss_fn(outputs, targets[st:en])
            loss.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight_mask.grad is not None:
                    v = torch.abs(m.weight_mask.grad)
                    result += v.sum()
        score = result

        return score

    def get_grasp(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        # get all applicable weights
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                layer.weight.requires_grad_(True)

        split_data = 1
        num_iters = 1
        T = 1
        net.zero_grad()
        N = inputs.shape[0]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            #forward/grad pass #1
            grad_w = None
            for _ in range(num_iters):
                #TODO get new data, otherwise num_iters is useless!
                outputs = net.forward(inputs[st:en])[1]/T
                loss = self.loss_fn(outputs, targets[st:en])
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
            outputs = net.forward(inputs[st:en])[1]/T
            loss = self.loss_fn(outputs, targets[st:en])
            grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
            
            # accumulate gradients computed in previous step and call backwards
            z, count = 0,0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if grad_w[count] is not None:
                        z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    v = -m.weight.data * m.weight.grad
                    result += v.sum()
        score = result.item()

        return score

    def get_ntktrace(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        net.zero_grad()
        _, outputs = net.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    v = m.weight.grad ** 2
                    result += v.sum()
        score = result.item()

        return score
    
    def get_effective_capacity(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        net.zero_grad()

        _, outputs = net.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    p = m.weight
                    v = (p.grad.data).abs()
                    result += ((v >= self.THR1).sum() - (v >= self.THR2).sum())
        score = result.item()

        return score
    
    def get_effective_gradnorm(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        net.zero_grad()

        # with data, with label
        _, outputs = net.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()


        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    p = m.weight
                    v = (p.grad.data).abs()
                    result += (((v >= self.THR1) * v).sum() - ((v >= self.THR2) * v).sum())
        score = result.item()

        return score

    def get_effective_snip(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()

        net.zero_grad()

        _, outputs = net.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    v = (m.weight.grad.data).abs()
                    gw = (m.weight.data * m.weight.grad.data).abs()
                    result += (((v >= self.THR1) * gw).sum() - ((v >= self.THR2) * gw).sum())
        score = result.item()

        return score

    def get_effective_grasp(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()
        # get all applicable weights
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                layer.weight.requires_grad_(True) # TODO isn't this already true?

        # NOTE original code had some input/target splitting into 2
        # I am guessing this was because of GPU mem limit
        split_data = 1
        num_iters = 1
        T = 1
        net.zero_grad()
        N = inputs.shape[0]
        for sp in range(split_data):
            st=sp*N//split_data
            en=(sp+1)*N//split_data

            #forward/grad pass #1
            grad_w = None
            for _ in range(num_iters):
                #TODO get new data, otherwise num_iters is useless!
                outputs = net.forward(inputs[st:en])[1]/T
                loss = self.loss_fn(outputs, targets[st:en])
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
            outputs = net.forward(inputs[st:en])[1]/T
            loss = self.loss_fn(outputs, targets[st:en])
            grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
            
            # accumulate gradients computed in previous step and call backwards
            z, count = 0,0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if grad_w[count] is not None:
                        with torch.no_grad():
                            grad_w[count].data *= (grad_w[count] >= self.THR1) * (grad_w[count] < self.THR2)
                            # grad_f[count].data *= (grad_f[count] >= THR1) * (grad_f[count] < THR2)
                        z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        result = 0
        count = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    hg = -m.weight.data * m.weight.grad * (grad_w[count] >= self.THR1) * (grad_w[count] < self.THR2)
                    result += hg.abs().sum()
                count += 1
        score = result.item()

        return score

    def get_effective_ntktrace(self, model):
        net = self.get_net(model.arch).cuda()
        inputs, targets = self.get_batch_data()
        inputs, targets = inputs.cuda(), targets.cuda()
        net.zero_grad()

        # with data, with label
        _, outputs = net.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        result = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    p = m.weight
                    v = (p.grad.data).abs()
                    result += (((v >= self.THR1) * v * v).sum() - ((v >= self.THR2) * v * v).sum())
        score = result.item()

        return score


def get_num_classes(dataset):
    return 100 if dataset == 'cifar100' else 10 if dataset == 'cifar10' else 120


def get_final_accuracy(dataset, api, uid, trainval):
    if dataset == 'cifar10':
        if not trainval:
            acc_type = 'ori-test'
        else:
            acc_type = 'x-valid'
    else:
        if not trainval:
            acc_type = 'x-test'
        else:
            acc_type = 'x-valid'
    if dataset == 'cifar10' and trainval:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', acc_type)
    else:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics(dataset, acc_type)
    return info['accuracy']


def random_architecture_func(max_nodes, op_names):
    # return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return random_architecture


def mutate_arch_func(op_names):
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """

    def mutate_arch_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch

    return mutate_arch_func



def my_evolution(
    cycles,
    population_size,
    time_budget,
    random_arch,
    mutate_arch,
    estimator,
    logger
):
    """Algorithm for regularized evolution (i.e. aging evolution).
    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".
    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      time_budget: the upper bound of searching cost
    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population =[]
    history, total_time_cost = [], 0
    arch_set = {}

    # Initialize the population with random models.
    start_time = time.time()
    while len(population) < population_size:
        model = Model()
        model.arch = random_arch()
        model.accuracy = estimator.get_score(model)
        population.append(model)
        history.append(model)
        logger.log(f"{time_string()}: {len(history)}, score {history[-1].accuracy}, arch {history[-1].arch}")
    total_time_cost += time.time() - start_time

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
    #while total_time_cost < time_budget:
        start_time = time.time()

        # Sample randomly chosen models from the current population.
        if random.random() < len(history) / cycles:
            parent = max(population, key=lambda i: i.accuracy)
        else:
            parent = random.choice(list(population))
        # parent = random.choice(list(population))

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.accuracy = estimator.get_score(child)

        population.append(child)
        history.append(child)

        time_cost = time.time() - start_time

        logger.log(f"{time_string()}: {len(history)}, score {history[-1].accuracy}, avg_score {sum([m.accuracy for m in population])/len(population):.2f}, arch {history[-1].arch}")

        if total_time_cost + time_cost > time_budget:  # return
            return history, total_time_cost
        else:
            total_time_cost += time_cost

        # Remove the worst model.
        pos, min_acc = 0, 1e100
        for i, model in enumerate(population):
            if model.accuracy < min_acc:
                min_acc = model.accuracy
                pos = i
        del population[pos]
    return history, total_time_cost


def main(xargs, nas_bench):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)
    acc = {}

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    split_Fpath = "exp_search/NAS-Bench-201/configs/cifar-split.txt"
    cifar_split = load_config(split_Fpath, None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    logger.log("Load split file from {:}".format(split_Fpath))
    config_path = "exp_search/NAS-Bench-201/configs/myEA.config"
    config = load_config(
        config_path, {"class_num": class_num, "xshape": xshape}, logger
    )
    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=xargs.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
        num_workers=xargs.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=xargs.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
        num_workers=xargs.workers,
        pin_memory=True,
    )
    logger.log(
        "||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(
            xargs.dataset, len(train_loader), len(valid_loader), xargs.batch_size
        )
    )
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    estimator = Estimator(xargs.score, train_loader, nn.CrossEntropyLoss().cuda(), xargs.THR1, xargs.THR2)

    search_space = get_search_spaces("cell", xargs.search_space_name)
    random_arch = random_architecture_func(xargs.max_nodes, search_space)
    mutate_arch = mutate_arch_func(search_space)
    # x =random_arch() ; y = mutate_arch(x)
    x_start_time = time.time()
    logger.log("{:} use nas_bench : {:}".format(time_string(), nas_bench))
    logger.log(
        "-" * 30
        + " start searching with the time budget of {:} s".format(xargs.time_budget)
    )
    history, total_cost = my_evolution(
        xargs.ea_cycles,
        xargs.ea_population,
        xargs.time_budget,
        random_arch,
        mutate_arch,
        estimator,
        logger
    )
    logger.log(
        "{:} my evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).".format(
            time_string(), len(history), total_cost, time.time() - x_start_time
        )
    )
    best_arch = max(history, key=lambda i: i.accuracy)
    best_arch = best_arch.arch
    logger.log("{:} best arch is {:}".format(time_string(), best_arch))

    info = nas_bench.query_by_arch(best_arch, "200")
    best_index = nas_bench.query_index_by_arch(best_arch)

    # added by Zhihao Zhang
    acc["cifar10_valid"] = get_final_accuracy("cifar10", nas_bench, best_index, True)
    acc["cifar10_test"] = get_final_accuracy("cifar10", nas_bench, best_index, False)
    acc["cifar100_valid"] = get_final_accuracy("cifar100", nas_bench, best_index, True)
    acc["cifar100_test"] = get_final_accuracy("cifar100", nas_bench, best_index, False)
    acc["in_valid"] = get_final_accuracy("ImageNet16-120", nas_bench, best_index, True)
    acc["in_test"] = get_final_accuracy("ImageNet16-120", nas_bench, best_index, False)

    if info is None:
        logger.log("Did not find this architecture : {:}.".format(best_arch))
    else:
        logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, best_index, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, help="The search space name.")
    parser.add_argument("--max_nodes", type=int, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, help="The number of channels.")
    parser.add_argument(
        "--num_cells", type=int, help="The number of cells in one stage."
    )
    parser.add_argument("--ea_cycles", type=int, help="The number of cycles in EA.")
    parser.add_argument("--ea_population", type=int, help="The population size in EA.")
    parser.add_argument(
        "--time_budget",
        type=int,
        help="The total time cost budge for searching (in seconds).",
    )
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log."
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--score", type=str, default="effective_capacity")
    parser.add_argument("--THR1", type=float, default=0.0001)
    parser.add_argument("--THR2", type=float, default=0.0005)
    args = parser.parse_args()

    if args.arch_nas_dataset is None or not os.path.isfile(args.arch_nas_dataset):
        nas_bench = None
    else:
        print(
            "{:} build NAS-Benchmark-API from {:}".format(
                time_string(), args.arch_nas_dataset
            )
        )
        nas_bench = API(args.arch_nas_dataset)
    if args.rand_seed < 0:
        save_dir, all_indexes, num = None, [], args.n_runs
        stat = {
            "cifar10_valid": [],
            "cifar10_test": [],
            "cifar100_valid": [],
            "cifar100_test": [],
            "in_valid": [],
            "in_test": [],
                }
        for i in range(num):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, num))
            args.rand_seed = i # random.randint(1, 100000)
            save_dir, index, acc = main(args, nas_bench)
            all_indexes.append(index)

            for key in stat:
                stat[key].append(acc[key])
        print("Stats:")
        for key in stat:
            print("{}: {}+-{}".format(key, mean(stat[key]), stdev(stat[key])))
        torch.save(all_indexes, save_dir / "results.pth")
    else:
        main(args, nas_bench)