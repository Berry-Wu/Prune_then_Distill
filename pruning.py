# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/6 15:38 
# @Author : wzy 
# @File : pruning.py
# @Notes : unstructured prune ; the 'LR rewinding' method of retrain the pruned model
# @Reference : https://github.com/ososos888/prune-then-distill
# ---------------------------------------
import copy
from torch import nn
from utills.arg_parse import parse_args
import torch.nn.utils.prune as prune
import wzy.Prune_then_Distill.train as train
from models.resnet import resnet18
from torchsummary import summary

args = parse_args()


def pruning_loop(model, args):
    for i in range(args.epoch):
        # prune
        model_pruning = copy.deepcopy(model)
        model_pruning, pruning_module_list, pruning_modulename_list = get_prune_module_info(model_pruning)

        # sparsity = 1 - np.power(0.8, i + 1)
        sparsity = 0.2
        # l1 global unstructured pruning，这里根据稀疏度对模块参数进行剪枝
        prune.global_unstructured(pruning_module_list, pruning_method=prune.L1Unstructured, amount=sparsity)

        # 根据copy模型被mask的内容，对原模型进行mask
        model = mask_copy(model, model_pruning, pruning_modulename_list)

        # fine-tuning
        train.main(model, device=args.device, mode=2)


def get_prune_module_info(model_pruning):
    """
    对copy的模型进行模块及对应名称获取
    :param model_pruning: The model copied for a prune.
    :return: pruning_module_list: List of module information for global l1pruning (2-dim(modules, module name))
    :return: pruning_modulename_list: List of names of modules to be pruned.
    """
    pruning_module_list = []
    pruning_modulename_list = []

    # 使用named_modules函数获得模型每层名称和module
    for name, module in model_pruning.named_modules():
        # 只对这两个结构进行剪枝
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # named_buffers：获取模块名称及对应参数(buffers在layers.py中对结构进行了修改，实现了buffers的注册)
            for mask_name, mask_param in model_pruning.named_buffers():
                #
                if f"{name}.weight_mask" in mask_name:
                    pruning_module_list.append((module, 'weight'))
                    pruning_modulename_list.append(mask_name)
    return model_pruning, pruning_module_list, pruning_modulename_list


def mask_copy(model, model_pruning, pruning_modulename_list):
    """
    将model_pruning被mask的内容根据对应的模块名称查找，并实现对原模型的修改
    :param model:
    :param model_pruning:
    :param pruning_modulename_list:
    :return:
    """
    for name_copiedmodel, mask_copiedmodel in model_pruning.named_buffers():
        # 找到哪些模块被prune，获取其名称
        if name_copiedmodel in pruning_modulename_list:
            # 原始模型的模块名称及对应mask，此时都是1
            for name_origmodel, mask_origmodel in model.named_buffers():
                # 如果根据copy的model中被剪枝的模块名称，对应修改原模型的模块参数
                if name_copiedmodel == name_origmodel:
                    mask_origmodel.data = mask_copiedmodel.data.clone().detach()
    return model


def prune_main(model, args):
    # 传入普通训练好的模型，对其进行剪枝
    print('【after prune】')
    pruning_loop(model, args)


if __name__ == '__main__':
    model = resnet18(input_shape=(3, 32, 32), num_classes=10).to(args.device)
    print('【before prune】')
    train.main(model, device=args.device, mode=1)
    prune_main(model, args)
