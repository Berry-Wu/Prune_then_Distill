# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/6 15:50 
# @Author : wzy 
# @File : arg_parse.py
# ---------------------------------------
import argparse
import torch


def parse_args():
    parse = argparse.ArgumentParser(description="The hyper-parameter of Prune and Distill")
    parse.add_argument('-b', '--bs', default=128)
    parse.add_argument('-l', '--lr', default=1e-3)
    parse.add_argument('-e', '--epoch', default=10)
    parse.add_argument('-d', '--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parse.parse_args()
    return args