import lmdb
import datetime
import argparse
import pandas as pd
import numpy as np
import random
import re
from collections import defaultdict
from typing import List, Tuple, Union
import argparse

import scipy.io
import pickle
import os
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm

from torch.optim import AdamW
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import math
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

# from model_utils import PreUNetMLPMixer, PixelTimeLSTM, PreUNetTemporalAttnMixer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser_args(parser):
    parser.add_argument('--seed', type=int, default=41, help='random seed(default: 41)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size(default: 16)')
    parser.add_argument('--dmodel', type=int, default=256, help='model dim of UNet(default: 256)')
    parser.add_argument('--lr', type=float, default=2e-4, help='old learning rate(default: 2e-4)')
    parser.add_argument('--max_lr', type=float, default=4e-4, help='max learning rate(default: 4e-4)')
    parser.add_argument('--min_lr', type=float, default=8e-6, help='min learning rate(default: 8e-6)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay(default: 1e-4)')
    parser.add_argument('--dataset_dir', type=str, default='./', help='lmdb dataset dir')
    parser.add_argument('--result_dir', type=str, default='./', help='result saving dir')
    parser.add_argument('--keep_ratio', type=float, default=0.9, help='keeping ratio for data')
    parser.add_argument('--squash_tanh', type=str2bool, default=True, help='tanh normalization')
    parser.add_argument('--data_scaling_factor', type=int, default=100, help='scaling factor for data when not using tanh scaling')
    parser.add_argument('--data_segment', type=int, default=1, help='number of data segment time points')
    
    # model relevent arguments
    parser.add_argument('--three_d_unet', type=str2bool, default=False, help='use 3d cnn based 3d unet')
    parser.add_argument('--use_lstm', type=str2bool, default=False, help='add lstm in resnet')

    # loss and init proj relevent arguments
    parser.add_argument('--use_wavelet_loss', type=str2bool, default=False, help='use wavelet loss')
    parser.add_argument('--wavelet_loss_weight', type=float, default=0.1, help='wavelet loss weight')
    parser.add_argument('--use_spectral_loss', type=str2bool, default=False, help='use spectral (FFT) loss')
    parser.add_argument('--spectral_loss_weight', type=float, default=0.1, help='spectral loss weight')
    parser.add_argument('--use_perceptual_loss', type=str2bool, default=False, help='use perceptual loss')
    parser.add_argument('--perceptual_loss_weight', type=float, default=0.1, help='perceptual loss weight')
    parser.add_argument('--unetinit_debug', type=str2bool, default=False, help='unetinit debug and data visualization')
    
    # step relevent arguments
    parser.add_argument('--time_steps', type=int, default=1000, help='diffusion time steps')
    parser.add_argument('--total_steps', type=int, default=100000, help='learning total steps')

    # logging relevent arguments
    parser.add_argument('--log_every', type=int, default=200, help='log steps stride')
    parser.add_argument('--eval_every', type=int, default=1000, help='eval steps stride')
    parser.add_argument('--save_samples_every', type=int, default=1000, help='save samples steps stride')
    parser.add_argument('--be_weight', type=int, default=0, help='be weight')

    return parser

parser = argparse.ArgumentParser(description='SOLID for EEG')
parser = get_parser_args(parser)
params = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(params.result_dir, exist_ok=True)