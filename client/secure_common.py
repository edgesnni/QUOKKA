import copy
import importlib
import logging
import math
import os, sys

import torch
import torch.nn.functional as F
import torch.distributed as dist

import crypten 
import crypten.nn as cnn

import secure_distributed as udist

# import torch.nn.functional as F






def setup_distributed(num_images=None):
    """Setup distributed related parameters."""
    # init distributed
    udist.init_dist(backend="gloo")