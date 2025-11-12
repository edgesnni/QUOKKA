import crypten 
import crypten.nn as cnn
import torch
import operator
import random
import numpy as np

def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False #Set to false for consistent results.

def Secureconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return cnn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def Secureconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return cnn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
def encrypt_compare(x, y, comparison_operator):
    valid_operators = {
    operator.gt,
    operator.lt,
    operator.ge,
    operator.le
}
    if comparison_operator not in valid_operators:
        raise ValueError("comparison argument needs to be a valid operator: >, <, >=, or <=")
    if isinstance(x, crypten.mpc.MPCTensor):
        if isinstance(y, crypten.mpc.MPCTensor):
            res = comparison_operator(x, y).get_plain_text().item()
        elif isinstance(y, float):
            res = comparison_operator(x, y).get_plain_text().item()
        else:
            raise ValueError(f"y needs to be a crypten Tensor or float, not {type(y)}")
    else:
        raise ValueError(f"x needs to be a crypten Tensor not {type(x)}")
    if res == 1:
        return True
    elif res == 0:
        return False
    else: 
        raise ValueError(f"Result \"{res}\" is invalid.")