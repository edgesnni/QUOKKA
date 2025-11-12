"""
File: secure_main.py
Author: Daphnee Chabal
Date: 2025-02-18

Description: Crypten-ized version of main.py for inference-only from Edanur Demir (https://github.com/eksuas/eenets.pytorch/blob/master/main.py) to enable SNNI (Secure Neural Network Inference) of early-exit networks.
"""

from __future__ import print_function
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy import stats
from secure_init import initializer
from secure_eenet import SEENet
#from plaintext.eenet import EENet
import loss_functions
import utils
import pandas as pd
import fix_hook
fix_hook.fix_torch_tensor()
fix_hook.fix_crypten_module()
fix_hook.fix_deps()

import torch.distributed as dist
import secure_distributed as udist
import secure_common as mc

import crypten 
import crypten.nn as cnn
import crypten.communicator as comm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../plaintext')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import statistics

def str_rank(rank):
    if rank == 0:
        return "SERVER"
    elif rank ==1:
        return "CLIENT"
    else:
        raise ValueError("rank is erroneous")
    
def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    model, args = initializer()
    _, test_loader, exit_tags, classes = utils.load_dataset(args)

    if args.testing:
        assert udist.get_world_size() == 2, ValueError("World size must be 2")
        rank = comm.get().get_rank()
        assert rank == 1, ValueError("Rank must be 1")
        print(f"\n\n****Beginning Inference as {str_rank(rank)}****")
        result = validate(args, model, test_loader, rank, classes)
        return

def validate(args, model, val_loader, rank, classes):
    """validate the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * val_loader:   validation data loader..

    This validates the model and prints the results of each epochs.
    Finally, it returns average accuracy, loss and comptational cost.
    """
    batch = {'inner_time':[], 'time':[], 'cost':[], 'flop':[], 'acc':[], 'val_loss':[]}
    exit_points = [0]*(args.num_ee+1)
    
    num_imgs = len(val_loader)
    comm.get().broadcast_obj(num_imgs, src=1)
    print(f"\n Sent to Server the number of inferences to run ({num_imgs} images).")
    crypt_threshold = crypten.cryptensor(args.exit_threshold, src=1)
    comm.get().broadcast_obj(0, src=1)
    print(f"\nSent to Server the confidence threshold.")

    count = 0
    with crypten.no_grad():
        for batch_id, (data, target) in enumerate(val_loader):
            count += 1
            print(f"image {count}")
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            
            if isinstance(model, SEENet):
                model.eval()
                enc_model = model.encrypt(src=0)
                start = time.process_time()
                crypt_data = crypten.cryptensor(data, src=1)
                zeroed_data = data.new_zeros(data.size())
                comm.get().broadcast_obj(zeroed_data, src=1)
                pred, idx, cost, inner_time = enc_model(crypt_data, crypt_threshold)
                elapsed_time = time.process_time()  - start
                loss = F.nll_loss(pred.log(), target) + args.lambda_coef * cost
                flop = cost * model.complexity[-1][0]
                exit_points[idx] += 1

            pred = pred.max(1, keepdim=True)[1]
            acc = pred.eq(target.view_as(pred)).sum().item()
            batch['acc'].append(acc*100.)
            batch['time'].append(elapsed_time)
            batch['inner_time'].append(inner_time)
            batch['cost'].append(cost*100.)
            batch['flop'].append(flop)
            batch['val_loss'].append(float(loss))

    utils.print_validation(args, batch, exit_points)

    result = {}
    for key, value in batch.items():
        result[key] = round(np.mean(value), 4)
        result[key+'_sem'] = round(stats.sem(value), 2)

     
    print(f"****Ended Inference as {str_rank(rank)}****\n\n")
    return result

if __name__ == '__main__':
    main()
