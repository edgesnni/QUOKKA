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
    
    if args.testing:
        assert udist.get_world_size() == 2, ValueError("World size must be 2")
        rank = comm.get().get_rank()
        assert rank == 0, ValueError("Rank must be 0 - this is the SERVER")
        print(f"\n\n****Beginning Inference as {str_rank(rank)}****")
        result = validate(args, model, None, rank, None)
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
    batch = {'inner_time':[], 'time':[]}
    exit_points = [0]*(args.num_ee+1)
    
    len_loader = comm.get().broadcast_obj(None, src=1)
    print(f"\nReceived from Client the number of inferences to run ({len_loader} images).")
    thres = comm.get().broadcast_obj(None, src=1)
    crypt_threshold = crypten.cryptensor(thres, src=1)
    print(f"\nReceived from Client the confidence threshold.")

    count = 0
    with crypten.no_grad():
        for i in range(len_loader):
            count += 1
            print(f"image {count}")

            if isinstance(model, SEENet):
                model.eval()
                start = time.process_time()
                enc_model = model.encrypt(src=0)
                zeroed_data = comm.get().broadcast_obj(None, src=1)
                crypt_data = crypten.cryptensor(zeroed_data, src=1)
                inner_time = enc_model(crypt_data, crypt_threshold)
                elapsed_time = time.process_time()  - start
         
            batch['time'].append(elapsed_time)
            batch['inner_time'].append(inner_time)
        print(f"2. Finished {len_loader} inferences.")
        print('\n\nAverage Secure Inference Time for SERVER: {:.4f}s [std: {:.4f}s] (inner time: {:.4f}s [std: {:.4f}s])'.format(np.mean(batch['time']),
                  np.std(batch['time']), np.mean(batch['inner_time']), np.std(batch['inner_time'])))
        print(f"\n\n****Ended all Inferences as {str_rank(rank)}****\n\n")

if __name__ == '__main__':
    main()
