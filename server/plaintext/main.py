"""
Edanur Demir
Training and validation of EENet
"""
from __future__ import print_function
import time
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from scipy import stats
from init import initializer
from eenet import EENet
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import loss_functions
import utils
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def calculate_topk_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        
        _, pred = output.topk(maxk, 1, True, True) # pred shape: (batch_size, maxk)
        pred = pred.t() 
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # Shape: (maxk, batch_size)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    model, optimizer, args = initializer()
    train_loader, test_loader, exit_tags, classes = utils.load_dataset(args)

    if args.testing:
        result = validate(args, model, test_loader)
        return
    else:
        raise ValueError("Training not implemented")

def validate(args, model, val_loader):
    """validate the model.

    Arguments are
    * args:         command line arguments entered by user.
    * model:        convolutional neural network model.
    * val_loader:   validation data loader..

    This validates the model and prints the results of each epochs.
    Finally, it returns average accuracy, loss and comptational cost.
    """
    
    all_predictions = []
    all_targets = []
    total_top1_correct = 0
    total_top5_correct = 0
    total_top10_correct = 0
    total_samples = 0
    batch = {'cost':[], 'flop':[], \
             'f1':[], 'recall':[], 'precision':[], 'acc':[], 'new_acc':[], \
             'top1':[], 'top5':[], 'top10':[], 'val_loss':[]}
    exit_points = [0]*(args.num_ee+1)
    
    # switch to evaluate mode
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(val_loader):
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)

            if isinstance(model, (EENet)):
                count += 1
                print("image ", count)
                model(data)
                pred, idx, cost = model(data)
                loss = F.nll_loss(pred.log(), target) + args.lambda_coef * cost
                flop = cost * model.complexity[-1][0]
                exit_points[idx] += 1

                top1_batch, top5_batch, top10_batch = calculate_topk_accuracy(pred, target, topk=(1, 5, 10))
                total_top1_correct += top1_batch.item() * data.size(0) / 100.0 
                total_top5_correct += top5_batch.item() * data.size(0) / 100.0
                total_top10_correct += top10_batch.item() * data.size(0) / 100.0 # Accumulate top-10
                total_samples += data.size(0)
                predicted_classes = pred.max(1, keepdim=True)[1] # Get the predicted class index
                all_predictions.append(predicted_classes.squeeze().cpu().numpy())
                all_targets.append(target.cpu().numpy())

            else:   
                raise ValueError("We only use EENet for inference")

            # START ACCURACY CALCULATIONS
            predicted_classes = pred.max(1, keepdim=True)[1]
            acc = predicted_classes.eq(target.view_as(predicted_classes)).sum().item()
            overall_accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            final_top1_accuracy = (total_top1_correct / total_samples) * 100.0
            final_top5_accuracy = (total_top5_correct / total_samples) * 100.0
            final_top10_accuracy = (total_top10_correct / total_samples) * 100.0

            batch['acc'].append(acc*100.)
            batch['new_acc'].append(overall_accuracy)
            batch['cost'].append(cost*100.)
            batch['flop'].append(flop)
            batch['val_loss'].append(float(loss))
            batch['f1'].append(f1)
            batch['precision'].append(precision)
            batch['recall'].append(recall)
            batch['top1'].append(final_top1_accuracy)
            batch['top5'].append(final_top5_accuracy)
            batch['top10'].append(final_top10_accuracy)

    utils.print_validation(args, batch, False, exit_points)
    return 


if __name__ == '__main__':
    main()
