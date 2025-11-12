"""
Based on Edanur Demir to initializer methods are defined in this code.
Modified by Daphnee Chabal for plaintext QUOKKA inference - July 2025
"""
import os
import csv
import argparse
import inspect
import six
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from eenet import EENet
from eenet import eenet18, eenet34, eenet50, eenet101, eenet152
from eenet import eenet20, eenet32, eenet44, eenet56, eenet110
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from resnet import set_complexity
from flops_counter import flops_to_string, params_to_string
from utils import parse_model_filename, parse_weights
from helpers import seed_everything


def initializer():
    """initializer of the program.

    This parses and extracts all training and testing settings.
    """
    parser = argparse.ArgumentParser(description='PyTorch Early-Exit Convolutional Neural Nets')
    parser.add_argument('--batch-size',   type=int,   default=32, metavar='N',
                                          help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch',   type=int,   default=1, metavar='N',
                                          help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs',       type=int,   default=150, metavar='N',
                                          help='number of epochs to train (default: 150)')
    parser.add_argument('--lr',           type=float, default=0.001, metavar='N',
                                          help='learning rate (default: 0.001)')
    parser.add_argument('--adaptive-lr',  action='store_true', default=False,
                                          help='adjust the learning rate')
    parser.add_argument('--momentum',     type=float, default=0.9, metavar='N',
                                          help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='N',
                                          help='weight decay for optimizers (default: 0.0001)')
    parser.add_argument('--no-cuda',      action='store_true', default=True, # Was originally False, but need to work out Crypten implementation if using GPUs
                                          help='disable CUDA training')
    parser.add_argument('--multi-gpu',    action='store_true', default=False,
                                          help='enable multi-gpu training')
    parser.add_argument('--seed',         type=int,   default=1, metavar='N',
                                          help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int,   default=1, metavar='N',
                                          help='how many epochs to wait before logging training '+\
                                           'status (default: 1)')
    parser.add_argument('--no-save-model',action='store_true', default=False,
                                          help='do not save the current model')
    parser.add_argument('--load-model',   type=str,   default=None, metavar='S',
                                          help='the path for loading and evaluating model')
    parser.add_argument('--testing',      action='store_true', default=False,
                                          help='activate testing to loaded model')
    parser.add_argument('--two-stage',    action='store_true', default=False,
                                          help='two stage learning for loss version 1')
   
    parser.add_argument('--lambda-coef',  type=float, default=1.0, metavar='N',
                                          help='lambda to arrange the balance between accuracy '+\
                                           'and cost (default: 1.0)')
    parser.add_argument('--num-ee',       type=int,   default=2, metavar='N',
                                          help='the number of early exit blocks (default: 2)')
    parser.add_argument('--dataset',      type=str,   default='cifar10',
                                          choices=['mnist','cifar10','svhn','imagenet',
                                           'tiny-imagenet'],
                                          help='dataset to be evaluated (default: cifar10)')
    parser.add_argument('--loss-func',    type=str,   default='v2', choices=['v1','v2','v3','v4'],
                                          help='loss function (default: v2)')
    parser.add_argument('--optimizer',    type=str,   default='Adam', choices=['SGD','Adam'],
                                          help='optimizer (default: Adam)')
    parser.add_argument('--distribution', type=str,   default='fine',
                                          choices=['gold_ratio', 'pareto', 'fine', 'linear'],
                                          help='distribution method of exit blocks (default: fine)')
    parser.add_argument('--exit-type',    type=str,   default='pool', choices=['plain', 'pool',
                                           'bnpool'],
                                          help='Exit block type.') # Should be bnpool
    parser.add_argument('--exit_threshold', type=float, default=0.5, metavar='N', 
                                          help='Confidence threshold [0-0.99] necessary to exit')
    parser.add_argument('--model',        type=str,   default='eenet20',
                                          choices=['eenet18', 'eenet34', 'eenet50', 'eenet101', 'eenet152',
                                           'eenet20', 'eenet32', 'eenet44', 'eenet56',  'eenet110',
                                           'resnet18','resnet34','resnet50','resnet101','resnet152',
                                           'resnet20','resnet32','resnet44','resnet56', 'resnet110'
                                          ],
                                          help='model to be evaluated (default: eenet20)')
    parser.add_argument('--device',       help=argparse.SUPPRESS)
    parser.add_argument('--start-epoch',  help=argparse.SUPPRESS)
    parser.add_argument('--recorder',     help=argparse.SUPPRESS)
    parser.add_argument('--results-dir',  help=argparse.SUPPRESS)
    parser.add_argument('--models-dir',   help=argparse.SUPPRESS)
    parser.add_argument('--hist-file',    help=argparse.SUPPRESS)
    parser.add_argument('--num-classes',  help=argparse.SUPPRESS, default=10)
    parser.add_argument('--input-shape',  help=argparse.SUPPRESS, default=(3, 32, 32))
    
    
    args = parser.parse_args()

    if args.dataset == 'mnist':
        args.input_shape = (1, 28, 28)

    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        args.input_shape = (3, 224, 224)

    elif args.dataset == 'tiny-imagenet':
        args.num_classes = 200
        args.input_shape = (3, 64, 64)

    if args.model[:6] == "resnet":
        args.num_ee = 0

    seed_everything(1)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        raise ValueError("Only using cpu for inference - set use_cuda to False!")
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    # model configurations
    kwargs = vars(args)
    if args.load_model is None:
        raise ValueError("For now, need to load model.")

    else:
        print("\n\nLOADING MODEL. . . .")
        weights = torch.load(args.load_model, weights_only=False, map_location=torch.device('cpu'))
        parse_model_filename(args.load_model, args)
        model_object = _get_object(args.model)
        model = model_object(**kwargs)
        parsed_weights = parse_weights(weights, True)
        model.load_state_dict(parsed_weights)
        try:
            model = model.to(args.device) #move to device.
            print(f"LOADING MODEL:          SUCCESS.\n")
        except:
            raise ValueError(f"LOADING MODEL:          FAILURE.")
        
    # use multiple GPU
    if use_cuda and torch.cuda.device_count() > 1 and args.multi_gpu:
        raise ValueError("We are not in a GPU machine.")

    optimizer = None
    return model, optimizer, args


def _get_object(identifier):
    """Object getter.

    This creates instances of the command line arguments by getting related objects.
    """
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier
