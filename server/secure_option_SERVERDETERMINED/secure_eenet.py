"""
File: secure_eenet.py
Author: Daphnee Chabal
Date: 2025-02-18

Description: Crypten-ized version of EENet from Edanur Demir (https://github.com/eksuas/eenets.pytorch/blob/master/eenet.py) to enable SNNI (Secure Neural Network Inference) of early-exit networks.
"""
import sys, os
import time
import torch
from torch import nn

import crypten 
import crypten.nn as cnn
import crypten.communicator as comm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../plaintext')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from secure_resnet import SResNet, SResNet6n2
from plaintext.resnet import ResNet, ResNet6n2
from plaintext.eenet import EENet, BasicBlock, Bottleneck, ExitBlock, conv1x1, conv3x3

from flops_counter import get_model_complexity_info
from helpers import Secureconv1x1, Secureconv3x3
import random
import numpy as np
import io
import pandas as pd



__all__ = ['SEENet',
           'seenet18', 'seenet34', 'seenet50', 'seenet101', 'seenet152',
           'seenet20', 'seenet32', 'seenet44', 'seenet56', 'seenet110',]


class SecureDownsample(cnn.Module):
    """
    Downsampling connection for Bottleneck block, replacing cnn.Sequential.
    It consists of a 1x1 convolution followed by BatchNorm2d.
    """
    def __init__(self, inplanes, outplanes, stride=1):
        super(SecureDownsample, self).__init__()
        # Secureconv1x1 must be defined elsewhere and is assumed to be cnn.Module
        self.conv = Secureconv1x1(inplanes, outplanes, stride)
        self.bn = cnn.BatchNorm2d(outplanes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Secure_BasicBlock(cnn.Module):
    """Basic Block defition.

    Basic 3X3 convolution blocks for use on SResNets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Secure_BasicBlock, self).__init__()
        self.conv1 = Secureconv3x3(inplanes, planes, stride)
        self.bn1 = cnn.BatchNorm2d(planes)
        self.relu = cnn.ReLU()
        self.conv2 = Secureconv3x3(planes, planes)
        self.bn2 = cnn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Secure_Bottleneck(cnn.Module):
    """Bottleneck Block defition.

    Bottleneck architecture for > 34 layer SResNets.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Secure_Bottleneck, self).__init__()
        self.conv1 = Secureconv1x1(inplanes, planes)
        self.bn1 = cnn.BatchNorm2d(planes)
        self.conv2 = Secureconv3x3(planes, planes, stride)
        self.bn2 = cnn.BatchNorm2d(planes)
        self.conv3 = Secureconv1x1(planes, planes * self.expansion)
        self.bn3 = cnn.BatchNorm2d(planes * self.expansion)
        self.relu = cnn.ReLU()
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity 
        
        out = self.relu(out)

        return out

class Secure_ExitBlock(cnn.Module):
    """Exit Block definition, recoded without cnn.Sequential or cnn.ModuleList."""
    def __init__(self, inplanes, num_classes, input_shape, exit_type):
        super(Secure_ExitBlock, self).__init__()
        _, width, height = input_shape
        self.exit_type = exit_type

        if exit_type != "pool":
            raise ValueError("Secure inference was only implemented for pool exits.")

        self.expansion = width * height if exit_type == 'plain' else 1
        
        self.adp = cnn.AdaptiveAvgPool2d(1)
        
        self.confidence_linear = cnn.Linear(inplanes * self.expansion, 1)
        self.confidence_activation = cnn.Sigmoid()
        
        self.classifier_linear = cnn.Linear(inplanes * self.expansion, num_classes)
        self.classifier_activation = cnn.Softmax(dim=1)

    def forward(self, x):
        x = self.adp(x)
        x = x.view(x.size(0), -1)
        conf = self.confidence_linear(x)
        conf = self.confidence_activation(conf)
        pred = self.classifier_linear(x)
        pred = self.classifier_activation(pred)
        
        return pred, conf
    
class SEENet(cnn.Module):
    """Builds a EENet like architecture.

    Arguments are
    * is_6n2model:        Whether the architecture of the model is 6n+2 layered SResNet.
    * block:              Block function of the architecture either 'BasicBlock' or 'Bottleneck'.
    * total_layers:       The total number of layers.
    * repetitions:        Number of repetitions of various block units.
    * num_ee:             The number of early exit blocks.
    * distribution:       Distribution method of the early exit blocks.
    * num_classes:        The number of classes in the dataset.
    * zero_init_residual: Zero-initialize the last BN in each residual branch,
                          so that the residual branch starts with zeros,
                          and each residual block behaves like an identity. This improves the model
                          by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    * input_shape:        Input shape of the model according to dataset. Should always be false because not implemented in Crypten.
    
    Returns:
        The cnn.Module.
    """
    def __init__(self, is_6n2model, block, original_block, total_layers, num_ee, distribution, num_classes,
                 input_shape, model, exit_type, exit_threshold, loss_func, repetitions=None, zero_init_residual=False,
                 **kwargs):
        super(SEENet, self).__init__()
        if is_6n2model:
            raise ValueError("not for eenet152")
            self.inplanes = 16
            repetitions = [(total_layers-2) // 6]*3
            torch_counterpart_model = ResNet6n2(original_block, total_layers, num_classes, input_shape)
        else:
            self.inplanes = 64
            torch_counterpart_model = ResNet(original_block, repetitions, num_classes, input_shape)

        torch_layers = nn.ModuleList()
        torch_exits = nn.ModuleList()
        torch_stages = nn.ModuleList()
        self.stages = []
        self.layers = []
        self.cost = []
        self.complexity = []
        self.model = model
        self.stage_id = 0
        self.num_ee = num_ee
        self.total_layers = total_layers
        self.exit_type = exit_type
        self.distribution = distribution
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.exit_threshold = exit_threshold

        if loss_func == 'v3':
            raise ValueError("We don't train here.")
            self.exit_threshold = 1./self.num_ee

        channel, _, _ = input_shape
        total_flops, total_params = self.get_complexity(torch_counterpart_model)
        self.set_thresholds(distribution, total_flops)

        if is_6n2model:
            raise ValueError("not for eenet152")

        else:

            torch_layers.append(nn.Sequential(
                nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ))

            #INTRO LAYERS
            self.convv = cnn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bnn = cnn.BatchNorm2d(64)
            self.ree = cnn.ReLU()
            self.mxx = cnn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            #EXIT LAYERS
            if self.model == "seenet18":
                self.adp = cnn.AdaptiveAvgPool2d(1)
                self.conf_0_1_128 = cnn.Linear(128, 1)
                self.conf_0_2 = cnn.Sigmoid()
                self.class_0_1_128 = cnn.Linear(128, num_classes)
                self.class_0_2 = cnn.Softmax(dim=1)

                self.conf_0_1_64 = cnn.Linear(64, 1)
                self.class_0_1_64 = cnn.Linear(64, num_classes)
                
                self.conf_1_256 = cnn.Linear(256, 1)
                self.conf_2 = cnn.Sigmoid()
                self.class_1_256 = cnn.Linear(256, num_classes)
                self.class_2 = cnn.Softmax(dim=1)
                self.conf_1_128 = cnn.Linear(128, 1)
                self.class_1_128 = cnn.Linear(128, num_classes)
            #FOR EENET152
            elif self.model == "seenet152": 
                self.adp = cnn.AdaptiveAvgPool2d(1)
                self.conf_0_1 = cnn.Linear(512, 1)
                self.conf_0_2 = cnn.Sigmoid()

                self.conf_1 = cnn.Linear(1024, 1)
                self.conf_2 = cnn.Sigmoid()

                self.class_0_1 = cnn.Linear(512, num_classes)
                self.class_0_2 = cnn.Softmax(dim=1)

                self.class_1 = cnn.Linear(1024, num_classes)
                self.class_2 = cnn.Softmax(dim=1)
            else:
                raise ValueError("only 152 and 18 implemented")            

        planes = self.inplanes
        stride = 1
        for repetition in repetitions:
            downsample = None
            torch_downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = SecureDownsample(
                    self.inplanes, 
                    planes * block.expansion, 
                    stride
                )
                
                torch_downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            torch_layers.append(original_block(self.inplanes, planes, stride, torch_downsample))
            self.layers.append(block(self.inplanes, planes, stride, downsample))

            self.inplanes = planes * block.expansion
            if self.is_suitable_for_exit(torch_stages, torch_layers):
                self.layers, self.stages, torch_stages, torch_layers, torch_exits = self.add_exit_block(exit_type, total_flops, self.layers, self.stages, torch_stages, torch_layers, torch_exits)

            for _ in range(1, repetition):
                torch_layers.append(original_block(self.inplanes, planes))
                self.layers.append(block(self.inplanes, planes))

                if self.is_suitable_for_exit(torch_stages, torch_layers):
                    self.layers, self.stages, torch_stages, torch_layers, torch_exits = self.add_exit_block(exit_type, total_flops, self.layers, self.stages, torch_stages, torch_layers, torch_exits)

            planes *= 2
            stride = 2
        assert len(torch_exits) == num_ee, \
            'The desired number of exit blocks is too much for the model capacity.'

        planes = 64 if is_6n2model else 512
        
        torch_layers.append(nn.AdaptiveAvgPool2d(1))
        added_layer = cnn.AdaptiveAvgPool2d(1)

        # FOR EENET18
        if self.model == "seenet18":
            self.final_conf_1_256 = cnn.Linear(256, 1)
            self.final_conf_2 = cnn.Sigmoid()

            self.final_class_1_256 = cnn.Linear(256, num_classes)
            self.final_class_2 = cnn.Softmax(dim=1)
            
            self.final_conf_1_512 = cnn.Linear(512, 1)
            self.final_class_1_512 = cnn.Linear(512, num_classes)

        #FOR EENET152
        elif self.model == "seenet152":
            self.final_conf_1 = cnn.Linear(2048, 1)
            self.final_conf_2 = cnn.Sigmoid()

            self.final_class_1 = cnn.Linear(2048, num_classes)
            self.final_class_2 = cnn.Softmax(dim=1)
        else:
            raise ValueError("only 152 and 18 implemented")


        self.stages.append(self.layers)
        torch_stages.append(nn.Sequential(*torch_layers))

        self.softmax = cnn.Softmax(dim=1)
        self.complexity.append((total_flops, total_params))


    def get_complexity(self, model):
        """get model complexity in terms of FLOPs and the number of parameters"""
        flops, params = get_model_complexity_info(model, self.input_shape,\
                        print_per_layer_stat=False, as_strings=False)
        return flops, params


    def add_exit_block(self, exit_type, total_flops, layers, stages, torch_stages, torch_layers, torch_exits):
        """add early-exit blocks to the model

        Argument is
        * total_flops:   the total FLOPs of the counterpart model.

        This add exit blocks to suitable intermediate position in the model,
        and calculates the FLOPs and parameters until that exit block.
        These complexity values are saved in the self.cost and self.complexity.
        """
        torch_stages.append(nn.Sequential(*torch_layers))

        stages.append(layers)
        torch_exits.append(ExitBlock(self.inplanes, self.num_classes, self.input_shape, exit_type))
        
        torch_intermediate_model = nn.Sequential(*(list(torch_stages)+list(torch_exits)[-1:]))
        flops, params = self.get_complexity(torch_intermediate_model)
        self.cost.append(flops / total_flops)
        self.complexity.append((flops, params))
        
        layers = []
        torch_layers = nn.ModuleList()

        self.stage_id += 1
        return layers, stages, torch_stages, torch_layers, torch_exits
        

    def set_thresholds(self, distribution, total_flops):
        """set thresholds

        Arguments are
        * distribution:  distribution method of the early-exit blocks.
        * total_flops:   the total FLOPs of the counterpart model.

        This set FLOPs thresholds for each early-exit blocks according to the distribution method.
        """
        gold_rate = 1.61803398875
        flop_margin = 1.0 / (self.num_ee+1)
        self.threshold = []
        for i in range(self.num_ee):
            if distribution == 'pareto':
                self.threshold.append(total_flops * (1 - (0.8**(i+1))))
            elif distribution == 'fine':
                self.threshold.append(total_flops * (1 - (0.95**(i+1))))
            elif distribution == 'linear':
                self.threshold.append(total_flops * flop_margin * (i+1))
            elif distribution == "gold":
                self.threshold.append(total_flops * (gold_rate**(i - self.num_ee)))


    def is_suitable_for_exit(self, torch_stages, torch_layers):
        """is the position suitable to locate an early-exit block"""
        torch_intermediate_model = nn.Sequential(*(list(torch_stages)+list(torch_layers)))
        flops, _ = self.get_complexity(torch_intermediate_model)
        return self.stage_id < self.num_ee and flops >= self.threshold[self.stage_id]


    def forward(self, x):
        for idx, crypt_exit in enumerate(self.exits):
            starttime = time.process_time()
            crypt_stages = self.stages[idx] 
            x = crypt_stages(x) 
            enc_pred, enc_conf = crypt_exit(x)
            endtime = time.process_time()
            elapsedtime = endtime-starttime
            pred = enc_pred.get_plain_text()
            conf = enc_conf.get_plain_text()
            instruction_from_client = comm.get().broadcast_obj(None, src=1)
            if instruction_from_client == "stop":
                return elapsedtime
            else:
                pass
            
        x = self.stages[-1](x)
        x = x.view(x.size(0), -1)
        enc_pred = self.secure_classifier(x)
        enc_conf = self.secure_confidence(x)
        endtime2 = time.process_time()
        pred = enc_pred.get_plain_text()
        conf = enc_conf.get_plain_text()
        elapsedtime2 = endtime2-starttime
        return elapsedtime2

def seenet18(**kwargs):
    """EENet-18 model"""
    model = SEENet(is_6n2model=False, block=Secure_BasicBlock, original_block=BasicBlock, total_layers=18,
                  repetitions=[2, 2, 2, 2], **kwargs)
    return model

def seenet34(**kwargs):
    """EENet-34 model"""
    model = SEENet(is_6n2model=False, block=Secure_BasicBlock, original_block=BasicBlock, total_layers=34,
                  repetitions=[3, 4, 6, 3], **kwargs)
    return model

def seenet50(**kwargs):
    """EENet-50 model"""
    model = SEENet(is_6n2model=False, block=Secure_Bottleneck, original_block=Bottleneck, total_layers=50,
                  repetitions=[3, 4, 6, 3], **kwargs)
    return model

def seenet101(**kwargs):
    """EENet-101 model"""
    model = SEENet(is_6n2model=False, block=Secure_Bottleneck, original_block=Bottleneck, total_layers=101,
                  repetitions=[3, 4, 23, 3], **kwargs)
    return model

def seenet152(**kwargs):
    """EENet-152 model"""
    model = SEENet(is_6n2model=False, block=Secure_Bottleneck, original_block=Bottleneck, total_layers=152,
                  repetitions=[3, 8, 36, 3], **kwargs)
    return model

def seenet20(**kwargs):
    """EENet-20 model"""
    model = SEENet(True, Secure_BasicBlock, BasicBlock, 20, **kwargs)
    return model

def seenet32(**kwargs):
    """EENet-32 model"""
    model = SEENet(True, Secure_BasicBlock, BasicBlock, 32, **kwargs)
    return model

def seenet44(**kwargs):
    """EENet-44 model"""
    model = SEENet(True, Secure_BasicBlock, BasicBlock, 44, **kwargs)
    return model

def seenet56(**kwargs):
    """EENet-56 model"""
    model = SEENet(True, Secure_BasicBlock, BasicBlock, 56, **kwargs)
    return model

def seenet110(**kwargs):
    """EENet-110 model"""
    model = SEENet(True, Secure_BasicBlock, BasicBlock, 110, **kwargs)
    return model
