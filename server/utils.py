"""
Edanur Demir, modified by Daphnee Chabal
Utilities are defined in this code.
"""
import os
import csv
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn.functional as F
from plaintext.eenet import EENet
from flops_counter import flops_to_string, params_to_string
from collections import OrderedDict

import sys
import crypten 
import crypten.nn as cnn
import crypten.communicator as comm

plt.switch_backend("agg")
        
def create_empty_dictionary(input_dict):
    """
    Creates a new dictionary with the same structure and keys as the input dictionary,
    but with all tensor values replaced by zero tensors of the same shape.

    Args:
        input_dict (dict): The input dictionary containing torch tensors.

    Returns:
        dict: A new dictionary with zeroed tensors.
    """
    zeroed_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            zeroed_dict[key] = torch.empty_like(value)
        else:
            raise ValueError("All should be tensors")
    return zeroed_dict

def parse_model_filename(model_name, args):
    """Parses the model filename and sets corresponding args attributes."""
    parts = model_name.split('_')
    if len(parts) >= 6:  # Ensure there are enough parts
        try:
            args.num_ee = int(parts[3][2:])
            args.model = parts[2]
            args.exit_type = parts[5]
            args.distribution = parts[4]
        except ValueError:
            print(f"Error parsing model filename: {model_name}")
    else:
        print(f"Invalid model filename format: {model_name}")

def parse_weights(state_dict, plaintext=False):
    new_dict = OrderedDict()
    keys = list(state_dict.keys())
    for key in keys:
        the_tensor = state_dict[key]
        if plaintext:
            key = key.replace(".module", "")
            new_dict[key] = the_tensor
        else:
            if "num_batches" in key:
                pass
            else:
                key = key.replace("confidence", "secure_confidence")
                key = key.replace("classifier", "secure_classifier")
                key = key.replace(".module", "")
                new_dict[key] = the_tensor
    return new_dict
      

def load_dataset(args):
    """dataset loader.

    This loads the dataset.
    """
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.device == 'cuda' else {}

    ### MNIST ###
    if args.dataset == 'mnist':
        root = '../data/mnist'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    ### CIFAR10 ###
    elif args.dataset == 'cifar10':
        root = '../data/cifar10'
        trainset = datasets.CIFAR10(root=root, train=True, download=True,\
            transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

        testset = datasets.CIFAR10(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    ### SVHN ###
    elif args.dataset == 'svhn':
        root = '../data/svhn'
        transform = transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.SVHN(root=root, split='train', download=True, transform=transform)
        testset = datasets.SVHN(root=root, split='test', download=True, transform=transform)
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    ### IMAGENET ###
    elif args.dataset == 'imagenet':
        root = '../data/imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        testset = datasets.ImageFolder(root=root+'/val', transform=transforms.Compose([\
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
        classes = trainset.classes

    ### TINY IMAGENET ###
    elif args.dataset == 'tiny-imagenet':
        create_val_img_folder()
        root = '../data/tiny-imagenet'
        trainset = datasets.ImageFolder(root=root+'/train', transform=transforms.Compose([\
          transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])]))

        testset = datasets.ImageFolder(root=root+'/val/images', transform=transforms.Compose([\
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])]))
    classes = trainset.classes

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                    shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch,\
                    shuffle=False, **kwargs)


    exit_tags = []
    for data, target in train_loader:
        exit_tags.append(torch.randint(0, args.num_ee+1, (len(target), 1)))

    return train_loader, test_loader, exit_tags, classes

def create_val_img_folder():
    """
    This method is responsible for separating validation images into separate sub folders.
    """
    val_dir = '../data/tiny-imagenet/val'
    img_dir = os.path.join(val_dir, 'images')

    file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    file.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

def plot_history(args):
    """plot figures

    Argument is
    * history:  history to be plotted.

    This plots the history in a chart.
    """
    data = pd.read_csv(args.results_dir+'/history.csv')
    data = data.drop_duplicates(subset='epoch', keep="last")

    data = data.sort_values(by='epoch')
    title = 'loss of '+args.model+' on '+args.dataset
    xticks = data[['epoch']]
    yticks = data[['train_loss', 'train_loss_sem', 'val_loss', 'val_loss_sem',
                   'pred_loss', 'pred_loss_sem', 'cost_loss', 'cost_loss_sem']]
    labels = ('epochs', 'loss')
    filename = args.results_dir+'/loss_figure.png'
    plot_chart(title, xticks, yticks, labels, filename)

    title = 'val. accuracy and cost rate of '+args.model+' on '+args.dataset
    xticks = data[['epoch']]
    yticks = data[['acc', 'acc_sem', 'cost', 'cost_sem']]
    labels = ('epochs', 'percent')
    filename = args.results_dir+'/acc_cost_figure.png'
    plot_chart(title, xticks, yticks, labels, filename)

    data = data.sort_values(by='flop')
    title = 'val. accuracy vs flops of '+args.model+' on '+args.dataset
    xticks = data[['flop', 'flop_sem']]
    yticks = data[['acc', 'acc_sem']]
    labels = ('flops', 'accuracy')
    filename = args.results_dir+'/acc_vs_flop_figure.png'
    plot_chart(title, xticks, yticks, labels, filename)

def plot_chart(title, xticks, yticks, labels, filename):
    """draw chart

    Arguments are
    * title:     title of the chart.
    * xtick:     array that includes the xtick values.
    * yticks:    array that includes the ytick values.
    * labels:    labels of x and y axises.
    * filename:  filename of the chart.

    This plots the history in a chart.
    """
    _, axis = plt.subplots()
    axis.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))

    xerr = None
    for key, value in xticks.items():
        if key.endswith('_sem'):
            xerr = value
        else: xtick = value

    if all(float(x).is_integer() for x in xtick):
        axis.xaxis.set_major_locator(tick.MaxNLocator(integer=True))

    xlabel, ylabel = labels
    min_x = np.mean(xtick)
    if min_x // 10**9 > 0:
        xlabel += ' (GMac)'
    elif min_x // 10**6 > 0:
        xlabel += ' (MMac)'
    elif min_x // 10**3 > 0:
        xlabel += ' (KMac)'

    legend = []
    for key, value in yticks.items():
        if not key.endswith('_sem'):
            legend.append(key)
            ytick = value
            yerr = yticks[key+'_sem']
            plt.errorbar(xtick, ytick, xerr=xerr, yerr=yerr, capsize=3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc='best')
    plt.savefig(filename)
    plt.clf()
    print('The figure is plotted under \'{}\''.format(filename))

def display_examples(args, model, dataset):
    """display examples

    Arguments are
    * model:    model object.
    * dataset:  dataset loader object.

    This method shows the correctly predicted sample of images from dataset.
    Produced table shows the early exit block which classifies that samples.
    """
    images = [[[] for j in range(10)] for i in range(args.num_ee+1)]
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataset):
            data = data.view(-1, *args.input_shape)
            data, target = data.to(args.device), target.to(args.device).item()
            output, idx, _ = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1].item()
            #print(f"utils 233 pred: {pred}")

            if pred == target:
                if len(images[idx][target]) < 10:
                    images[idx][target].append(idx)

        for idx in range(args.num_ee+1):
            fig, axarr = plt.subplots(10, 10)
            for class_id in range(10):
                for example in range(10):
                    axarr[class_id, example].axis('off')
                for example in range(len(images[idx][class_id])):
                    axarr[class_id, example].imshow(
                        dataset[images[idx][class_id][example]][0].view(args.input_shape[1:]))
            fig.savefig(args.results_dir+'/exitblock'+str(idx)+'.png')

def save_model(args, model, epoch, best=False):
    """save model

    Arguments are
    * model:    model object.
    * best:     version number of the best model object.

    This method saves the trained model in pt file.
    """
    # create the folder if it does not exist
    if not os.path.exists(args.models_dir):
        os.makedirs(args.model_dir)

    filename = args.models_dir+'/model'
    if best is False:
        torch.save(model, filename+'.v'+str(epoch)+'.pt')
    else:
        train_files = os.listdir(args.models_dir)
        for train_file in train_files:
            if not train_file.endswith('.v'+str(epoch)+'.pt'):
                os.remove(os.path.join(args.models_dir, train_file))
        os.rename(filename+'.v'+str(epoch)+'.pt', filename+'.pt')

def get_active_exit(args, epoch):
    quantize = min(args.num_ee, ((epoch-1) // 7))
    sequential = (epoch-1) % (args.num_ee+1)
    return args.num_ee - sequential

def adaptive_learning_rate(args, optimizer, epoch):
    """adaptive learning rate

    Arguments are
    * optimizer:   optimizer object.
    * epoch:       the current epoch number when the function is called.

    This method adjusts the learning rate of training. The same configurations as the ResNet paper.
    """
    # Default SGD: lr=? momentum=0, dampening=0, weight_decay=0, nesterov=False
    if args.optimizer == "SGD":
        # Assummed batch-size is 128. Converge until 165 epochs

        if args.dataset == "cifar10":
            learning_rate = 0.1
            # EENet-110 model
            if args.model[-3:] == "110" and epoch <= 2:
                learning_rate = 0.01

            # divide by 10 per 100 epochs
            if epoch % 100 == 0:
                learning_rate = 0.1 / (10**(epoch // 100))

        # Assumed batch-size is 256. Converge until 150-160 epochs
        elif args.dataset == "imagenet":
            learning_rate = 0.05 * 0.1**((epoch-1) // 30 + 1)

    # Default Adam: lr=0.001 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    # elif args.optimizer == "Adam":
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def to_csv(csv_filename, df):
            if os.path.exists(csv_filename):
                existing_df = pd.read_csv(csv_filename)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(csv_filename, index=False)
                print(f"Appended new data to {csv_filename}")
            else:
                # CSV file doesn't exist, create it
                df.to_csv(csv_filename, index=False)
                print(f"Created {csv_filename} and saved data")
                
def print_validation(args, batch, time=True, exit_points=None):
    """print validation results

    Arguments are
    * batch:         validation batch results.
    * exit_points:   the number of samples exiting from the specified exit blocks.

    This method prints the results of validation.
    """
    # print the validation results of epoch
    print('\n\nRESULTS (average over entire validation dataset):')
    if time:
        print('Average Secure Inference Time for CLIENT: {:.4f}s [std: {:.4f}s] (inner time: {:.4f}s [std: {:.4f}s])'.format(np.mean(batch['time']),
                    np.std(batch['time']), np.mean(batch['inner_time']), np.std(batch['inner_time'])))
    print('Validation Loss: {:.4f} ; Validation Accuracy: {:.2f}% ; New Acc: {:.2f}%'
          .format(np.mean(batch['val_loss']),
                  np.mean(batch['acc']),
                  np.mean(batch['new_acc'])))
    print('Top-1 : {:.2f} ; Top-5 : {:.2f}% ; Top-10 : {:.2f}%'
          .format(np.mean(batch['top1']),
                  np.mean(batch['top5']),
                  np.mean(batch['top10'])))
    print('F1-score: {:.2f} ; Precision: {:.2f}% ; Recall: {:.2f}%'
          .format(np.mean(batch['f1']),
                  np.mean(batch['precision']),
                  np.mean(batch['recall'])))

    # detail print for EENet based models
    if exit_points is not None:
        print('FLOPs Cost: {:.2f}%; exits: ['.format(np.mean(batch['cost'])), end='')
        for i in range(args.num_ee+1):
            print('{:d},'.format(exit_points[i]), end='')
        print(']')
        print("\n\n")

def save_history(args, record):
    """save a record to the history file"""
    args.recorder.writerow([str(record[key]) for key in sorted(record)])
    args.hist_file.flush()

def close_history(args):
    """close the history file and print the best record in the history file"""
    args.hist_file.close()
    args.hist_file = open(args.results_dir+'/history.csv', 'r', newline='')
    reader = csv.DictReader(args.hist_file)
    best_epoch = 0
    best = {}
    for epoch, record in enumerate(reader):
        if not best or record['val_loss'] < best['val_loss']:
            best = record
            best_epoch = epoch+1

    print('\nThe best avg val_loss: {}, avg val_cost: {}%, avg val_acc: {}%\n'
           .format(best['val_loss'], best['cost'], best['acc']))

    return best_epoch

def x_fmt(x_value, _):
    """x axis formatter"""
    if x_value // 10**9 > 0:
        return '{:.1f}'.format(x_value / 10.**9)
    if x_value // 10**6 > 0:
        return '{:.1f}'.format(x_value / 10.**6)
    if x_value // 10**3 > 0:
        return '{:.1f}'.format(x_value / 10.**3)
    return str(x_value)
