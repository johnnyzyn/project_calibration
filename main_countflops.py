import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
# from scipy.special import softmax
import datetime
import argparse
from thop import profile

from nats_bench import create
from nats_bench.api_utils import time_string
from xautodl.models import get_cell_based_tiny_net

import data.cifar10 as cifar10
import data.cifar100 as cifar100
import calibration as cal
import calibration.metric as metric
import calibration.tace as tace
from xautodl.datasets.get_dataset_with_transform import get_datasets
from torch.utils.data import DataLoader
# import calibration.ece_kde as ece_kde
import inspect

from calibration.temperature_scaling import ModelWithTemperature
from calibration.temp_scale import accuracy
from torch.utils.data.sampler import SubsetRandomSampler

import os

def get_preds_and_targets(model, dataloader, device):
    preds, pred_classes, targets = [], [], []

    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to the selected device (CPU or GPU)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output_tuple = model(data)

            output = output_tuple[1]

            prob = F.softmax(output, dim = 1)  # Compute probabilities
            _, pred = torch.max(prob, 1)  # Get predicted class

            preds.extend(prob.cpu().numpy())  # Move probabilities to CPU and convert to numpy array
            pred_classes.extend(pred.cpu().numpy())  # Move predictions to CPU and convert to numpy array
            targets.extend(target.cpu().numpy())  # Move targets to CPU and convert to numpy array

    return np.array(preds), np.array(pred_classes), np.array(targets)

def get_preds_and_targets2(model, dataloader, device):
    #use this when using temp scale since the network output is logits not tuple 

    preds, pred_classes, targets = [], [], []

    model.eval()  
    model.to(device)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output_tuple = model(data)

            output = output_tuple

            prob = F.softmax(output, dim=1)
            _, pred = torch.max(prob, 1)

            preds.extend(prob.cpu().numpy())  # Move probabilities to CPU and convert to numpy array
            targets.extend(target.cpu().numpy())  # Move targets to CPU and convert to numpy array

    return np.array(preds), np.array(pred_classes), np.array(targets)



def get_param_dict(func, *args, **kwargs):
    result = func(*args, **kwargs)
    
    # Get the function's signature and parameters
    signature = inspect.signature(func)
    params = signature.parameters

    # Create a dictionary with default parameter values
    default_params = {k: v.default for k, v in params.items() if v.default != inspect.Parameter.empty}

    # Update the default parameter values with the provided kwargs
    all_params = {**default_params, **kwargs}
    
    all_params['result'] = result
    return all_params


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Model evaluation and result storage")
    parser.add_argument("--csv_file", type=str, default="results.csv", help="CSV file name to store or update results")
    parser.add_argument("--bin_sizes", type=int, default=[5, 10,15,20,25,50,100,200,500], help="Number of bins for calibration metrics")
    parser.add_argument("--archi_num", type=int, default=6111, help="Key number of architecture")
    parser.add_argument("--image_dataset", type=str, default="cifar10", help="CIFAR-10, CIFAR-100, and ImageNet16-120")
    parser.add_argument("--api_type", type=str, default="tss", help="NATS-Bench dataset type (tss or sss)")
    parser.add_argument("--post_temp", type=str, default='False', help="if using temp scale")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")

    args = parser.parse_args()

    # Use the parsed arguments
    bin_sizes = args.bin_sizes
    archi_num = args.archi_num
    csv_file = args.csv_file
    image_dataset = args.image_dataset
    api_type = args.api_type
    post_temp = args.post_temp
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(post_temp)


    sss_dir = "/hdd/datasets/NATSBench/sss-full/"
    tss_dir = "/hdd/datasets/NATSBench/NATS-tss-v1_0-3ffb9-full-extracted/NATS-tss-v1_0-3ffb9-full/"

    
    if api_type =='tss':
        api = create(tss_dir, api_type, fast_mode=True, verbose=False)
        config = api.get_net_config(archi_num, image_dataset)
        archi_info = api.get_more_info(archi_num, image_dataset, hp='200', is_random=False)
        # get the info of architecture of the 6111-th model on CIFAR-10
        net = get_cell_based_tiny_net(config)
        arch = api.arch(archi_num) 

        # Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
        params = api.get_net_param(archi_num, image_dataset, None, hp ='200')
    elif api_type =='sss':
        api = create(sss_dir, api_type, fast_mode=True, verbose=False)
        
        config = api.get_net_config(archi_num, image_dataset)
        archi_info = api.get_more_info(archi_num, image_dataset, hp='90', is_random=False)
        # get the info of architecture of the 6111-th model on CIFAR-10
        net = get_cell_based_tiny_net(config)
        arch = api.arch(archi_num) 

        # Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
        params = api.get_net_param(archi_num, image_dataset, None, hp ='90')
    else:
        raise ValueError('api_type must be either tss or sss')

    net.load_state_dict(next(iter(params.values())))
    net = net.to(device)

    input = torch.randn(1, 3, 32, 32).to(device)
    macs, flops = profile(net, inputs=(input, ))

    df = pd.DataFrame({"archi_num": [archi_num], "flops": [flops]})

    # If the CSV file already exists, load it and append the new data
    if os.path.isfile(csv_file):
        df_existing = pd.read_csv(csv_file)
        df = pd.concat([df_existing, df])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)