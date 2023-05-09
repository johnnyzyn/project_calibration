import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
# from scipy.special import softmax
import datetime
import argparse

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

    if image_dataset == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size=100, shuffle=False, num_workers=4, pin_memory=False)
        if post_temp == 'True':
            train_loader, val_loader = cifar10.get_train_valid_loader(batch_size = 100,augment = False,
                            random_seed=42,
                            valid_size=0.2,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=False,
                            get_val_temp=0)
        
    elif image_dataset == 'cifar100':
        test_loader = cifar100.get_test_loader(batch_size=100, shuffle=False, num_workers=4, pin_memory=False)
    elif image_dataset == 'ImageNet16-120':
        root = '/home/younan/project_calibration/project_calibration/datasets/ImagenNet16'
        train_data, test_data, xshape, class_num = get_datasets(image_dataset, root, 0)

        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


    

    # preds, pred_classes,targets = get_preds_and_targets(net, test_loader, device)
    if post_temp == 'True':
        val_probs, val_pred_classes, val_targets = get_preds_and_targets(net, val_loader, device)
        test_probs, test_pred_classes, test_targets = get_preds_and_targets(net, test_loader, device)

        scaled_model = ModelWithTemperature(net)
        scaled_model.set_temperature(val_loader,device=device)

        preds, pred_classes,targets  = get_preds_and_targets2(scaled_model, test_loader, device)
    else:
        preds, pred_classes,targets = get_preds_and_targets(net, test_loader, device)

    # tensor_preds = torch.tensor(preds).to(device)
    # tensor_targets = torch.tensor(targets).to(device)
        
    # bandwidth = ece_kde.get_bandwidth(tensor_preds,device)

    # print('ECE_KDE:', ece_kde.get_ece_kde(tensor_preds, tensor_targets, bandwidth = bandwidth, p = 1, mc_type = 'canonical', device = device).item())

    
    # Initialize lists for metrics that require n_bins parameter
    # Initialize empty strings for metrics that require n_bins parameter
    ece_str = ''
    sce_str = ''
    tace_str = ''
    ace_str = ''
    mce_str = ''
    cwECE_str = ''
    ECE_em_str = ''
    ole_str = ''
    ole_loss = tace.OELoss()
    
    for n_bin in bin_sizes:
        ece_str += str(get_param_dict(metric.get_ece, preds, targets, n_bins=n_bin)) + ', '
        sce_str += str(get_param_dict(metric.get_sce,preds, targets, n_bins=n_bin,logits = False)) + ', '
        tace_str += str(get_param_dict(metric.get_tace,preds, targets, n_bins=n_bin,logits = False)) + ', '
        ace_str += str(get_param_dict(metric.get_ace,preds, targets, n_bins=n_bin,logits = False)) + ', '
        mce_str += str(get_param_dict(metric.get_mce, preds, targets, n_bins=n_bin)) + ', '
        cwECE_str += str(get_param_dict(metric.get_classwise_ece, preds, targets, n_bins=n_bin)) + ', '
        ECE_em_str += str(get_param_dict(cal.get_ece_em, preds, targets, num_bins=n_bin)) + ', '
        ole_str = str(get_param_dict(ole_loss.loss, preds, targets, n_bins=n_bin,logits = False)) + ', '

    # Remove the trailing comma and space
    ece_str = ece_str.rstrip(', ')
    sce_str = sce_str.rstrip(', ')
    tace_str = tace_str.rstrip(', ')
    ace_str = ace_str.rstrip(', ')
    mce_str = mce_str.rstrip(', ')
    cwECE_str = cwECE_str.rstrip(', ')
    ECE_em_str = ECE_em_str.rstrip(', ')
    ole_str = ole_str.rstrip(', ')

    data = {
        'config': [archi_num],
        'info' : [accuracy(preds,targets)],
        'dataset': [image_dataset],
        'arch': [arch],
        'ece': ece_str,
        'sce': sce_str,
        'tace': tace_str,
        'ace': ace_str,
        'MCE': mce_str,
        'cwECE': cwECE_str,
        'Marginal_CE_debias': [get_param_dict(cal.get_calibration_error,preds, targets,p=1)],
        'Marginal_CE': [get_param_dict(cal.get_calibration_error,preds, targets, debias=False,p=1)],
        'ECE_em': ECE_em_str,
        'Ole': ole_str,
        'KSCE': [get_param_dict(metric.get_KSCE,preds, targets)],
        'KDECE': [get_param_dict(metric.get_KDECE,preds, targets)],
        'MMCE': [get_param_dict(metric.get_MMCE,preds, targets)],
        'NLL': [get_param_dict(metric.get_nll,preds, targets)],
        'brier': [get_param_dict(metric.get_brierscore,preds, targets)],
        # 'ECE_KDE': [get_param_dict(ece_kde.get_ece_kde(tensor_preds, tensor_targets, bandwidth=bandwidth, p=1, mc_type='canonical', device=device).item())],
        'timestamp': [datetime.datetime.now()]
    }

    # print(data)
    # Step 2: Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)
    print(df.head(1))


    if os.path.exists(csv_file):
        # Load existing data
        df_existing = pd.read_csv(csv_file)

        # Append the new data
        df_existing = df_existing.append(df, ignore_index=True)

        # Save the updated DataFrame to the CSV file
        df_existing.to_csv(csv_file, index=False)
    else:
        # Save the new DataFrame to a new CSV file
        df.to_csv(csv_file, index=False)