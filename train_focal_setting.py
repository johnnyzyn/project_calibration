import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.special import softmax
import datetime
import argparse
from torch.utils.data import SubsetRandomSampler
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
class MMCE(nn.Module):
    def __init__(self, lamda=2):
        super(nn.Module, self).__init__()
        self.mmce_weighted = MMCE_weighted()
        self.ce = nn.CrossEntropyLoss()
        self.lamda = lamda
    def forward(self, input, target):
        return self.lamda*self.mmce_weighted(input, target) + self.ce(input, target)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=3, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class MMCE_weighted(nn.Module):
    """
    Computes MMCE_w loss.
    """
    def __init__(self, device):
        super(MMCE_weighted, self).__init__()
        self.device = device

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def get_pairs(self, tensor1, tensor2):
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
                                    dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
                                    dim=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.mean(tensor1*tensor2)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

        correct_mask = torch.where(torch.eq(predicted_labels, target),
                                    torch.ones(predicted_labels.shape).to(self.device),
                                    torch.zeros(predicted_labels.shape).to(self.device))

        k = torch.sum(correct_mask).type(torch.int64)
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
        cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        k = torch.max(k, torch.tensor(1).to(self.device))*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
        k_p = torch.max(k_p, torch.tensor(1).to(self.device))*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                            (correct_mask.shape[0] - 2))


        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

        correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)  

        sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

        correct_correct_vals = self.get_out_tensor(correct_kernel,
                                                          sampling_weights_correct)
        sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

        incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
                                                          sampling_weights_incorrect)
        sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

        correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
                                                          sampling_correct_incorrect)

        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)

        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals) 
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)
        return torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))

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
    # 1, 128, 256, 512, 1024, 2048, 4096, 8192, 6111 15000, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
    parser.add_argument("--archi_num", type=int, default=6111, help="Key number of architecture")
    parser.add_argument("--image_dataset", type=str, default="cifar10", help="CIFAR-10, CIFAR-100, and ImageNet16-120")
    parser.add_argument("--api_type", type=str, default="tss", help="NATS-Bench dataset type (tss or sss)")
    parser.add_argument("--post_temp", type=str, default='False', help="if using temp scale")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")
    parser.add_argument("--loss", type=str)

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
    tss_simple_dir = "/hdd/datasets/NATSBench/NATS-sss-v1_0-50262-simple"

    
    if api_type =='tss':
        api = create(tss_simple_dir, api_type, fast_mode=True, verbose=False)
        config = api.get_net_config(archi_num, image_dataset)
        net = get_cell_based_tiny_net(config)
        arch = api.arch(archi_num) 
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

    # net.load_state_dict(next(iter(params.values())))
    net.to(device)

    # net.load_state_dict(params[777])
    if image_dataset == 'cifar10':
        train_loader,_ = cifar10.get_train_valid_loader(batch_size = 256,
                                random_seed = 42,
                                valid_size=0,
                                shuffle=True,
                                num_workers=4, pin_memory=False)
        
        if post_temp == 'True':
            test_loader, val_loader = cifar10.get_test_valid_loader(batch_size = 256,
                                random_seed = 42,
                                valid_size=0.2,
                                shuffle=True,
                                num_workers=4, pin_memory=False)
        else:
            test_loader = cifar10.get_test_loader(batch_size=256, shuffle=False, num_workers=4, pin_memory=False)
    elif image_dataset == 'cifar100':
        if post_temp == 'True':
            test_loader, val_loader = cifar100.get_test_valid_loader(batch_size = 256,
                                random_seed = 42,
                                valid_size=0.2,
                                shuffle=True,
                                num_workers=4, pin_memory=False)
        else:
            test_loader = cifar100.get_test_loader(batch_size=256, shuffle=False, num_workers=4, pin_memory=False)
    elif image_dataset == 'ImageNet16-120':
        

        root = './datasets/ImagenNet16'
        train_data, test_data, xshape, class_num = get_datasets(image_dataset, root, 0)

        if post_temp == 'True':
            def imagenet_get_test_valid_loader(batch_size = 256, random_seed= 42, valid_size = 0.2, shuffle = True,
                                        num_workers=4, pin_memory=False,
                                    test_dataset=test_data):
                num_test = len(test_dataset)
                indices = list(range(num_test))
                split = int(np.floor(valid_size * num_test))

                if shuffle:
                    np.random.seed(random_seed)
                    np.random.shuffle(indices)

                test_idx, valid_idx = indices[split:], indices[:split]
                

                test_sampler = SubsetRandomSampler(test_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)

                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, sampler=test_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                valid_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                return test_loader, valid_loader
            test_loader, val_loader = imagenet_get_test_valid_loader(batch_size = 256, random_seed= 42, valid_size = 0.2, shuffle = True,
                                    num_workers=4, pin_memory=False)
        else:
            test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


    


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


# {
#   "scheduler": ["str",   "cos"],
#   "eta_min"  : ["float", "0.0"],
#   "epochs"   : ["int",   "200"],
#   "warmup"   : ["int",   "0"],
#   "optim"    : ["str",   "SGD"],
#   "LR"       : ["float", "0.1"],
#   "decay"    : ["float", "0.0005"],
#   "momentum" : ["float", "0.9"],
#   "nesterov" : ["bool",  "1"],
#   "criterion": ["str",   "Softmax"],
#   "batch_size": ["int", "256"]
# }
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=0.0, last_epoch=-1
    )
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'mmce':
        criterion = MMCE()

    for epoch in range(200):
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        # evaluate on test set
        preds, pred_classes, targets = get_preds_and_targets(net, test_loader, device)

        ece = get_param_dict(metric.get_ece, preds, targets, n_bins=15)
        accuracy = 100. * (pred_classes == targets).mean()

        # print performance every epoch and current time
        print("Time: {}  Epoch: {} | ECE: {:.3f} | Accuracy: {:.3f} | LR: {:.5f}".format(
            datetime.datetime.now(), 
            epoch, 
            ece['result'], 
            accuracy, 
            optimizer.param_groups[0]['lr']))

# sudo scp -r linwei@10.165.232.195:/media/linwei/disk1/NATS-Bench/NATS-sss-v1_0-50262-simple /hdd/datasets/NATSBench

