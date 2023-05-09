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
import calibration as cal
import calibration.metric as metric
# import calibration.ece_kde as ece_kde
import inspect

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
    parser.add_argument("--n_bins", type=int, default=15, help="Number of bins for calibration metrics")
    parser.add_argument("--archi_num", type=int, default=6111, help="Key number of architecture")
    parser.add_argument("--image_dataset", type=str, default="cifar10", help="CIFAR-10, CIFAR-100, and ImageNet16-120")
    parser.add_argument("--nats_bench_path", type=str, default="/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", help="Path to the NATS-Bench dataset")
    parser.add_argument("--api_type", type=str, default="tss", help="NATS-Bench dataset type (tss or sss)")

    args = parser.parse_args()

    # Use the parsed arguments
    n_bins = args.n_bins
    archi_num = args.archi_num
    csv_file = args.csv_file
    image_dataset = args.image_dataset
    nats_bench_path = args.nats_bench_path
    api_type = args.api_type

    api = create(nats_bench_path, api_type, fast_mode=True, verbose=False)
    # api = create("/media/linwei/disk1/NATS-Bench/NATS-sss-full/sss-full", 'sss', fast_mode=True, verbose=False)
    config = api.get_net_config(archi_num, image_dataset)
    # get the info of architecture of the 6111-th model on CIFAR-10
    net = get_cell_based_tiny_net(config)
    arch = api.arch(archi_num) 

    # Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
    params = api.get_net_param(archi_num, image_dataset, None)
    net.load_state_dict(next(iter(params.values())))

    test_loader = cifar10.get_test_loader(batch_size=100, shuffle=False, num_workers=2, pin_memory=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preds, pred_classes,targets = get_preds_and_targets(net, test_loader, device)

    # tensor_preds = torch.tensor(preds).to(device)
    # tensor_targets = torch.tensor(targets).to(device)
        
    # bandwidth = ece_kde.get_bandwidth(tensor_preds,device)

    print('ece:', metric.get_ece(preds, targets))
    print('sce:', metric.get_sce(preds, targets))
    print('tace:', metric.get_tace(preds, targets))
    print('ace:', metric.get_ace(preds, targets))

    print('MCE:', metric.get_mce(preds, targets))
    print('cwECE:', metric.get_classwise_ece(preds, targets))

    print('TCE_debias:', cal.get_top_calibration_error(preds, targets))
    print('Marginal_CE_debias:', cal.get_calibration_error(preds, targets))
    print('TCE:', cal.get_top_calibration_error(preds, targets,debias=False))
    print('Marginal_CE:', cal.get_calibration_error(preds, targets,debias=False))
    print('ECE_em:', cal.get_ece_em(preds, targets))

    print('KSCE:', metric.get_KSCE(preds, targets))
    print('KDECE:', metric.get_KDECE(preds, targets))
    print('MMCE:', metric.get_MMCE(preds, targets))
    print('NLL:', metric.get_nll(preds, targets))
    print('brier:', metric.get_brierscore(preds, targets))
    # print('ECE_KDE:', ece_kde.get_ece_kde(tensor_preds, tensor_targets, bandwidth = bandwidth, p = 1, mc_type = 'canonical', device = device).item())

    data = {
        'config': [6111],
        'dataset': ["cifar10"],
        'arch': [arch],
        'ece': [get_param_dict(metric.get_ece,preds, targets,n_bins)],
        'sce': [get_param_dict(metric.get_sce,preds, targets,n_bins)],
        'tace': [get_param_dict(metric.get_tace,preds, targets,n_bins)],
        'ace': [get_param_dict(metric.get_ace,preds, targets,n_bins)],
        'MCE': [get_param_dict(metric.get_mce,preds, targets,n_bins)],
        'cwECE': [get_param_dict(metric.get_classwise_ece,preds, targets,n_bins)],
        'TCE_debias': [get_param_dict(cal.get_top_calibration_error,preds, targets)],
        'Marginal_CE_debias': [get_param_dict(cal.get_calibration_error,preds, targets)],
        'TCE': [get_param_dict(cal.get_top_calibration_error,preds, targets, debias=False)],
        'Marginal_CE': [get_param_dict(cal.get_calibration_error,preds, targets, debias=False)],
        'ECE_em': [get_param_dict(cal.get_ece_em,preds, targets,n_bins)],
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
    print(df.head(5))

    csv_file = 'results.csv'

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


import torch
import os
import pandas as pd
import numpy as np
from nats_bench import create
import xautodl
from xautodl.models import get_cell_based_tiny_net
from datatest import get_logits_labels, get_valid_test_loader, ECELoss, AdaptiveECELoss, ClasswiseECELoss, \
    get_logits_labels2
from temprature import ModelWithTemperature
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#cifar10>935
#a1 = [1, 42, 69, 81, 99, 111, 129, 138, 168, 203, 244, 250, 258, 265, 295, 322, 352, 378, 439, 541, 543, 571, 582, 587, 654, 655, 684, 690, 699, 722, 844, 857, 869, 873, 937, 1000, 1027, 1062, 1114, 1177, 1317, 1414, 1425, 1459, 1462, 1556, 1583, 1617, 1654, 1656, 1669, 1716, 1842, 1856, 1906, 1941, 1963, 1965, 2013, 2018, 2030, 2053, 2066, 2159, 2162, 2210, 2225, 2246, 2254, 2472, 2508, 2510, 2528, 2538, 2566, 2580, 2581, 2676, 2704, 2725, 2772, 2776, 2778, 3035, 3045, 3166, 3255, 3383, 3385, 3456, 3540, 3558, 3621, 3625, 3731, 3803, 3888, 3907, 3927, 3928, 3982, 3986, 4062, 4099, 4141, 4181, 4208, 4237, 4244, 4361, 4385, 4418, 4511, 4533, 4557, 4562, 4641, 4681, 4747, 4800, 4902, 4907, 4926, 4946, 4954, 4983, 5087, 5095, 5111, 5115, 5176, 5256, 5292, 5369, 5374, 5525, 5539, 5544, 5664, 5698, 5711, 5721, 5731, 5749, 5776, 5780, 5801, 5823, 5883, 5902, 5979, 6016, 6022, 6111, 6118, 6119, 6304, 6330, 6489, 6499, 6515, 6530, 6584, 6640, 6676, 6697, 6747, 6804, 6830, 6931, 6993, 6999, 7209, 7313, 7353, 7384, 7407, 7419, 7463, 7482, 7548, 7556, 7570, 7602, 7619, 7632, 7725, 7897, 7906, 7963, 8002, 8058, 8114, 8135, 8159, 8162, 8216, 8253, 8285, 8288, 8290, 8401, 8413, 8426, 8449, 8452, 8529, 8572, 8592, 8606, 8610, 8621, 8649, 8654, 8695, 8708, 8733, 8736, 8845, 8855, 8860, 8877, 8884, 9016, 9046, 9099, 9121, 9149]
a1 = [9161, 9302, 9431, 9493, 9572, 9586, 9612, 9640, 9648, 9656, 9718, 9722, 9743, 9819, 9916, 9923, 9930, 9941, 9950, 9972, 9997, 10000, 10007, 10063, 10073, 10086, 10131, 10154, 10165, 10188, 10189, 10192, 10200, 10279, 10335, 10337, 10360, 10370, 10439, 10476, 10484, 10545, 10603, 10676, 10699, 10715, 10721, 10733, 10749, 10750, 10767, 10835, 10838, 10872, 10902, 10935, 10948, 10975, 11145, 11146, 11157, 11174, 11217, 11218, 11226, 11232, 11270, 11271, 11298, 11299, 11307, 11351, 11372, 11376, 11400, 11443, 11465, 11472, 11564, 11584, 11692, 11711, 11722, 11737, 11754, 11777, 11796, 11797, 11876, 12002, 12043, 12123, 12158, 12295, 12384, 12399, 12415, 12421, 12451, 12464, 12466, 12476, 12491, 12574, 12576, 12606, 12608, 12617, 12645, 12663, 12687, 12692, 12761, 12799, 12839, 12956, 12967, 13032, 13112, 13127, 13166, 13171, 13325, 13363, 13376, 13390, 13402, 13417, 13451, 13539, 13551, 13585, 13644, 13652, 13714, 13755, 13778, 13801, 13809, 13869, 13925, 13981, 13987, 13991, 14029, 14118, 14139, 14152, 14174, 14203, 14029, 14118, 14139, 14152, 14174, 14203, 14218, 14228, 14296, 14375, 14443, 14462, 14469, 14572, 14580, 14638, 14654, 14669, 14709, 14714, 14720, 14805, 14861, 14866, 14985, 15028, 15144, 15159, 15283, 15341, 15383, 15425, 15461, 15485, 15508, 15537, 15567, 15586, 15615]
#cifar100>71
a2 = [69, 81, 168, 265, 543, 582, 655, 690, 699, 722, 857, 862, 869, 937, 1317, 1346, 1425, 1459, 1462, 1556, 1583, 1656, 2018, 2030, 2053, 2159, 2210, 2508, 2538, 2554, 2566, 2580, 2581, 2725, 2772, 2778, 2889, 3035, 3192, 3383, 3385, 3456, 3540, 3621, 3731, 3888, 3928, 3982, 4062, 4141, 4208, 4237, 4244, 4361, 4418, 4557, 4562, 4747, 4907, 4926, 4954, 5111, 5115, 5207, 5292, 5374, 5711, 5749, 5979, 6111, 6118, 6119, 6304, 6530, 6830, 6925, 6931, 7209, 7294, 7337, 7419, 7548, 7556, 7570, 7602, 7725, 7897, 7906, 8002, 8004, 8102, 8114, 8135, 8162, 8216, 8290, 8449, 8452, 8529, 8572, 8610, 8621, 8625, 8695, 8768, 8853, 8860, 9149, 9431, 9586, 9635, 9648, 9718, 9930, 9972, 10000, 10007, 10154, 10165, 10189, 10315, 10321, 10337, 10360, 10400, 10436, 10439, 10476, 10603, 10676, 10699, 10715, 10721, 10749, 10770, 10835, 10902, 10948, 10957, 11000, 11298, 11307, 11400, 11465, 11584, 11692, 11711, 11722, 11754, 11797, 11876, 11915, 12076, 12201, 12295, 12384, 12399, 12415, 12421, 12574, 12576, 12687, 12761, 12799, 12887, 12919, 12967, 13363, 13390, 13403, 13417, 13539, 13551, 13644, 13714, 13778, 13925, 13981, 13991, 14174, 14296, 14375, 14443, 14462, 14572, 14624, 14654, 14714, 14805, 14861, 14866, 15190, 15383, 15425, 15485, 15510]
#ImageNet>45
#a3 = [65, 81, 109, 129, 168, 203, 265, 352, 374, 435, 439, 542, 571, 582, 648, 655, 699, 857, 862, 869, 915, 1013, 1027, 1084, 1237, 1317, 1414, 1425, 1459, 1555, 1556, 1716, 1742, 1939, 2018, 2030, 2053, 2159, 2162, 2210, 2212, 2213, 2217, 2246, 2272, 2387, 2508, 2538, 2554, 2581, 2700, 2720, 2743, 2778, 2807, 2889, 2934, 3192, 3275, 3385, 3424, 3456, 3458, 3540, 3558, 3580, 3621, 3697, 3731, 3738, 3771, 3818, 3888, 3928, 3934, 3982, 4013, 4062, 4133, 4135, 4208, 4212, 4237, 4244, 4275, 4361, 4418, 4424, 4496, 4557, 4562, 4681, 4707, 4747, 4902, 4926, 4954, 4986, 5063, 5111, 5176, 5292, 5301, 5374, 5584, 5633, 5711, 5721, 5749, 6111, 6118, 6225, 6231, 6304, 6375, 6459, 6530, 6585, 6640, 6676, 6687, 6932, 6982, 7129, 7176, 7293, 7337, 7408, 7430, 7442, 7463, 7548, 7602, 7826, 7855, 7901, 7906, 7926, 8002, 8058, 8102, 8114, 8135, 8162, 8290, 8449, 8452, 8572, 8621, 8695, 8733, 8736, 8768, 8827]
a3 = [ 8853, 8860, 9080, 9106, 9149, 9274, 9372, 9493, 9586, 9596, 9608, 9836, 9902, 9903, 9923, 9930, 9972, 10000, 10007, 10066, 10087, 10154, 10165, 10189, 10315, 10324, 10337, 10360, 10400, 10436, 10439, 10593, 10603, 10676, 10699, 10715, 10721, 10733, 10749, 10793, 10838, 10902, 10935, 10948, 11060, 11145, 11298, 11307, 11400, 11564, 11584, 11711, 11777, 11797, 11876, 11882, 11887, 11915, 12015, 12068, 12080, 12123, 12156, 12201, 12384, 12411, 12415, 12466, 12476, 12572, 12574, 12606, 12677, 12687, 12702, 12724, 12761, 12887, 12922, 12967, 13111, 13127, 13171, 13390, 13417, 13532, 13583, 13644, 13714, 13717, 13778, 13814, 13904, 13925, 14016, 14029, 14039, 14118, 14140, 14174, 14217, 14375, 14417, 14443, 14538, 14640, 14720, 14840, 14861, 14866, 14872, 14985, 15089, 15159, 15166, 15190, 15257, 15383, 15446, 15508, 15510, 15524, 15559, 15599]

ids={
    'cifar10':a1,
    'cifar100':a2,
    'ImageNet16-120':a3
}

# Press the green button in the gutter to run the script.
#15625
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #sss-full
    #api = create(r"/media/linwei/disk1/NATS-Bench/NATS-sss-full/sss-full", 'sss', fast_mode=True, verbose=True)

    api = create(r"/media/linwei/disk1/NATS-Bench/NATS-tss-v1_0-3ffb9-full/NATS-tss-v1_0-3ffb9-full", 'sss', fast_mode=True, verbose=True)
    #print(len(api))
    # api = create(r"X:/NATS-Bench/NATS-sss-v1_0-50262-simple", 'sss', fast_mode=True, verbose=True)
    #bins = 15
    #for dset in ['cifar10', 'cifar100', 'ImageNet16-120']:
    for dset in ['cifar10', 'ImageNet16-120']:
    #for dset in ['ImageNet16-120']:
        valloader, testloader = get_valid_test_loader(dset, '/media/linwei/disk1/NATS-Bench/cifar.python', batch=256)
        a = ids[dset]
        idx_list = []
        acc_list = []
        ece_bef_list = []
        ece_aft_list = []
        aece_bef_list = []
        aece_aft_list = []
        cece_bef_list = []
        cece_aft_list = []
        #for idx in range(15625):
        #for idx in range(100):

        for idx in np.random.randint(0,15625,20):
            idx_list.append(idx)
            testacc = api.get_more_info(idx, dset, hp='200', is_random=False)["test-accuracy"]
            acc_list.append(testacc)

            config = api.get_net_config(idx, dset)
            network = get_cell_based_tiny_net(config)
            params = api.get_net_param(idx, dset, None, hp='200')
            network.load_state_dict(next(iter(params.values())))
            print(params.keys())

            with torch.no_grad():
                net1 = network.cuda(0)
                logits, labels = get_logits_labels(testloader, net1)
                x=[]
                y=[]
                z=[]
                for bins in [5,10,15,20]:
                    ece_criterion = ECELoss(n_bins=bins).cuda(0)
                    ece = ece_criterion(logits, labels).item()
                    x.append(ece)
                    aece_criterion = AdaptiveECELoss(n_bins=bins).cuda(0)
                    aece = aece_criterion(logits, labels).item()
                    y.append(aece)
                    cece_criterion = ClasswiseECELoss(n_bins=bins).cuda(0)
                    cece = cece_criterion(logits, labels).item()
                    z.append(cece)
                ece_bef_list.append(x)
                aece_bef_list.append(y)
                cece_bef_list.append(z)

            scaled_model = ModelWithTemperature(network)
            scaled_model.set_temperature(valloader)

            with torch.no_grad():
                net2 = scaled_model.cuda(0)
                logits, labels = get_logits_labels2(testloader, net2)
                p = []
                q = []
                r = []
                for bins in [5, 10, 15, 20]:
                    ece_criterion = ECELoss(n_bins=bins).cuda(0)
                    ece = ece_criterion(logits, labels).item()
                    p.append(ece)
                    aece_criterion = AdaptiveECELoss(n_bins=bins).cuda(0)
                    aece = aece_criterion(logits, labels).item()
                    q.append(aece)
                    cece_criterion = ClasswiseECELoss(n_bins=bins).cuda(0)
                    cece = cece_criterion(logits, labels).item()
                    r.append(cece)
                ece_aft_list.append(p)
                aece_aft_list.append(q)
                cece_aft_list.append(r)

        ece_bef_list = np.array(ece_bef_list)
        ece_aft_list = np.array(ece_aft_list)
        aece_bef_list = np.array(aece_bef_list)
        aece_aft_list = np.array(aece_aft_list)
        cece_bef_list = np.array(cece_bef_list)
        cece_aft_list = np.array(cece_aft_list)



        df = pd.DataFrame(zip(idx_list, acc_list, ece_bef_list[:,0], ece_aft_list[:,0], aece_bef_list[:,0], aece_aft_list[:,0], cece_bef_list[:,0], cece_aft_list[:,0],
                                ece_bef_list[:, 1], ece_aft_list[:, 1], aece_bef_list[:, 1], aece_aft_list[:, 1], cece_bef_list[:,1], cece_aft_list[:, 1],
                              ece_bef_list[:, 2], ece_aft_list[:, 2], aece_bef_list[:, 2], aece_aft_list[:, 2], cece_bef_list[:,2], cece_aft_list[:, 2],
                              ece_bef_list[:, 3], ece_aft_list[:, 3], aece_bef_list[:, 3], aece_aft_list[:, 3], cece_bef_list[:,3], cece_aft_list[:, 3],))

        df.to_csv('/media/linwei/disk1/NATS-Bench/'+dset+'-bins -part2.csv')