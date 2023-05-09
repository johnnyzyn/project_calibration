import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error
from scipy.special import softmax
from KDEpy import FFTKDE
from netcal.metrics import MMCE as MMCE
import calibration.utils as cal
from typing import List, Tuple, TypeVar

# from netcal.metrics import PICP as PICP
# from netcal.metrics import PinballLoss as PinballLoss
# from netcal.metrics import UCE as UCE
# from netcal.metrics import ENCE as ENCE
# from netcal.metrics import QCE as QCE

def get_ece(preds, targets, n_bins=15, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)
    
    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers,  
    return ece

def get_sce(preds, targets, n_bins=15, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    n_objects, n_classes = preds.shape
    res = 0.0
    for cur_class in range(n_classes):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            cur_class_conf = preds[:, cur_class]
            in_bin = np.logical_and(cur_class_conf > bin_lower, cur_class_conf <= bin_upper)

            # cur_class_acc is ground truth probability of chosen class being the correct one inside the bin.
            # NOT fraction of correct predictions in the bin
            # because it is compared with predicted probability
            bin_acc = (targets[in_bin] == cur_class)
            
            bin_conf = cur_class_conf[in_bin]

            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                avg_confidence_in_bin = np.mean(bin_conf)
                avg_accuracy_in_bin = np.mean(bin_acc)
                delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#                 print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
                res += delta * bin_size / (n_objects * n_classes)
    return res

def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    try:
        n_objects, n_classes = preds.shape
        res = 0.0
        for cur_class in range(n_classes):
            cur_class_conf = preds[:, cur_class]
            
            targets_sorted = targets[cur_class_conf.argsort()]
            cur_class_conf_sorted = np.sort(cur_class_conf)
            
            targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
            cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

            
            bin_size = len(cur_class_conf_sorted) // n_bins

                    
            for bin_i in range(n_bins):
                bin_start_ind = bin_i * bin_size
                if bin_i < n_bins-1:
                    bin_end_ind = bin_start_ind + bin_size
                else:
                    bin_end_ind = len(targets_sorted)
                    bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
                bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class)
                bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
                avg_confidence_in_bin = np.mean(bin_conf)
                avg_accuracy_in_bin = np.mean(bin_acc)
                delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
                # print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
                res += delta * bin_size / (n_objects * n_classes)
                
        return res
    except:
        return 0

def get_ace(preds, targets, n_bins=15, **args):
    return get_tace(preds, targets, n_bins, threshold=0)


def to_one_hot(targets, num_classes):
    one_hot_targets = np.zeros((targets.shape[0], num_classes))
    one_hot_targets[np.arange(targets.shape[0]), targets] = 1
    return one_hot_targets

def get_brierscore(preds, targets):
    num_classes = preds.shape[1]
    targets_one_hot = to_one_hot(targets, num_classes)
    brierscore = mean_squared_error(targets_one_hot, preds)
    return brierscore

def get_nll(preds, targets):
    num_classes = preds.shape[1]
    targets_one_hot = to_one_hot(targets, num_classes)
    nll = log_loss(targets_one_hot, preds)
    return nll

def get_mce(preds, targets, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)

    max_abs_delta = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)

        if np.any(in_bin):
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin

            if np.abs(delta) > max_abs_delta:
                max_abs_delta = np.abs(delta)

    return max_abs_delta


def get_classwise_ece(preds, targets, n_bins=15):
    if not np.array_equal(preds.shape, targets.shape):
        targets = label_binarize(np.array(targets), classes=range(preds.shape[1]))

    n_classes = preds.shape[1]
    classwise_ece = []

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for c in range(n_classes):
        confidences = preds[:, c]
        true_labels = targets[:, c].astype(float)

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(true_labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                delta = avg_confidence_in_bin - accuracy_in_bin
                ece += np.abs(delta) * prop_in_bin

        classwise_ece.append(ece)

    return np.mean(classwise_ece)

def get_KSCE(preds, targets):
    # KS stands for Kolmogorov-Smirnov

    def ensure_numpy(a):
        if not isinstance(a, np.ndarray): a = a.numpy()
        return a

    def get_top_results(preds, targets, nn=-1, inclusive=False, return_topn_classid=False) :

        #  nn should be negative, -1 means top, -2 means second top, etc
        # Get the position of the n-th largest value in each row
        topn = [np.argpartition(score, nn)[nn] for score in preds]
        nthscore = [score[n] for score, n in zip(preds, topn)]
        labs = [1.0 if int(target) == int(n) else 0.0 for target, n in zip(targets, topn)]

        # Change to tensor
        tscores = np.array(nthscore)
        tacc = np.array(labs)

        if return_topn_classid:
            return tscores, tacc, topn
        else:
            return tscores, tacc

    try:
        scores, targets = get_top_results(preds, targets)

        # Change to numpy, then this will work
        scores = ensure_numpy(scores)
        targets = ensure_numpy(targets)

        # Sort the data
        order = scores.argsort()
        scores = scores[order]
        targets = targets[order]

        # Accumulate and normalize by dividing by num samples
        n = scores.shape[0]
        integrated_scores = np.cumsum(scores) / n
        integrated_accuracy = np.cumsum(targets) / n

        # Work out the Kolmogorov-Smirnov error
        KS_error_max = np.amax(np.abs(integrated_scores - integrated_accuracy))

        return KS_error_max
    except:
        return 0



def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return np.concatenate((d, 2*xmax-d))
    else:
        return 
    

def get_KDECE(preds, labels, p_int=None, order=1):
    try:
        p = preds  # No need to calculate softmax

        # to 1-hot
        label = label_binarize(np.array(labels), classes=range(p.shape[1]))

        # points from numerical integration
        if p_int is None:
            p_int = np.copy(p)

        p = np.clip(p,1e-256,1-1e-256)
        p_int = np.clip(p_int,1e-256,1-1e-256)


        x_int = np.linspace(-0.6, 1.6, num=2**14)


        N = p.shape[0]

        # this is needed to convert labels from one-hot to conventional form
        label_index = np.array([np.where(r==1)[0][0] for r in label])
        with torch.no_grad():
            if p.shape[1] !=2:
                p_new = torch.from_numpy(p)
                p_b = torch.zeros(N,1)
                label_binary = np.zeros((N,1))
                for i in range(N):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    if pred_label == label_index[i]:
                        label_binary[i] = 1
                    p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])
            else:
                p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
                label_binary = label_index

        method = 'triweight'

        dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
        kbw = abs(np.std(p_b.numpy())*(N*2)**-0.2)
        kbw = np.std(dconf_1)*(N*2)**-0.2

        # Mirror the data about the domain boundary
        low_bound = 0.0
        up_bound = 1.0
        dconf_1m = mirror_1d(dconf_1,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
        pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp1 = pp1 * 2  # Double the y-values to get integral of ~1


        p_int = p_int/np.sum(p_int,1)[:,None]
        N1 = p_int.shape[0]
        with torch.no_grad():
            p_new = torch.from_numpy(p_int)
            pred_b_int = np.zeros((N1,1))
            if p_int.shape[1]!=2:
                for i in range(N1):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    pred_b_int[i] = p_int[i,pred_label]
            else:
                for i in range(N1):
                    pred_b_int[i] = p_int[i,1]

        low_bound = 0.0
        up_bound = 1.0
        pred_b_intm = mirror_1d(pred_b_int,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
        pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp2 = pp2 * 2  # Double the y-values to get integral of ~1


        if p.shape[1] !=2: # top label (confidence)
            perc = np.mean(label_binary)
        else: # or joint calibration for binary cases
            perc = np.mean(label_index)

        integral = np.zeros(x_int.shape)
        reliability= np.zeros(x_int.shape)
        for i in range(x_int.shape[0]):
            conf = x_int[i]
            if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
                accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
                if np.isnan(accu)==False:
                    integral[i] = np.abs(conf-accu)**order*pp2[i]
                    reliability[i] = accu
            else:
                if i>1:
                    integral[i] = integral[i-1]

        ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
        return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])
    except:
        return 0

def get_MMCE(preds, targets):
    try:
        get_MMCE = MMCE()
        return get_MMCE.measure(preds, targets)
    except:
        return 0

# def get_PinballLoss(X, y, q=0.5, kind='cumulative', reduction='mean'):
#     pinball_loss = PinballLoss()
#     return pinball_loss.measure(X, y, q, kind=kind, reduction=reduction)


# def get_PICP(preds, targets,bins  = 15):
#     picp = PICP(bins)
#     return picp.measure(preds, targets,q=0.5)



# def UCE(logits, labels):
#     uce = UCE()
#     scores = softmax(logits, axis=1)
#     return uce.measure(scores, labels)

# def ENCE(logits, labels):
#     ence = ENCE()
#     scores = softmax(logits, axis=1)
#     return ence.measure(scores, labels)

# def QCE(logits, labels, quantile=0.5):
#     qce = QCE(quantile)
#     scores = softmax(logits, axis=1)
#     return qce.measure(scores, labels)

Data = List[Tuple[float, float]]  # List of (predicted_probability, true_label).
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0.
BinnedData = List[Data]  # binned_data[i] contains the data in bin i.
T = TypeVar('T')
def split(sequence: List[T], parts: int) -> List[List[T]]:
    assert parts <= len(sequence)
    part_size = int(np.ceil(len(sequence) * 1.0 / parts))
    assert part_size * parts >= len(sequence)
    assert (part_size - 1) * parts < len(sequence)
    return [sequence[i:i + part_size] for i in range(0, len(sequence), part_size)]

def equal_width_bins(probs: List[float], num_bins: int=10) -> Bins:
    return [i * 1.0 / num_bins for i in range(1, num_bins + 1)]

def equal_mass_bins(probs: List[float], num_bins: int=10) -> Bins:
    """Get bins that contain approximately an equal number of data points."""
    sorted_probs = sorted(probs)
    binned_data = split(sorted_probs, num_bins)
    bins: Bins = []
    for i in range(len(binned_data) - 1):
        last_prob = binned_data[i][-1]
        next_first_prob = binned_data[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    bins = sorted(list(set(bins)))
    return bins


def CWCE(preds, labels, bins=15):
    return cal.lower_bound_scaling_ce(
        preds,
        labels.flatten(),
        p=1,
        num_bins=bins,
        debias=False,
        mode='marginal',
        binning_scheme=equal_width_bins)


def dEMsTCE(logits, labels, bins=15):
    return cal.lower_bound_scaling_ce(
        logits,
        labels.flatten(),
        p=1,
        num_bins=bins,
        debias=True,
        mode='top-label',
        binning_scheme=equal_mass_bins)


def sTCE(logits, labels, bins=15):
    return cal.lower_bound_scaling_ce(
        logits,
        labels.flatten(),
        num_bins=bins,
        debias=False,
        mode='top-label',
        binning_scheme=equal_width_bins)
