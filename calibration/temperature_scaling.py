'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece





class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, log=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log


    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)


    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature


    def set_temperature(self,
                        valid_loader,
                        cross_validate='ece', device=torch.device('cuda')):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # with torch.no_grad():
        #     for input, label in valid_loader:
        #         input = input
        #         label = label.to(device)  # Move labels to the same device as the model
        #         output = self.model(input)
        #         # logits = output[1]
        #         logits = output
        #         logits_list.append(logits)  # Assuming logits is a tuple and you need the first element
        #         labels_list.append(label)
        #     logits = torch.cat(logits_list).to(device)
        #     labels = torch.cat(labels_list).to(device)
        # print("Logits shape:", logits.shape)
        # print("Labels shape:", labels.shape)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.to(device)
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        self.to(device)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % self.temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


    def get_temperature(self):
        return self.temperature