import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

def get_preds_and_targets(model, dataloader, device):
    preds, pred_classes, targets = [], [], []

    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to the selected device (CPU or GPU)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output_tuple = model(data)

            output = output_tuple[1]

            prob = F.softmax(output, dim=1)  # Compute probabilities
            _, pred = torch.max(prob, 1)  # Get predicted class

            preds.extend(prob.cpu().numpy())  # Move probabilities to CPU and convert to numpy array
            pred_classes.extend(pred.cpu().numpy())  # Move predictions to CPU and convert to numpy array
            targets.extend(target.cpu().numpy())  # Move targets to CPU and convert to numpy array

    return np.array(preds), np.array(pred_classes), np.array(targets)


import inspect
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