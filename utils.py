import random
import time
import torch


def set_seeds(SEED=1234):
    """
    Set seeds for deterministic results.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    """
    Returns time taken by model to run an epoch.
    Params:
        start_time: starting time of epoch
        end_time: ending time of epoch
    Returns:
        elapsed_mins (int): minutes taken by epoch
        elapsed_secs (int): seconds taken by epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    Params:
        preds (torch.tensor): predictions made by model
        y (torch.tensor): ground truth labels
    Returns:
        acc (float): accuracy of the model (correct predictions / total predictions)
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc
