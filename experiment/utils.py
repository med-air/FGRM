import warnings
import cv2
import enum
import torch
import numpy as np
import pandas as pd
import torchio as tio
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class Task(enum.Enum):
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'
    REGRESSION = 'regression'
class Organ(enum.Enum):
    ESD = 'ESD'
    LC = 'LC'
    HEART = 'heart'



def normalise(data, nmax=1., nmin=0.):
    return (data-data.min()) * ((nmax - nmin) / (data.max() - data.min() + 1e-8)) + nmin

def enable_dropout(m):
    counter = 0
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()
            counter += 1
    return counter

# def dice_loss(logits, true, eps=1e-7):
#     """Computes the Sørensen–Dice loss.
#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.
#     Args:
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         true: a tensor of shape [B, 1, H, W].
#         eps: added to the denominator for numerical stability.
#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     num_classes = logits.shape[1]
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1).to(torch.int64)]
#         # to .contigious() to suppress channels_last mixed tensor memory format
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         # true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous().float()
#         probas = F.softmax(logits, dim=1)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     cardinality = torch.sum(probas + true_1_hot, dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()
#     return (1 - dice_loss)

def make_folders(base_path, experiment_name, configs):
    ''' Create experiment folder with subfolders figures, models, segmentations
    Arguments
        :pathlib.Path base_path: where the experiment folder path will be created
        :str experiment_name: all experiment related outputs will be here

    @returns a tuple of pathlib.Path objects
    (models_path, figures_path, seg_out_path)
    '''
    results_path = Path(base_path / experiment_name)
    figures_path = results_path / 'figures'
    models_path = results_path / 'models'
    metrics_out_path = results_path / 'metrics'
    if not results_path.exists():
        results_path.mkdir(parents=True)
    if not figures_path.exists():
        figures_path.mkdir()
    if not models_path.exists():
        models_path.mkdir()
    if not metrics_out_path.exists():
        metrics_out_path.mkdir()

    return models_path, figures_path, metrics_out_path


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
