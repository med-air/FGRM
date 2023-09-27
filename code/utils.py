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


class Task(enum.Enum):
    SEGMENTATION = 'segmentation'


class Organ(enum.Enum):
    ESD = 'ESD'
    LC = 'LC'


def normalise(data, nmax=1., nmin=0.):
    return (data-data.min()) * ((nmax - nmin) / (data.max() - data.min() + 1e-8)) + nmin


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



def run_epoch(action, loader, model, criterion, optimiser, device, num_training_subjects=None):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    for batch in tqdm(loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        inputs, targets = prepare_batch(batch, device)
        optimiser.zero_grad()
        with torch.set_grad_enabled(is_training):
            if num_training_subjects is not None:
                if is_training:
                    batch_loss = model.sample_elbo(inputs=inputs,
                                                labels=targets,
                                                criterion=criterion,
                                                sample_nbr=5,
                                                complexity_cost_weight=1./num_training_subjects) 
                    batch_loss.backward()
                    optimiser.step()
                else:
                    logits = forward(model, inputs)
                    batch_loss = criterion(logits, targets)
            else:
                logits = forward(model, inputs)
                batch_loss = criterion(logits, targets)
                if is_training:
                    batch_loss.backward()
                    optimiser.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    return epoch_losses.mean()


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
