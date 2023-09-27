import numpy as np
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc
from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import math
from MutualInformation import MutualInformation
from calibration import get_calibration_error, PlattBinnerMarginalCalibrator
import sys, os
import torchvision.transforms as transforms
import torch


def calc_ece_softmax(edl_uncertainty, edl_u, softmax, label, bins=5, sample_wise=False):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)
    edl_u = torch.tensor(edl_u)
    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)
    batch_size = softmax.shape[0]
    num_classes = softmax.shape[1]
    plugin_ece = torch.zeros(batch_size)
    for i in range(batch_size):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            if edl_uncertainty:
                in_bin = edl_u[i].gt(bin_lower.item()) * edl_u[i].le(bin_upper.item())
            else:
                in_bin = softmax_max[i].gt(bin_lower.item()) * softmax_max[i].le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0.0:
                accuracy_in_bin = correctness[i][in_bin].float().mean()
                if edl_uncertainty:
                    avg_confidence_in_bin = edl_u[i][in_bin].mean()
                else:
                    avg_confidence_in_bin = softmax_max[i][in_bin].mean()
                plugin_ece[i] += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    # batch_size = softmax.shape[0]
    # num_classes = softmax.shape[1]
    # debiased_ece = torch.zeros(batch_size)
    # for i in range(batch_size):
    #
    #     zs = np.reshape(torch.tensor(softmax[i]).permute(1, 2, 0).numpy(), (-1, num_classes))
    #     ys = np.reshape(label[i], (-1, ))
    #     debiased_ece[i] = get_calibration_error(zs, ys)
        # calibrator = PlattBinnerMarginalCalibrator(zs.shape[0], num_bins=10)
        #
        # calibrator.train_calibration(zs, ys)
        # calibrated_zs = calibrator.calibrate(zs)
        # calibrated_zs = calibrated_zs / np.expand_dims(np.linalg.norm(calibrated_zs, ord=1, axis=1), axis=1)
        # debiased_ece[i] = get_calibration_error(calibrated_zs, ys)
    ece = plugin_ece
    if sample_wise:
        ece = ece
    else:
        ece = ece.mean().item()
    
    return ece

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def calc_mi(evidence, labels, sample_wise=False):
    device = get_device()
    evidence = torch.from_numpy(evidence)
    labels = torch.from_numpy(labels)
    _, preds = torch.max(evidence, 1)
    match = torch.eq(preds, labels)
    match = match.unsqueeze(1)

    alpha = evidence + 1

    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
    uncertainty, _ = torch.max(expected_prob, dim=1, keepdim=True)
    match, uncertainty = match.cuda(), uncertainty.cuda()
    MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True, device=device)
    score = ( MI(match, uncertainty) + MI(uncertainty, match) ) / 2.
    if sample_wise:
        return score
    score = score.mean().item()
    return score


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt, num_classes):
        """ computational formula
        """

        all_dice = 0

        seg_pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
        label_ratio_ones = torch.ones_like(seg_pred)
        label_ratio_sum = torch.sum(label_ratio_ones)
        for i in range(num_classes):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt))
            A = torch.sum(each_pred)
            B = torch.sum(each_gt)
            union = A + B
            if union == 0:
                dice = 1
            else:
                dice = (2. * intersection) / union
            label_ratio = (B*1.0) / label_ratio_sum
            all_dice += label_ratio * dice

        return all_dice * 1.0

    def forward(self, pred, gt):
        pass

def get_evaluations(s, l, evidence, soft_output, hard_labels, edl_uncertainty, edl_u, spacing=(1, 1, 1)):
    """
    Helper function to compute all the evaluation metrics:

    - Segmentation volume, number of regions, min and max vol for each region
    - TPF (segmentation and detection)
    - FPF (segmentation and detection)
    - DSC (segmentation and detection)
    - PPV (segmentation and detection)
    - Volume difference
    - Haursdoff distance (standard and modified)
    - Custom f-score

    Inputs:
    - gt: 3D np.ndarray, reference image (ground truth)
    - mask: 3D np.ndarray, input MRI mask
    - spacing: sets the input resolution (def: (1, 1, 1))

    Output:
    - (dict) containing each of the evaluated results

    """

    metrics = {}
    dice_metric = DiceLoss()
    num_classes = soft_output.shape[1]

    metrics['dsc_seg'] = dice_metric.dice_coef(s, l, num_classes)
    metrics['ece'] = calc_ece_softmax(edl_uncertainty, edl_u, soft_output, hard_labels)
    metrics['mi'] = calc_mi(evidence, hard_labels)

    return metrics

