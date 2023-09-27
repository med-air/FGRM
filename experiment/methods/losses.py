
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import numpy as np
import logging
from torch.autograd import Variable
from torch import nn
from torch.nn.modules.loss import _Loss
from metrics import calc_mi, calc_ece_softmax, get_device
from torch.distributions.dirichlet import Dirichlet
from data.esd_loader import target_img_size
from torch.autograd import Variable
from calibration import get_calibration_error, PlattBinnerMarginalCalibrator
from copy import deepcopy


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, embedding):
        embedding = nn.Softmax(dim=1)(embedding)
        minus_entropy = embedding * torch.log(embedding)
        minus_entropy = torch.sum(minus_entropy, dim=1)
        return minus_entropy.mean()

class dissimilar_loss(nn.Module):
    def __init__(self):
        super(dissimilar_loss, self).__init__()

    def forward(self, protos):
        loss = -1 * torch.mean(torch.cdist(protos, protos))
        return loss

def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)

class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1,1]) + 1e-16
        gkernel[int(ksize/2), int(ksize/2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize/2)
        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False
    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()

class CELossWithSVLS_2D(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, ksize=3):
        super(CELossWithSVLS_2D, self).__init__()
        self.cls = torch.tensor(classes)
        self.svls_layer = get_svls_filter_2d(ksize=3, sigma=sigma, channels=self.cls).cuda()

    def forward(self, inputs, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()
        svls_labels = self.svls_layer(oh_labels)
        # print(inputs.shape, svls_labels.shape)
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    Mbeta = torch.ones(alpha.shape).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    # print(beta.shape, lnB_uni.shape)
    kl = torch.sum((alpha - Mbeta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


class DiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)
        if preds.ndim==5:
            preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        else:
            preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        log_P = F.log_softmax(preds, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device) + 1e-8
        class_mask.scatter_(1, targets, 1.)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.alpha * torch.sum(FP, dim=(0)).float() + self.beta * torch.sum(FN, dim=(0)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss
        loss = 1 - dice
        if weight is not False:
            loss *= weight.squeeze(0)
        loss = loss.sum()
        if self.size_average:
            if weight is not False:
                loss /= weight.squeeze(0).sum()
            else:
                loss /= C

        return loss
class SDiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(SDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets,weight_map=None):
        N = preds.size(0)
        C = preds.size(1)
        num_class = C
        if preds.ndim==5:
            preds = preds.permute(0, 2, 3, 4, 1)
        else:
            preds = preds.permute(0, 2, 3, 1)
        pred = preds.contiguous().view(-1, num_class)
        # pred = F.softmax(pred, dim=1)
        ground = targets.view(-1, num_class)
        n_voxels = ground.size(0)
        if weight_map is not None:
            weight_map = weight_map.view(-1)
            weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
            ref_vol = torch.sum(weight_map_nclass * ground, 0)
            intersect = torch.sum(weight_map_nclass * ground * pred, 0)
            seg_vol = torch.sum(weight_map_nclass * pred, 0)
        else:
            ref_vol = torch.sum(ground, 0)
            intersect = torch.sum(ground * pred, 0)
            seg_vol = torch.sum(pred, 0)
        dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
        # dice_loss = 1.0 - torch.mean(dice_score.data[1:dice_score.shape[0]])
        k = -torch.log(dice_score)
        # 1. mean-loss
        dice_mean_score = torch.mean(-torch.log(dice_score))
        # 2. sum-loss
        # dice_score1 = torch.sum(dice_score,0)
        # dice_mean_score1 = -torch.log(torch.sum(dice_score,0))
        # dice_mean_score = torch.mean(-torch.log(dice_score.data[1:dice_score.shape[0]]))
        return dice_mean_score
        # dice_score = torch.mean(-torch.log(dice_score))
        # return dice_score


def TDice(output, target,criterion_dl):
    dice = criterion_dl(output, target)
    return dice

def U_entropy(logits,c):
    pc = F.softmax(logits, dim=1)
    logits = F.log_softmax(logits, dim=1)
    pc = pc.view(-1, 1)
    u = -pc* logits/c
    return u

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    if input_tensor.ndim == 5:
        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    else:
        input_tensor = input_tensor.permute(0, 2, 3, 1)
    # input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def dce_evidence_u_loss(p, alpha, c, current_step, lamda_step, total_step, eps, disentangle, evidence, backbone_pred):
    # c: class number
    criterion_dl = DiceLoss()
    # notes: may be use SDiceloss:
    # criterion_dl = SDiceLoss()

    # soft_p = get_soft_label(soft_p, c)
    if alpha.ndim == 5:
        soft_p = p.unsqueeze(1)
    else:
        soft_p = p

    L_dice = TDice(evidence, soft_p, criterion_dl)
    # print(L_dice)

    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    # KL loss
    annealing_coef = min(1, current_step / lamda_step)
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_step * current_step)
    annealing_AU = min(1., annealing_AU)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)
    # AU Loss
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = c / S
    target = p.view(-1, 1)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))
    if disentangle:
        acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + eps)
        inacc_certain = - torch.log((1 - pred_scores) * uncertainty + eps)
    else:
        acc_uncertain = - pred_scores * torch.log(1 - uncertainty + eps)
        inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + eps)
    L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain

    return L_ace + L_KL + (1 - annealing_AU)*L_dice + L_AU



def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes)
    device = labels.get_device()
    y = y.to(device)
    return y[labels].permute(0,3,1,2).float()

def policy_gradient_loss_ece_calibration(edl_uncertainty, edl_u, p, alpha, evidence, batch_idx):
    device = get_device()
    batch_size = evidence.shape[0]
    num_classes = evidence.shape[1]

    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
    ece = calc_ece_softmax(edl_uncertainty, edl_u.detach().cpu().numpy(), expected_prob.detach().cpu().numpy(),
                           p.detach().cpu().numpy(), sample_wise=True)

    calibrated_probs = []
    for i in range(batch_size):
        zs = np.reshape(expected_prob[i].permute(1, 2, 0).detach().cpu().numpy(), (-1, num_classes))
        ys = np.reshape(p[i].detach().cpu().numpy(), (-1, ))
        calibrator = PlattBinnerMarginalCalibrator(zs.shape[0], num_bins=10)
        calibrator.train_calibration(zs, ys)
        calibrated_zs = calibrator.calibrate(zs)
        calibrated_zs = calibrated_zs / np.expand_dims(np.linalg.norm(calibrated_zs, ord=1, axis=1), axis=1)
        calibrated_zs = np.reshape(calibrated_zs, (256, 256, num_classes))
        calibrated_probs.append(calibrated_zs)
    calibrated_probs = np.stack(calibrated_probs, axis=0) # (batch, 256, 256, num_classes)
    calibrated_probs = torch.tensor(calibrated_probs).to(device)
    expected_prob = expected_prob.permute(0, 2, 3, 1)

    log_probs = torch.zeros(batch_size)
    entropy = torch.zeros(batch_size)
    for i in range(batch_size):
        m = Dirichlet(alpha[i].permute(1, 2, 0).view(-1, num_classes))
        log_probs[i] = torch.sum(m.log_prob(calibrated_probs[i].view(-1, num_classes))) / (
                target_img_size * target_img_size)
        entropy[i] = torch.sum(m.entropy()) / (
                target_img_size * target_img_size)
    ece = - torch.log(ece)
    loss_ece = - log_probs.to(device) * ece.to(device)
    entropy = entropy.to(device)
    expected_prob = expected_prob.permute(0, 3, 1, 2)
    prob_entropy = -torch.sum(expected_prob * torch.log(expected_prob + 1e-6), dim=1)
    loss_ece = torch.mean(loss_ece)
    loss_entropy = torch.mean(entropy)
    prob_entropy = torch.mean(prob_entropy)

    return loss_ece - 0.1*prob_entropy


def nll_evidence_loss(p, alpha, c):
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    return L_ace
