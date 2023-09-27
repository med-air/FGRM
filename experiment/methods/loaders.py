import sys 
sys.path.append('..')
from functools import partial
import configs
import torch
import torchio as tio
from torch.utils.data import DataLoader
from data.esd_loader import ESD_Dataset, get_split
from data.lc_loader import LC_Dataset, get_lc_split
from sklearn.model_selection import GroupShuffleSplit
from utils import Task, Organ


def get_loaders(dataset, train_EDL_transforms, train_FGRM_transforms,
                test_transforms, edl_train, batch_size=None):

    if dataset == 'ESD': 
        train_ids, val_ids, test_ids = get_split()
        train_dataset = ESD_Dataset(train_ids, train_EDL_transforms)
        if edl_train:
            val_dataset = ESD_Dataset(val_ids, test_transforms)
        else:
            val_dataset = ESD_Dataset(val_ids, train_FGRM_transforms)
        test_dataset = ESD_Dataset(test_ids, test_transforms)
    elif dataset == 'LC':
        train_ids, val_ids, test_ids = get_lc_split()
        train_dataset = LC_Dataset(train_ids)
        val_dataset = LC_Dataset(val_ids)
        test_dataset = LC_Dataset(test_ids)
    else:
        train_dataset = None
        val_dataset = None
        test_dataset = None

    train_batch_size = batch_size if batch_size is not None else configs.BATCH_SIZE

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.VAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

