import sys 
sys.path.append('..')
import configs
import torch
from torch.utils.data import DataLoader
from data.lc_loader import LC_Dataset, get_lc_split

def get_loaders(dataset, data_transforms, edl_train, batch_size=None):


    train_ids, val_ids, test_ids, test_video = get_lc_split()
    train_dataset = LC_Dataset(train_ids, data_transforms, 'train')
    test_video_dataset = {}

    val_dataset = LC_Dataset(val_ids, data_transforms, 'val')
    test_dataset = LC_Dataset(test_ids, data_transforms, 'test')
    for video_name, video_img in test_video.items():
        test_video_dataset[video_name] = LC_Dataset(video_img, data_transforms, 'test')


    train_batch_size = batch_size if batch_size is not None else configs.BATCH_SIZE

    test_video_loader = {}
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.VAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False)
    for video_name, video_dataset in test_video_dataset.items():
        test_video_loader[video_name] = DataLoader(video_dataset, batch_size=configs.TEST_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_video_loader

