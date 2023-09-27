import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize
)
from torchvision.transforms import ToTensor
from torchvision import transforms
import albumentations as A

target_img_size = 256



def image_transform(p=1):
    return Compose([
        Resize(target_img_size, target_img_size, cv2.INTER_LINEAR),
        Normalize(p=1)], is_check_shapes=False
    )


def mask_transform(p=1):
    return Compose([
        Resize(target_img_size, target_img_size, cv2.INTER_NEAREST)
    ], is_check_shapes=False)

def get_split():
    data_path = Path('/research/d5/gds/hzyang22/data/new_esd_seg')
    # data_path = Path('/research/d5/gds/hzyang22/data/ESD_organized/')
    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    data_file_names = []
   
    data_ids = ['01', '03', '05', '06',
                '08', '09', '11', '12', '13', '14', '15', '16', '18', '23',
                '24', '25', '26', '27', '28', '29', '30', '31',
                '32', '33', '34', '35', '36']
 
    for data_id in data_ids:
        data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))

    random.shuffle(data_file_names)
    training_num = 700
    train_file_names = data_file_names[:training_num]
    val_file_names = data_file_names[training_num:850]
    test_file_names = data_file_names[850:]

    return train_file_names, val_file_names, test_file_names


class ESD_Dataset(Dataset):
    def __init__(self, file_names, transforms=None):
        self.file_names = file_names
        self.image_transform = image_transform()
        self.mask_transform = mask_transform()
        self.transforms = transforms


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]

        image = load_image(img_file_name)
        mask = load_mask(img_file_name)
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        label = transformed["mask"].long()


        # data = {"image": image, "mask": mask}
        # mask = self.mask_transform(image=image, mask=mask)
        # mask = mask["mask"]
        # image = self.image_transform(image=image)
        #
        # image = image['image']
        # image = img_to_tensor(image)
        #
        # label = torch.from_numpy(mask).long()
        sample = {'image': image, 'label': label, 'id': str(img_file_name).split('/')[-1]}
        return sample


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_folder = 'mask'
    factor = 85
    path = str(path).replace('image', mask_folder)
    identifier = path.split('/')[-1]
    path = path.replace(identifier, identifier[:-4] + '_mask' + '.png')
    mask = cv2.imread(path, 0)
  
    mask[mask == 255] = 4
    mask[mask == 212] = 0
    mask[mask == 170] = 0
    mask[mask == 128] = 3
    mask[mask == 85] = 2
    mask[mask == 42] = 1
    # mask[mask == 255] = 3
    # mask[mask == 170] = 2
    # mask[mask == 85] = 1
    return mask.astype(np.uint8)
