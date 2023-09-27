import torch
import torchvision.models as models
from collections import OrderedDict 
from segmentation_models_pytorch.unet.model import Unet
from segmentation_models_pytorch.unetplusplus.model import UnetPlusPlus
from segmentation_models_pytorch.encoders import get_encoder
from typing import Optional, List, Union
from torch.nn import ModuleList
from utils import Task, Organ

def get_model_for_task(task, num_classes, encoder_weights='imagenet'):
    # TODO check the number of classes for cardiac (segmentation: 4, classification: ?)
    num_classes = num_classes
    if task == Task.SEGMENTATION:
        architecture = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=3,
            classes=num_classes,
        )

        return architecture, []
