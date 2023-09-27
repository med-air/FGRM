from pathlib import Path
import torch
import multiprocessing
from utils import Task, Organ
RANDOM_SEED = 13

TASK = Task.SEGMENTATION
SAVE_SEGMENTATION_OUTPUTS = True  # if False, only evaluation metrics are saved


EXPERIMENT_NAME = 'LC_dce_evidence_u_loss'

aug = True
if 'ESD' in EXPERIMENT_NAME:
    DATASET = 'ESD'
    ORGAN = Organ.ESD  # Organ.BREAST | Organ.HEART

if 'LC' in EXPERIMENT_NAME:
    DATASET = 'LC'
    ORGAN = Organ.LC  # Organ.BREAST | Organ.HEART

wandb_active = True
NUM_CLASSES = 5
ROOT_PATH = Path('/research/d5/gds/hzyang22/project/')
SOURCE_CODE_PATH = ROOT_PATH / 'code'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

NUM_WORKERS = 2
LR = 1e-4
LR_MIN = 0.001
LR_DECAY_FACTOR = 0.5
TENSORBOARD = True
OVERWRITE = False
TENSORBOARD_ROOT = SOURCE_CODE_PATH / 'tensorboard'
RESULTS_PATH = SOURCE_CODE_PATH / 'results'

edl_uncertainty = False
LAMBDA = 1.
DROPOUT_RATE = 0.0
NUM_EPOCHS = 100
FGRM_EPOCHS = 1
AU_WARMUP = 10
LAMBDA_EPOCH = 10
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 10
BATCH_SIZE = 4
VAL_BATCH_SIZE = BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE

