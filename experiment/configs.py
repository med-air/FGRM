from pathlib import Path
import torch
import multiprocessing
from utils import Task, Organ
RANDOM_SEED = 13

TASK = Task.SEGMENTATION
SAVE_SEGMENTATION_OUTPUTS = True


EXPERIMENT_NAME = 'ESD_dce_evidence_u_loss'

if 'ESD' in EXPERIMENT_NAME:
    DATASET = 'ESD'
    ORGAN = Organ.ESD

if 'LC' in EXPERIMENT_NAME:
    DATASET = 'LC'
    ORGAN = Organ.LC

wandb_active = False
NUM_CLASSES = 5
ROOT_PATH = Path('/research/d5/gds/hzyang22/project/')
SOURCE_CODE_PATH = ROOT_PATH / 'ESD'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# print('WILL RUN ON DEVICE:', DEVICE)

NUM_WORKERS = 2
# print('NUMBER OF CPU WORKERS:', NUM_WORKERS)
NUM_ENSEMBLES = 16
LR = 0.0001
LR_MIN = 0.001
LR_DECAY_FACTOR = 0.5
OVERWRITE = False
RESULTS_PATH = SOURCE_CODE_PATH / 'results'

edl_uncertainty = False
LAMBDA = 1.
DROPOUT_RATE = 0.0
NUM_EPOCHS = 100
AU_WARMUP = 100
LAMBDA_EPOCH = 10
EARLY_STOPPING_PATIENCE = 2
SCHEDULER_PATIENCE = 100  # NOTE same as num epochs, never reduces lr
BATCH_SIZE = 4
VAL_BATCH_SIZE = BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE

