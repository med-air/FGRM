import scipy
import warnings 
from scipy.ndimage.morphology import binary_erosion
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils import make_folders, EarlyStopping, Task, Organ
from metrics import get_evaluations
from methods.commons import get_model_for_task
from methods.loaders import get_loaders

from pathlib import Path
from PIL import Image
# from torchsummary import summary

from configs import NUM_CLASSES, AU_WARMUP
from methods.losses import dce_evidence_u_loss, policy_gradient_loss_ece_calibration, nll_evidence_loss
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from copy import deepcopy
from torch.autograd import Variable

# torch.autograd.set_detect_anomaly(True)

class BaseMethod:
    def __init__(self, configs):
        # Task is either 'segmentation' or 'classification' or 'regression'
        # Organ is either 'breast' or 'heart'
        self.task = configs.TASK
        self.organ = configs.ORGAN
        if self.organ == Organ.ESD:
            self.num_classes = NUM_CLASSES
        elif self.organ == Organ.LC:
            self.num_classes = NUM_CLASSES
        self.configs = configs
        self.loss_lambda = configs.LAMBDA
        self.results_path = configs.RESULTS_PATH
        self.experiment_name = configs.EXPERIMENT_NAME
        self.dataset = configs.DATASET
        self.model = None
        self.overwrite = configs.OVERWRITE
        self.device = configs.DEVICE
        self.num_epochs = configs.NUM_EPOCHS
        self.au_warmup = configs.AU_WARMUP
        # Make experiment folders
        self.final_model_path, self.figures_path, self.metrics_out_path, self.results_csv_file = self.folders()

        self.lamda_epochs = configs.LAMBDA_EPOCH
        self.precision_matrices = {}
        self.wandb_active = configs.wandb_active

    def run_FGRM(self):
        train_EDL_transforms, train_FGRM_transforms, test_transforms = self.prepare_transforms()
        train_loader, val_loader, test_loader = self.loaders(self.dataset, train_EDL_transforms, train_FGRM_transforms,
                                                             test_transforms, edl_train=False)

        self.model, self.intermediate_layers = self.prepare_model(model_path=self.best_model_path)

        self.optimizer = self.prepare_optimizer(self.model)
        self.scheduler = self.prepare_scheduler(self.optimizer)
        self.train_FGRM(train_loader, val_loader, test_loader)

    def train_FGRM(self, train_loader, val_loader, test_loader):
        best_val_dice = 0
        best_val_ece = 10
        best_val_mi = 0
        best_val_loss = 10000
        val_loss, val_dice, val_ece, val_mi = self.validate_FGRM_epoch(self.model, test_loader)
        print(f'val dice : {val_dice:.6f} '
              f'val ece : {val_ece:.6f} val mi : {val_mi:.6f}.')
        for _, epoch in zip(np.linspace(0.01, 0.99, num=self.num_epochs), range(self.num_epochs)):
            # Train
            train_loss = self.train_FGRM_epoch(self.model, val_loader, self.optimizer)

            print(f'EPOCH {epoch}/{self.num_epochs}: Loss:', train_loss.item())

            val_loss, val_dice, val_ece, val_mi = self.validate_FGRM_epoch(self.model, test_loader)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
            if val_mi > best_val_mi:
                best_val_mi = val_mi
            if val_ece < best_val_ece:
                best_val_ece = val_ece

            print(f'val dice : {val_dice:.6f} '
                  f'val ece : {val_ece:.6f} val mi : {val_mi:.6f}.')
            print(f'Best val dice : {best_val_dice:.6f} '
                  f'Best val ece : {best_val_ece:.6f} Best val mi : {best_val_mi:.6f}.')

            self.scheduler.step(val_loss)

        return None

    @staticmethod
    def normalize_tensor(data):
        return ((data - torch.min(data)) + 1e-6) / (torch.max(data) - torch.min(data) + 1e-6)

    @staticmethod
    def get_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        return device

    @staticmethod
    def variable(t: torch.Tensor, use_cuda=True, **kwargs):
        if torch.cuda.is_available() and use_cuda:
            t = t.cuda()
        return Variable(t, **kwargs)

    def diag_fisher(self, model, train_loader):
        params = {n: p for n, p in model.named_parameters()}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = self.variable(p.data)
        training = model.training
        model.train()
        for batch_idx, batch in enumerate(train_loader):

            data, target = batch['image'], batch['label']
            data, target = data.cuda(), target.cuda()
            model.zero_grad()
            outputs = model(data)
            evidence = F.softplus(outputs, beta=20)
            alpha = evidence + 1

            total_loss = nll_evidence_loss(target, alpha, self.num_classes)
            total_loss = torch.mean(total_loss)
            total_loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(train_loader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        model.train(training)
        return precision_matrices

    def train_FGRM_epoch(self, model, train_loader, optimizer):
        model.train()
        train_loss = 0
        self.precision_matrices = self.diag_fisher(model, train_loader)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.segmentation_head.parameters():
            p.requires_grad = True

        for batch_idx, batch in enumerate(train_loader):

            data, target = batch['image'], batch['label']
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            outputs = model(data)

            evidence = F.softplus(outputs, beta=20)
            alpha = evidence + 1
            edl_u = self.configs.NUM_CLASSES / torch.sum(alpha, dim=1, keepdim=False)

            reward = policy_gradient_loss_ece_calibration(self.configs.edl_uncertainty, edl_u, target,
                                                                alpha, evidence, batch_idx)

            total_loss = reward
            total_loss = torch.mean(total_loss)
            total_loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    dims = len(self.precision_matrices[n].shape)
                    dims = tuple([i for i in range(dims)])
                    tmp_precision = torch.nn.functional.normalize(1 / self.precision_matrices[n], p=1, dim=dims)
                    p.grad.data = p.grad.data * tmp_precision.data
            train_loss += total_loss
            optimizer.step()

        for p in model.parameters():
            p.requires_grad = True
        return train_loss / len(train_loader)

    def validate_FGRM_epoch(self, model, val_loader):

        model.eval()
        val_loss = 0
        val_dice = []
        val_ece = []
        val_mi = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data, target = batch['image'], batch['label']
                data, target = data.cuda(), target.cuda()
                outputs = model(data)

                evidence = F.softplus(outputs, beta=20)
                alpha = evidence + 1
                edl_u = self.configs.NUM_CLASSES / torch.sum(alpha, dim=1, keepdim=False)
                soft_output = F.normalize(alpha, p=1, dim=1)

                seg = torch.argmax(evidence.squeeze(), dim=1).detach().cpu().numpy()
                lbl = target.squeeze().detach().cpu().numpy()
                for s, l in zip(seg, lbl):
                    evals = get_evaluations(s, l, evidence.detach().cpu().numpy(),
                                            soft_output.detach().cpu().numpy(), target.detach().cpu().numpy(),
                                            self.configs.edl_uncertainty, edl_u.detach().cpu().numpy(), spacing=(1, 1))
                    val_dice.append(evals['dsc_seg'])
                    val_ece.append(evals['ece'])
                    val_mi.append(evals['mi'])

        assert all([dsc <= 1.0001 for dsc in val_dice])
        return val_loss / len(val_loader), sum(val_dice) / len(val_dice), sum(val_ece) / len(val_ece), sum(val_mi) / len(val_mi)

    def run_EDL(self, run_test=True):
        if self.wandb_active:
            wandb.init(
                project="uncertainty_estimation",
                name=self.experiment_name,
                config={
                    "method": self.experiment_name[len(self.dataset) + 1:],
                    "lambda_epochs": self.lamda_epochs,
                    'total_epochs': self.num_epochs,
                    'AU_warmup': self.au_warmup,
                    'experiment_name': self.experiment_name,
                    'lr': self.configs.LR,
                }
            )
        train_EDL_transforms, train_FGRM_transforms, test_transforms = self.prepare_transforms()
        train_loader, val_loader, test_loader = self.loaders(self.dataset, train_EDL_transforms, train_FGRM_transforms,
                                                             test_transforms, edl_train=True)
        if self.final_model_path.exists() and not self.overwrite:
            print(f'EXPERIMENT {self.experiment_name} EXISTS!')
            print(f'TESTING ONLY')
            self.model, self.intermediate_layers = self.prepare_model(model_path=self.best_model_path)

        else:

            self.model, self.intermediate_layers = self.prepare_model(self.target_shape)
            self.optimizer = self.prepare_optimizer(self.model)
            self.scheduler = self.prepare_scheduler(self.optimizer)
            self.early_stopping = self.prepare_early_stopping()
            self.train_EDL(train_loader, val_loader)

        if run_test:
            self.test(test_loader)

    def train_EDL(self, train_loader, val_loader):
        
        for alpha, epoch in zip(np.linspace(0.01, 0.99, num=self.num_epochs), range(self.num_epochs)):
            # Train
            train_loss, all_head_losses = self.train_EDL_epoch(self.model, train_loader, self.optimizer, epoch)

            print(f'EPOCH {epoch}/{self.num_epochs}: Loss:', all_head_losses)
            # wandb.log({'training_loss': all_head_losses})
            # Validate

            val_loss, val_dice, val_ece, val_mi = self.validate_EDL_epoch(self.model, val_loader, epoch)
           
            annealing_start = torch.tensor(0.01, dtype=torch.float32)
            annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / self.au_warmup * epoch)
            annealing_coef = min(1, epoch / self.lamda_epochs)

            # Update scheduler and early stopping
            if self.wandb_active:
                wandb.log({'training_loss': all_head_losses, 'val dice': val_dice,
                           'val ece': val_ece, 'val mi': val_mi,
                           'annealing_au': annealing_AU, 'annealing_coef': annealing_coef})
            print(self.experiment_name)
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)
            # Push to tensorboard if enabled

            if self.early_stopping.early_stop:
                print(f'EARLY STOPPING at EPOCH {epoch + 1}')
                break

        torch.save(self.model.state_dict(), self.final_model_path)
        return train_loss, val_loss

    def train_EDL_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0
        all_train_losses = []
        for batch_idx, batch in enumerate(train_loader):

            data, target = batch['image'], batch['label']
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            outputs = model(data)
            evidence = F.softplus(outputs, beta=20)
            alpha = evidence + 1

           
            total_loss = dce_evidence_u_loss(target, alpha, self.num_classes, epoch,
                                                 self.lamda_epochs, self.au_warmup, eps=1e-10, disentangle=False,
                                                 evidence=evidence, backbone_pred=evidence)
            total_loss = torch.mean(total_loss)
            total_loss = total_loss

            all_train_losses.append(total_loss.item())
            total_loss.backward()
            train_loss += total_loss
            optimizer.step()
        all_train_losses = np.asarray(all_train_losses).mean(axis=0)
        return train_loss / len(train_loader), all_train_losses

    def validate_EDL_epoch(self, model, val_loader, epoch):

        model.eval()
        val_loss = 0
        val_dice = []
        val_ece = []
        val_mi = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data, target = batch['image'], batch['label']
                data, target = data.cuda(), target.cuda()
                outputs = model(data)
                evidence = F.softplus(outputs, beta=20)
                alpha = evidence + 1
                soft_output = F.normalize(evidence, p=1, dim=1)
                edl_u = self.configs.NUM_CLASSES / torch.sum(alpha, dim=1, keepdim=False)

                seg = torch.argmax(evidence.squeeze(), dim=1).detach().cpu().numpy()
                lbl = target.squeeze().detach().cpu().numpy()
                for s, l in zip(seg, lbl):
                    evals = get_evaluations(s, l, evidence.detach().cpu().numpy(),
                                            soft_output.detach().cpu().numpy(), target.detach().cpu().numpy(),
                                            self.configs.edl_uncertainty, edl_u.detach().cpu().numpy(), spacing=(1, 1))
                    val_dice.append(evals['dsc_seg'])
                    val_ece.append(evals['ece'])
                    val_mi.append(evals['mi'])

                
                total_loss = dce_evidence_u_loss(target, alpha, self.num_classes, epoch,
                                                     self.lamda_epochs, self.au_warmup, eps=1e-10, disentangle=False,
                                                     evidence=evidence, backbone_pred=evidence)

                total_loss = torch.mean(total_loss)
                total_loss = total_loss
                val_loss += total_loss.item()
        assert all([dsc <= 1.0001 for dsc in val_dice])
        return val_loss / len(val_loader), sum(val_dice) / len(val_dice), sum(val_ece) / len(val_ece), sum(val_mi) / len(val_mi)

    def run_routine(self, run_test=True):
        # Load data
        train_EDL_transforms, train_FGRM_transforms, test_transforms = self.prepare_transforms()
        train_loader, val_loader, test_loader = self.loaders(self.dataset, train_EDL_transforms, train_FGRM_transforms,
                                                             test_transforms, edl_train=False)
        # Load pretrained model if already exists and not overwriting
        if self.final_model_path.exists() and not self.overwrite:
            print(f'EXPERIMENT {self.experiment_name} EXISTS!')
            print(f'TESTING ONLY')
            self.model, self.intermediate_layers = self.prepare_model(target_shape=self.target_shape, model_path=self.best_model_path)
            self.test(test_loader)

    def folders(self):
        models_path, figures_path, metrics_out_path = make_folders(self.results_path,
                                                                   self.experiment_name,
                                                                   self.configs)
        best_model_path = models_path / 'best_model.pt'
        final_model_path = models_path / 'final_model.pt'
        results_csv_file = f'results_{self.experiment_name}.csv'
        return best_model_path, final_model_path, figures_path, metrics_out_path, results_csv_file

    def prepare_model(self, model_path=None, model_type=None):
        if model_type == 'PostNet':
            N = torch.tensor([3000., 3000., 3000., 3000., 3000.])
            model, intermediate_layers = PosteriorNetwork(N, self.num_classes), []
        else:
            model, intermediate_layers = get_model_for_task(self.task, self.num_classes, encoder_weights=None)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.cuda()
        return model, intermediate_layers
    
    def prepare_optimizer(self, model):
        # A separate function in case we want to add different optimizers
        return torch.optim.Adam(model.parameters(), lr=self.configs.LR)
    
    def prepare_scheduler(self, optimizer):
        # A separate function in case we want to add different schedulers
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.configs.LR_DECAY_FACTOR, patience=self.configs.SCHEDULER_PATIENCE,
                                                          min_lr=self.configs.LR_MIN, verbose=True)

    def prepare_early_stopping(self):
        # A separate function in case we want to add different early stopping strategies
        return EarlyStopping(patience=self.configs.EARLY_STOPPING_PATIENCE, verbose=True)
    
    def prepare_transforms(self, target_image_size=256,
                           min_intensity=0, max_intensity=1,
                           min_percentile=0, max_percentile=100,
                           perturb_test=False,
                           ):
        # This could be called directly in self.run_routine(), but keep it here as it may change in the future
        train_FGRM_transforms = A.Compose(
            [
                A.Resize(256, 256),
                A.HorizontalFlip(p=1),
                A.ElasticTransform(p=1),
                A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=30, p=1),
                A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=25, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],  is_check_shapes=False
        )
        train_EDL_transforms = A.Compose(
            [
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], is_check_shapes=False
        )
        test_transforms = A.Compose(
            [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()],
            is_check_shapes=False
        )
        return train_EDL_transforms, train_FGRM_transforms, test_transforms

    def loaders(self, dataset, train_EDL_transforms, train_FGRM_transforms, test_transforms, edl_train, batch_size=None):
        # This could be called directly in self.run_routine(), but keep it here as it may change in the future
        train_loader, val_loader, test_loader = get_loaders(dataset, train_EDL_transforms, train_FGRM_transforms,
                                                            test_transforms, edl_train, batch_size)
        return train_loader, val_loader, test_loader
