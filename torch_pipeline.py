# from copy import deepcopy
# import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from base_pipeline import LRFinderBase, BaseScheduler


class PytorchScheduler(BaseScheduler, optim.lr_scheduler._LRScheduler):
    """ Pytorch specific learning rate scheduler

        Parameters
        ----------
            min_lr: float   
                Lower bound of learning rate scheduler
            max_lr: float
                Upper bound of learin rate scheduler
            scaler: Scaler, callable
                Scale strategy function
            data_len: int
                Number of samples in dataset
            n_times:
                Number of epochs for one cycle      
            opeimizer: torch.optim.Optimizer
                Neural network optimizer
    """
    def __init__(self, min_lr, max_lr, scaler, data_len, n_times, optimizer):
        self.optimizer = optimizer
        self.last_epoch = -1
        super(PytorchScheduler, self).__init__(min_lr=min_lr,
                                               max_lr=max_lr,
                                               scaler=scaler,
                                               data_len=data_len,
                                               n_times=n_times,
                                               optimizer=optimizer,
                                               last_epoch=self.last_epoch)

        def get_lr(self):
            lr = self.compute_lr(self.last_epoch, self.stepsize)
            return lr


class LRFinderPytorch(LRFinderBase):
    """ Class for finding best learning rate for pytorch models

        Parameters
        ----------
            model: torch.nn.Module
                Pytorch model for witch you want to find optimal lr
            min_lr: float
                Lower bound of learning rate scheduler
            max_lr: float
                Upper bound of learin rate scheduler
            loader: torch.DataLodaer
                Batch generator

        Attributes
        ----------
            losses: list
                Loss values at each iteration
            lr_s: list
                Learning rate values at each iteration
            data_len: int
                Number of batches in dataset
            device: str
                Pytorch device
    """
    def __init__(self, model: nn.Module,
                 min_lr: float, max_lr: float,
                 loader: DataLoader, n_epochs: int):
        self.model = model
        self.device = next(self.model.parameters()).device
        super().__init__(model=model, min_lr=min_lr, max_lr=max_lr,
                         loader=loader, n_epochs=n_epochs)

    def _run_epoch(self, optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler, criterion: nn.Module):
        lr_step = self.min_lr
        with tqdm(self.loader,
                  postfix=["Current state: ", dict(loss=0, lr=lr_step)]) as t:
            for data, target in t:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                lr_step = optimizer.state_dict()['param_groups'][0]['lr']
                self.lr_s += [lr_step]
                self.losses += [loss.item()]
                t.postfix[1]['loss'] = loss.item()
                t.postfix[1]['lr'] = lr_step
                t.update()

    def run(self, criterion: nn.Module, optimizer: optim.Optimizer):
        for p_group in optimizer.param_groups:
            p_group['lr'] = self.min_lr

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, [self._calculate_lr])
        self.model.train()
        for ep in range(self.n_epochs):
            print(f'epoch {ep}')
            self._run_epoch(optimizer, scheduler, criterion)
        self._smooth_losses()
