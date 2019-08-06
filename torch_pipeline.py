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
    def __init__(self, model: nn.Module,
                 max_lr: float, min_lr: float,
                 bs: int, loader: DataLoader, n_epochs: int):
        super().__init__(model, max_lr, min_lr,
                         bs, loader, n_epochs)

    def _run_epoch(self, optimizer: optim.Optimizer,
                   scheduler: PytorchScheduler, criterion: nn.Module):
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
        optimizer.lr = self.min_lr
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._calculate_lr)
        self.model.train()
        for ep in range(self.n_epochs):
            print(f'epoch {ep}')
            self._run_epoch(optimizer, scheduler, criterion)


if __name__ == '__main__':
    model = MobileNetV2(2, 1.0)
    dataset = make_classification(2000)

    data = dataset[0]
    labels = dataset[1]

    finder = LRFinderPytorch(model, 1.0, 1e-6, )