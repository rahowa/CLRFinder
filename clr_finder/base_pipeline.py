from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class LRFinderBase(ABC):
    """ Base class for finding best learning rate

        Parameters
        ----------
            model: keras.Model, torch.Module
                Model for witch you want to find optimal lr
            min_lr: float
                Lower bound of learning rate scheduler
            max_lr: float
                Upper bound of learin rate scheduler
            loader: keras.Sequence, torch.DataLodaer
                Batch generator

        Attributes
        ----------
            losses: list
                Loss values at each iteration
            lr_s: list
                Learning rate values at each iteration
            data_len: int
                Number of batches in dataset
    """
    def __init__(self, model, min_lr, max_lr, loader, n_epochs):
        self.model = deepcopy(model)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.loader = loader
        self.n_epochs = n_epochs
        self.data_len = len(self.loader)
        self.lr_s = []
        self.losses = []

    def _calculate_lr(self, it):
        """ Calculate learning rate at current iteratrion

            Parameters
            ----------
                it: int
                    Number of current iteration

            Returns
            ----------
                current_lr: float
                    Learnin rate at current iteration
        """
        denominator = (self.n_epochs*self.data_len)
        nominator = it * np.log(self.max_lr/self.min_lr)
        current_lr = np.exp(nominator / denominator)
        return current_lr

    @abstractmethod
    def _run_epoch(self):
        """ Abstarct method for running model for one epoch
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, criterion, optimizer):
        """ Abstarcat method for run model and find learning rate
            Should run self._run_epoch and self._calculate_lr methods inside,
            append learning rate to self.lr_s,
            append loss value to self.losses
            should call self._smooth_losses at the end
        """
        raise NotImplementedError

    def _smooth_losses(self):
        """ Make Loss / learning rate ratio smoother
            for easyer debugging and chosig learning rate
        """
        self.smoothed_loss = [self.losses[0]]
        smoothing = 0.025
        for i in range(1, len(self.losses)):
            self.smoothed_loss.append(smoothing
                                      * self.losses[i]
                                      + (1 - smoothing)
                                      * self.smoothed_loss[-1])

    def plot_result(self):
        """ Plot Loss / learning rate ratio
        """
        plt.title("Loss / learning rate")
        plt.subplot(211)
        plt.plot(self.lr_s)
        plt.xlabel('$iterations$')
        plt.grid()

        plt.subplot(212)
        plt.plot(self.lr_s, self.smoothed_loss)
        plt.xscale('log')
        plt.xlabel("$learning rate$")
        plt.grid()

        plt.show()

    def best_lr(self):
        """ Compute best lower and upper bound learning rate
            for Cyclic scheduler

            Returns
            -------
                lower_bound_lr: float
                    Lower bound value for cyclic lr
                upper_bound_lr:
                    Upper bound value for cyclic lr
        """
        self._smooth_losses()
        best_lr = self.lr_s[np.argmin(self.smoothed_loss)]
        upper_bound_lr = best_lr  # / 10.
        lower_bound_lr = upper_bound_lr / 6.
        return (lower_bound_lr, upper_bound_lr)


class BaseScheduler(ABC):
    """ Base learning rate scheduler

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

        Attributes
        ----------
            stepsize: int
                Number of samples for one cycle
    """
    def __init__(self, min_lr, max_lr, scaler, data_len, n_times):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.scaler = scaler
        self.data_len = data_len
        self.n_times = n_times
        self.stepsize = self.n_times * self.data_len

    def relative(self, it, stepsize):
        """ Compute coefficient for learning rate at current iteration

            Parameters
            ----------
                it: int
                    Current iteration
                stepsize: int
                    Number of samples for half of the cycle

            Returns
            -------
                coeff: float
                    Coefficient for learning rate at current iteration
        """
        cycle = np.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        if self.scaler.cycle_mode == 'cycle':
            coeff = max(0, (1 - x)) * self.scaler(cycle)
        elif self.scaler.cycle_mode == 'iteration':
            coeff = max(0, (1 - x)) * self.scaler(it)
        else:
            raise NotImplementedError
        return coeff

    def compute_lr(self, it, stepsize):
        """ Compute learning rate at current iteration

            Returns
            -------
                lr: float
                    Learning rate
        """
        lr = self.min_lr + (self.max_lr - self.min_lr) * self.relative(it, self.stepsize)
        return lr
