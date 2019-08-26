import keras
import keras.backend as K
import numpy as np

from base_pipeline import LRFinderBase, BaseScheduler

# TODO: ------------------------>
#                               |
#                               |
#                               |
#                               \/


class LRFinderKeras(keras.callback.Callback, LRFinderBase):
        """ Base class for finding best learning rate

        Parameters
        ----------
            model: keras.Model, torch.Module
                Model for witch you want to find optimal lr
            min_lr: float
                Lower bound of learning rate scheduler
            max_lr: float
                Upper bound of learin rate scheduler
            steps_per_epoch: int
                Show how much iteration should make optimizer every epoch

        Attributes
        ----------
            losses: list
                Loss values at each iteration
            lr_s: list
                Learning rate values at each iteration
            data_len: int
                Number of batches in dataset
            iteration: int
                Number of current iteration
    """

    def __init__(self, model, min_lr, max_lr, loader, steps_per_epoch=None,
                 epochs=None):
        super().__init__()
        self.model = model
        self.loader = loader
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0

    def _run_epoch(self):
        if isinstance(self.loader, (tuple, list, np.ndarray)):
            self.model.fit(x=self.loader[0], y=self.loader[1])
        else:
            self.model.fit_generator(self.loader)
    

    def run(self, criterion, optimizer):
        self.model.compile(optimizer, criterion)
        self._run_epoch()
        


    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr-self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        keras.backend.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        self.iteration += 1
        keras.backend.set_value(self.model.optimizer.lr, self.clr())


class KerasScheduler(BaseScheduler, keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, scaler, data_len, n_times):
        self.it = 0
        super().__init__(min_lr, max_lr, scaler, datas_len, n_times)
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr is not None:
            self.min_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size

    def on_train_begin(self, logs={}):
        if self.it == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.compute_lr(self.it, self.stepsize))

    def on_batch_end(self, epoch):
        self.it += 1
        K.set_value(self.model.optimizer.lr,
                    self.compute_lr(self.it, self.step_size))
