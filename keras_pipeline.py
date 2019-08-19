import keras
from base_pipeline import LRFinderBase, BaseScheduler
import keras.backend as K

# TODO: ------------------------>
#                               |
#                               |
#                               |
#                               \/


class LRFinderKeras(keras.callback.Callback, LRFinderBase):
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None,
                 epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0

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
        super().__init__(min_lr, max_lr, scaler, data_len, n_times)
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
