import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class Scaler(object):
    """ Class to handle creation of new scalers and 
        check their functionality.

        Parameters
        ----------
            model: str
                Shows how to update scale of learning rate

            function: callable
                Any that return learning rate at each iteration

            *args, **kwargs
                Additional parameters of sclaer ('function')
    """

    def __init__(self, mode, function, *args, **kwargs):
        self.mode = mode
        self.function = function
        self.fun_args = args
        self.fun_kwargs = kwargs

    @property
    def cycle_mode(self):
        return self.mode

    def plot(self, min_lr=1e-6, max_lr=0.1, stepsize=10, n_iterations=1e3):
        """ Plot function for easier debugging.

            Parameters
            ----------
                min_lr: float
                    Minimum learning rate for debugging scale function

                max_lr: float
                    Max learning rate for debugging scale function

                stepsize:
                    Step (half of full cycle) of changing lr

                n_iterations:
                    Number of iterations to check
        """
        iterations = np.arange(n_iterations)
        result = deque(maxlen=len(iterations))

        for it in iterations:
            cycle = np.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            if self.mode == "cycle":
                coeff = max(0, (1 - x)) * self.function(cycle, self.fun_args, self.fun_kwargs)
            elif self.mode == "iteration":
                coeff = max(0, (1 - x)) * self.function(it, self.fun_args, self.fun_kwargs)
            current_lr = min_lr + (max_lr - min_lr) * coeff
            result.append(current_lr)
        plt.plot(iterations, result)
        plt.show()

    def __call__(self, x):
        """ Apply scaler.
        """
        return self.function(x, self.fun_args, self.fun_kwargs)


def exp_range(x, *args, **kwargs):
    result = np.power(gamma, x)
    return result


def triangular(x, *args, **kwargs):
    return 1


def triangular2(x, *args, **kwargs):
    result = 1. / (np.power(2, (x - 1)))
    return result


if __name__ == "__main__":
    gamma = 2.
    scaler1 = Scaler("iteration", exp_range, gamma=1.1)
    scaler2 = Scaler('cycle', triangular)
    scaler3 = Scaler('cycle', triangular2)

    scaler1.plot(stepsize=2, n_iterations=100)
    scaler2.plot()
    scaler3.plot()
