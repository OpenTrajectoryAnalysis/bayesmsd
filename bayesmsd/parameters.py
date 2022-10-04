from abc import ABCMeta, abstractmethod

import numpy as np

class Parameter():
    def __init__(self,
                 bounds=(-np.inf, np.inf),
                 fix_to=None,
                 linearization=None,
                 max_linearization_moves=(10, 10),
                 ):
        self.bounds = np.array(bounds)
        self.fix_to = fix_to

        if linearization is None:
            self.linearization = self.suggest_linearization()
        else:
            self.linearization = linearization

        self.linearization.param = self
        self.max_linearization_moves = max_linearization_moves

    # Copying
    # TL;DR: deepcopy()-ing Parameter objects works fine
    # copying `Parameter`s in principle could be problematic, since they
    # recursively contain a reference to themselves:
    # ``param.linearization.param is param``. Fortunately, copy.deepcopy() is
    # prepared for this and correctly copies that self-reference as
    # self-reference to the new object.

    def suggest_linearization(self):
        n_bounds_inf = np.sum(np.isinf(self.bounds))
        if n_bounds_inf == 0:
            return Linearize.Bounded(self)
        elif n_bounds_inf == 1 and self.bounds[0]*self.bounds[1] > 0:
            return Linearize.Multiplicative(self)
        else:
            return Linearize.Exponential(self)

class Linearize: # this is just a namespace
    class ABC(metaclass=ABCMeta):
        def __init__(self):
            # self.param should be set manually after initialization; this is
            # necessary, since the Linearize instance should be passed to
            # self.param upon initialization (Parameter.__init__()).
            self.param = None

        @abstractmethod
        def from_linear(self, pe, n): # should be vectorized in n
            raise NotImplementedError

        @abstractmethod
        def to_linear(self, pe, x): # should be vectorized in x
            raise NotImplementedError

        def move(self, pe, x, n):
            return self.from_linear(pe, self.to_linear(pe, x) + n)

        def distance(self, pe, x, y):
            return np.abs(self.to_linear(pe, x) - self.to_linear(pe, y))

        def mean(self, pe, x):
            return self.from_linear(pe, np.mean(self.to_linear(pe, x)))

    class Bounded(ABC):
        def __init__(self, n_steps=10):
            super().__init__()
            self.n_steps = n_steps

        def from_linear(self, pe, n):
            bounds = self.param.bounds
            return bounds[0] + (bounds[1]-bounds[0])*n/self.n_steps

        def to_linear(self, pe, x):
            bounds = self.param.bounds
            return self.n_steps * (x - bounds[0])/(bounds[1]-bounds[0])

    class Multiplicative(ABC):
        def __init__(self, factor=2):
            super().__init__()
            self.log_factor = np.log(factor)

        def from_linear(self, pe, n):
            return pe*np.exp(log_factor*n)

        def to_linear(self, pe, x):
            return np.log(x/pe)/log_factor

    class Exponential(ABC):
        def __init__(self, step=1, base=2):
            super().__init__()
            self.step = step
            self.log_base = np.log(base)

        def from_linear(self, pe, n):
            return pe + np.sign(n)*self.step*(np.exp(self.log_base*np.abs(n)) - 1)

        def to_linear(self, pe, x):
            return np.sign(x-pe) * np.log(1 + np.abs(x-pe)/self.step)/self.log_base
