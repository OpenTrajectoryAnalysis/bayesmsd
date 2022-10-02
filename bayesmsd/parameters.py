from abc import ABCMeta, abstractmethod

import numpy as np

class Parameter():
    def __init__(self,
                 bounds=(-np.inf, np.inf),
                 fix_to=None,
                 bracket_strategy=None,
                 ):
        self.bounds = np.array(bounds)
        self.fix_to = fix_to
        if bracket_strategy is None:
            self.bracket_strategy = self.suggest_bracket_strategy()
        else:
            self.bracket_strategy = bracket_strategy

    def suggest_bracket_strategy(self):
        n_bounds_inf = np.sum(np.isinf(self.bounds))
        if n_bounds_inf == 0:
            return BracketStrategy.Bounded(self)
        elif n_bounds_inf == 1 and self.bounds[0]*self.bounds[1] > 0:
            return BracketStrategy.Multiplicative(self)
        else:
            return BracketStrategy.Exponential(self)

class BracketStrategy: # this is just a namespace for everything to do with
                       # bracket strategies
    class NonIdentifiableException(Exception):
        pass

    class ABC(metaclass=ABCMeta):
        def __init__(self, parameter):
            self.param = parameter
            self.max_steps = 10

        def get_step(self, pe, n):
            if np.abs(n) > self.max_steps:
                raise BracketStrategy.NonIdentifiableException

            return self._get_step(pe, n)

        @abstractmethod
        def _get_step(self, pe, n):
            raise NotImplementedError

    class Bounded(ABC):
        def __init__(self, parameter, n_steps=10):
            super().__init__(self, parameter)

            diff_bounds = self.param.bounds[1]-self.param.bounds[0]
            assert np.isfinite(diff_bounds)

            self.step = diff_bounds / n_steps

        def _get_step(self, pe, n):
            return pe + n*self.step

    class Multiplicative(ABC):
        def __init__(self, parameter, factor=2):
            super().__init__(self, parameter)
            assert self.params.bounds[0]*self.params.bounds[1] > 0

            self.factor = factor

        def _get_step(self, pe, n):
            return pe*(self.factor**n)

    class Exponential(ABC):
        def __init__(self, parameter, step=1):
            super().__init__(self, parameter)
            self.step = step

        def _get_step(self, pe, n):
            return pe + np.sign(n)*self.step**np.abs(n)
