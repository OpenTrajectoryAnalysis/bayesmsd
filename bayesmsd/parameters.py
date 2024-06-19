from abc import ABCMeta, abstractmethod

import numpy as np

class Parameter():
    """
    Definition of one fit parameter

    Parameters
    ----------
    bounds : (lower, upper), optional
        the bounds to impose on this parameter
    fix_to : float, str, callable, or None; optional
        whether and how to tie the value of this parameter to another one. Can be

         + float constant: parameter will be fixed to this value
         + name of another parameter: parameter will always be the same as the other one
         + callable: signature should be ``fun(params)->float``, where
           ``params`` will be a dict with all the already determined parameter
           values.
         + ``None``: keep this parameter independent.

    linearization : Linearize.ABC, optional
        specifies a coordinate transform to a space where taking steps of 1 is
        (in some sense) reasonable. If unspecified, use
        `suggest_linearization()` to find a suitable transform.
    max_linearization_moves : 2-tuple of int, optional
        how far to move in the linearized space before considering the
        parameter unidentifiable. These are the maximum bounds in which the
        `Profiler <bayesmsd.profiler.Profiler>` will explore the posterior over
        this parameter.

    Notes
    -----
    Copying a `!Parameter`: use ``copy.deepcopy()``. Through the
    `!linearization`, a `Parameter` contains a self-reference; ``deepcopy()``
    is smart enough to detect this and copy properly, i.e. as self-reference to
    the new object.
    """
    def __init__(self,
                 bounds=(-np.inf, np.inf),
                 fix_to=None,
                 linearization=None,
                 max_linearization_moves=(10, 10),
                 ):
        self.bounds = np.array(bounds, dtype=float)
        self.fix_to = fix_to

        if linearization is None:
            self.linearization = self.suggest_linearization()
        else:
            self.linearization = linearization

        self.linearization.param = self
        self.max_linearization_moves = max_linearization_moves

    def suggest_linearization(self):
        """
        Suggest a good linearization to use

        Returns
        -------
        Linearize.ABC
        """
        n_bounds_inf = np.sum(np.isinf(self.bounds))
        if n_bounds_inf == 0:
            return Linearize.Bounded()
        elif n_bounds_inf == 1 and self.bounds[0]*self.bounds[1] > 0:
            return Linearize.Multiplicative()
        else:
            return Linearize.Exponential()

class Linearize: # this is just a namespace
    class ABC(metaclass=ABCMeta):
        """
        Abstract base class for linearizations

        Attributes
        ----------
        param : Parameter
            the `Parameter` this linearization applies to. This is set by the
            `Parameter` initializer that instances of this class should be
            handed to.
        """
        def __init__(self):
            self.param = None

        @abstractmethod
        def from_linear(self, pe, n): # should be vectorized in n
            """
            Convert from linear space to "real" parameter space

            Parameters
            ----------
            pe : float
                current point estimate
            n : float
                coordinate in the linear space

            Returns
            -------
            x : float
                coordinate in original parameter space

            See also
            --------
            to_linear

            Notes
            -----
            When subclassing, ensure that ``from_linear(pe, to_linear(pe, x))
            == x`` and ``to_linear(pe, from_linear(pe, n)) == n`` for generic
            ``x`` and ``n``.

            `from_linear` should be vectorized in ``n``, i.e. work properly on
            arrays.
            """
            raise NotImplementedError # pragma: no cover

        @abstractmethod
        def to_linear(self, pe, x): # should be vectorized in x
            """
            Convert from "real" parameter space to linear space

            Parameters
            ----------
            pe : float
                current point estimate
            x : float
                coordinate in original parameter space

            Returns
            -------
            n : float
                coordinate in the linear space

            See also
            --------
            from_linear

            Notes
            -----
            When subclassing, ensure that ``from_linear(pe, to_linear(pe, x))
            == x`` and ``to_linear(pe, from_linear(pe, n)) == n`` for generic
            ``x`` and ``n``.

            `to_linear` should be vectorized in ``x``, i.e. work properly on
            arrays.
            """
            raise NotImplementedError # pragma: no cover

        def move(self, pe, x, n):
            """
            Move a given position ``x`` in real space by ``n`` units in the
            linearized space.
            """
            return self.from_linear(pe, self.to_linear(pe, x) + n)

        def distance(self, pe, x, y):
            """
            Calculate linearized distance between ``x`` and ``y``
            """
            return np.abs(self.to_linear(pe, x) - self.to_linear(pe, y))

        def mean(self, pe, x):
            """
            Calculate linearized mean of ``x`` (an array)
            """
            return self.from_linear(pe, np.mean(self.to_linear(pe, x)))

    class Bounded(ABC):
        """
        Linearization to use when both bounds are finite

        This linearization is a simple rescaling of the "real" parameter space,
        such that the interval between the two bounds corresponds to `!n_steps`
        units.

        ``n = 0`` corresponds to the lower bound.

        Parameters
        ----------
        n_steps : int
            in how many bins to subdivide the interval between the bounds
        """
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
        """
        Log-spacing; appropriate for strictly positive/negative parameters

        Each step multiplies the parameter by a constant factor. Clearly this
        will never cross zero, so appropriate only for strictly positive or
        negative parameters.

        ``n = 0`` corresponds to the point estimate.

        Parameters
        ----------
        factor : float
            the factor to apply at each step
        """
        def __init__(self, factor=2):
            super().__init__()
            self.log_factor = np.log(factor)

        def from_linear(self, pe, n):
            return pe*np.exp(self.log_factor*n)

        def to_linear(self, pe, x):
            return np.log(x/pe)/self.log_factor

    class Exponential(ABC):
        """
        Take exponentially growing steps from the point estimate.

        This is similar to `Multiplicative` in its applicability to variables
        that might span several orders of magnitude, but is not restricted to
        parameters with definite sign.

        ``n = 0`` corresponds to the point estimate; ``|n| = 1`` corresponds to
        one `!step` from the point estimate; higher ``n`` grow exponentially
        with given `!base`.

        Parameters
        ----------
        step : float
            the step size at ``|n| = 1``.
        base : float
            the base to use for exponential growth.
        """
        def __init__(self, step=1, base=2):
            super().__init__()
            self.step = step
            self.log_base = np.log(base)

        def from_linear(self, pe, n):
            return pe + np.sign(n)*self.step*(np.exp(self.log_base*np.abs(n)) - 1)

        def to_linear(self, pe, x):
            return np.sign(x-pe) * np.log(1 + np.abs(x-pe)/self.step)/self.log_base
