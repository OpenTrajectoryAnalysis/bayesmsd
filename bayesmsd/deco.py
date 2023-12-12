"""
Decorators to use when defining custom MSD functions

There is some functionality that should be common to all/most functions to be
used in `Fit.params2msdm` implementations. This module provides decorators
implementing this common functionality. Specifically, these fall into two
categories:

+ technical convenience, i.e. making sure that the argument is properly cast to
  a numpy array and dealing with ``dt <= 0`` (i.e. ``MSD(0) == 0`` should be
  true for any MSD function for technical reasons, even if ``lim_{dt-->0}
  MSD(dt) != 0``). This is taken care of by the `MSDfun` decorator.
+ contributions to realistic MSD curves due to common imaging artifacts, i.e.
  localization error and motion blur. These are implemented in the `imaging`
  decorator.

Taking both together, the definition for a simple powerlaw MSD function would
take the following form:

>>> G, alpha = ... # from input
...
... @bayesmsd.deco.MSDfun
... @bayesmsd.deco.imaging(noise2=..., f=..., alpha0=alpha)
... def powerlawMSD(dt, G=G, alpha=alpha):
...     return G*dt**alpha

Note that `MSDfun` should always be the topmost decorator, since it handles
user input. The `imaging` decorator first adds motion blur---parametrized by
the fractional exposure `!f` which is given experimentally and the early time
(true) MSD scaling `!alpha0`, which should be determined from the true MSD
curve---and then adds the (squared) localization error `!noise2`. Overall,
there is thus just one additional free parameter, namely the localization
error.

Note how we pass ``G`` and ``alpha`` (which are defined in the surrounding
namespace) to the function as default arguments. This gets the scoping correct,
since default arguments are evaluated at definition time (as opposed to just
using the externally defined variables, whose values might change between
definition and execution of the function).

See also
--------
bayesmsd
"""
import inspect
import functools

import numpy as np

def method_verbosity_patch(meth):
    """
    (internal) Decorator for class methods, temporarily changing verbosity
    """
    @functools.wraps(meth)
    def wrapper(self, *args, verbosity=None, **kwargs):
        old_verbosity = self.verbosity
        if verbosity is not None:
            self.verbosity = verbosity
        try:
            return meth(self, *args, **kwargs)
        finally:
            self.verbosity = old_verbosity

    return wrapper
        
def MSDfun(fun):
    """
    Decorator for MSD functions

    This is a decorator to use when implementing `Fit.params2msdm`. It takes
    over some of the generic polishing: it assumes that the decorated function
    has the signature ``function(np.array) --> np.array`` and

    - ensures that the argument is cast to an array if necessary (such that you
      can then also call ``msd(5)`` instead of ``msd(np.array([5]))``
    - ensures that ``dt > 0`` by taking an absolute value and setting
      ``msd[dt==0] = 0`` without calling the wrapped function. You can thus
      ignore the ``dt == 0`` case in implementing an MSD function. Note that
      ``msd[dt==0] = 0`` should always be true. However, e.g. in the case of
      localization error, we might have ``lim_{Δt-->0} MSD(Δt) = 2σ² != 0``.

    See also
    --------
    Fit <bayesmsd.fit.Fit>, imaging
    """
    def msdfun(dt, **kwargs):
        # Preproc
        dt = np.abs(np.asarray(dt))
        was_scalar = len(dt.shape) == 0
        if was_scalar:
            dt = np.array([dt])
        
        # Calculate non-zero dts and set zeros
        msd = np.empty(dt.shape)
        ind0 = dt == 0
        msd[~ind0] = fun(dt[~ind0], **kwargs)
        msd[ind0] = 0
        
        # Postproc
        if was_scalar:
            msd = msd[0]
        return msd

    # Assemble a useful docstring
    try:
        fun_kwargstring = fun._kwargstring
    except AttributeError:
        params = inspect.signature(fun).parameters
        arglist = list(params)
        fun_kwargstring = ', '.join(str(params[key]) for key in arglist[1:])
    msdfun.__doc__ = f"\nfull signature: msdfun(dt, {fun_kwargstring})"
    msdfun._kwargstring = fun_kwargstring

    return msdfun

def imaging(noise2=0, f=0, alpha0=1):
    """
    Add imaging artifacts (localization error & motion blur) to MSDs.

    This decorator should be used when defining MSD functions for SPT
    experiments, to add artifacts due to the imaging process. These are a)
    localization error and b) motion blur, caused by finite exposure times. Use
    this decorator after `MSDfun`, like so:

    >>> @bayesmsd.deco.MSDfun
    ... @bayesmsd.deco.imaging(...)
    ... def msd(dt):
    ...     ...

    Parameters
    ----------
    noise2 : float >= 0
        the variance (σ²) of the Gaussian localization error to add
    f : float, 0 <= f <= 1
        the exposure time as fraction of the frame time.
    alpha0 : float, 0 <= alpha0 <= 2 or 'auto'
        the effective short time scaling exponent. Set to ``'auto'`` to
        determine from finite differences of the "raw" MSD around Δt=f.
    Notes
    -----
    ``f = 0`` is ideal stroboscopic illumination, i.e. no motion blur.
    Accordingly, the value of ``alpha0`` is not used in this case.

    See also
    --------
    MSDfun
    """
    def decorator(msdfun):
        def wrap(dt, noise2=noise2, f=f, alpha0=alpha0, **kwargs):
            if f == 0:
                return msdfun(dt, **kwargs) + 2*noise2

            if alpha0 == 'auto':
                t0 = np.array([f])
                t1 = 1.001*t0
                alpha0 = np.log(msdfun(t1, **kwargs) / msdfun(t0, **kwargs)) / np.log(t1/t0)

            a = alpha0
            B = msdfun(np.array([f]), **kwargs)[0] / ( (a+1)*(a+2) )

            # dt is in (0, inf], so we have to be careful with inf (but not 0)
            phi = f/dt
            b = np.empty(len(phi), dtype=float)
            ind = phi > 0
            phi = phi[ind]
            b[ind] = ( (1+phi)**(a+2) + (1-phi)**(a+2) - 2 ) / ( phi**2 * (a+1) * (a+2) )
            b[~ind] = 1

            out = b*msdfun(dt, **kwargs) - 2*B + 2*noise2

            # In some edge cases, the MSD can get close to zero, i.e.
            # numerically potentially negative.
            if np.any(out < 0): # pragma: no cover
                if np.min(out) > -1e-10:
                    out[out < 0] = 0
                else:
                    raise ValueError("MSD became significantly (by more than 1e-10) negative during imaging correction. This is probably due to a bug; please report it.")

            return out

        # Assemble a useful docstring
        try:
            fun_kwargstring = msdfun._kwargstring
        except AttributeError:
            params = inspect.signature(msdfun).parameters
            arglist = list(params)
            fun_kwargstring = ', '.join(str(params[key]) for key in arglist[1:])

        params = inspect.signature(wrap).parameters
        arglist = list(params)
        wrap_kwargstring = ', '.join([str(params[key]) for key in arglist if key not in ['dt', 'kwargs']])

        wrap._kwargstring = ', '.join([wrap_kwargstring, fun_kwargstring])
        return wrap
    return decorator
