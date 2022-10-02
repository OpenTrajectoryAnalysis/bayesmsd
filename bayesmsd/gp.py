"""
Implementation of Gaussian process logic

This module is mostly for internal use of the `bayesmsd` package. It covers the
definition of (stationary) Gaussian processes in terms of their MSD, and the
associated likelihood function `ds_logL`.

Gaussian processes can also serve as generative model, i.e. one can sample
trajectories with a given MSD curve. This is implemented in the `generate`
function in this module.

See also
--------
bayesmsd, ds_logL, generate
"""
import itertools
import inspect

import numpy as np
from scipy import linalg

from noctiluca import Trajectory, TaggedSet
from noctiluca import parallel

################## Covariance in terms of MSD #################################

def msd2C_ss0(msd, ti):
    """
    0th order steady state covariance from MSD. For internal use.

    Parameters
    ----------
    msd : np.ndarray
    ti : np.ndarray, dtype=int
        times at which there are data in the trajectory

    Returns
    -------
    np.ndarray
        the covariance matrix

    See also
    --------
    msd2C_fun
    """
    return 0.5*( msd[-1] - msd[np.abs(ti[:, None] - ti[None, :])] )

def msd2C_ss1(msd, ti):
    """
    1st order steady state covariance from MSD. For internal use.

    Parameters
    ----------
    msd : np.ndarray
    ti : np.ndarray, dtype=int
        times at which there are data in the trajectory

    Returns
    -------
    np.ndarray
        the increment covariance matrix

    See also
    --------
    msd2C_fun
    """
    return 0.5*(  msd[np.abs(ti[1:, None] - ti[None,  :-1])] + msd[np.abs(ti[:-1, None] - ti[None, 1:  ])]
                - msd[np.abs(ti[1:, None] - ti[None, 1:  ])] - msd[np.abs(ti[:-1, None] - ti[None,  :-1])] )

def msd2C_fun(msd, ti, ss_order):
    """
    msd2C for MSDs expressed as functions / non-integer times.

    Parameters
    ----------
    msd : callable, use the `MSDfun` decorator
        note that for ``ss_order == 0`` we expect ``msd(np.inf)`` to be
        well-defined.
    ti : np.ndarray, dtype=int
        times at which there are data in the trajectory
    ss_order : {0, 1}
        steady state order. See module documentation.

    Returns
    -------
    np.ndarray
        covariance matrix

    See also
    --------
    MSDfun, bayesmsd
    """
    if ss_order == 0:
        return 0.5*( msd(np.inf) - msd(ti[:, None] - ti[None, :]) )
    elif ss_order == 1:
        return 0.5*(  msd(ti[1:, None] - ti[None,  :-1]) + msd(ti[:-1, None] - ti[None, 1:  ])
                    - msd(ti[1:, None] - ti[None, 1:  ]) - msd(ti[:-1, None] - ti[None,  :-1]) )
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")

################## Gaussian Process likelihood ###############################

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)
GP_verbosity = 1
def GP_vprint(v, *args, **kwargs):
    if GP_verbosity >= v: # pragma: no cover
        print("[bayesmsd.GP]", (v-1)*'--', *args, **kwargs)

class BadCovarianceError(RuntimeError):
    pass

def _GP_core_logL(C, x):
    # Implementation notes
    # - (slogdet, solve) is faster than eigendecomposition (~3x)
    # - don't check positive definiteness here, that should be done before—if
    #   necessary; c.f. Fit.constraint_Cpositive
    with np.errstate(under='ignore'):
        s, logdet = np.linalg.slogdet(C)

    # Since we need logdet anyways, this is a check that we can do for free
    if s <= 0: # pragma: no cover
        raise BadCovarianceError("Covariance matrix has negative determinant: slogdet = ({}, {})".format(s, logdet))
        
    try:
        xCx = x @ linalg.solve(C, x, assume_a='pos')
    except (FloatingPointError, linalg.LinAlgError) as err: # pragma: no cover
        # what's the problematic case that made me insert this?
        # --> can (probably?) happen in numerical edge cases. Should usually be
        #     prevented by `Fit` in the first place
        GP_vprint(3, f"Problem when inverting covariance; slogdet = ({s}, {logdet})")
        GP_vprint(3, type(err), err)
        raise BadCovarianceError("Inverting covariance did not work")

    return -0.5*(xCx + logdet) - len(C)*LOG_SQRT_2_PI

def GP_logL(trace, ss_order, msd, mean=0):
    """
    Gaussian process likelihood for a given trace

    Parameters
    ----------
    trace : (T,) np.ndarray
        the data. Should be recorded at constant time lag; missing data are
        indicated with ``np.nan``.
    ss_order : {0, 1}
        steady state order; see module documentation.
    msd : np.ndarray
        the MSD defining the Gaussian process, evaluated up to (at least) ``T =
        len(trace)``.
    mean : float
        the first moment of the Gaussian process. For ``ss_order == 0`` this is
        the mean (i.e. same units as the trajectory), for ``ss_order == 1``
        this is the mean of the increment process, i.e. has units trajectory
        per time.

    Returns
    -------
    float
        the log-likelihood

    See also
    --------
    ds_logL
    """
    ti = np.nonzero(~np.isnan(trace))[0]
    
    if ss_order == 0:
        X = trace[ti] - mean
        C = msd2C_ss0(msd, ti)
    elif ss_order == 1:
        X = np.diff(trace[ti]) - mean*np.diff(ti)
        C = msd2C_ss1(msd, ti)
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")
    
    return _GP_core_logL(C, X)

def _GP_logL_for_parallelization(params):
    # just unpacking arguments
    return GP_logL(*params)

def ds_logL(data, ss_order, msd_ms):
    """
    Gaussian process likelihood on a data set.

    Parameters
    ----------
    data : a `TaggedSet` of `Trajectory`
    ss_order : {0, 1}
        steady state order; see module documentation.
    msd_ms : list of tuples (msd, mean)
        this should be a list with one entry for each spatial dimension of the
        data set. The first entry of each tuple is a function, ideally
        decorated with `MSDfun`. The second should be a float giving the
        mean/drift for the process (often zero).

    Returns
    ------
    float
        the total log-likelihood of the data under the Gaussian process
        specified by the given MSD.

    Notes
    -----
    Parallel-aware (unordered). However, in practice I find that it is usually
    faster to parallelize runs over multiple data sets / with different
    parameters, if possible. In a benchmarking run for the internal
    parallelization in this function I saw no observable benefit of running on
    more than 5 cores (presumably due to overhead in moving data to the
    workers). 

    See also
    --------
    Fit, noctiluca.parallel
    """
    # msd_ms : list of tuples: d*[(msd, m)]
    # Implementation note: it does *not* make sense to allow a single tuple for
    # msd_ms, because broadcasting to spatial dimensions would require a
    # prefactor of 1/d in front of the component MSDs. It is easier (more
    # readable) to do this in the MSD implementation than keeping track of it
    # here
    d = data.map_unique(lambda traj : traj.d)
    if len(msd_ms) != d: # pragma: no cover
        raise ValueError(f"Dimensionality of MSD ({len(msd_ms)}) != dimensionality of data ({d})")
        
    # Convert msd to array, such that parallelization works
    Tmax = max(map(len, data))
    dt = np.arange(Tmax)
    array_msd_ms = [[msd(dt), m] for msd, m in msd_ms]
    if ss_order == 0:
        for dim, (msd, m) in enumerate(msd_ms):
            array_msd_ms[dim][0] = np.append(array_msd_ms[dim][0], msd(np.inf))

    job_iter = itertools.chain.from_iterable((itertools.product((traj[:][:, dim] for traj in data),
                                                                [ss_order], [msd], [m],
                                                               )
                                              for dim, (msd, m) in enumerate(array_msd_ms)
                                             ))
    
    return np.sum(list(parallel._umap(_GP_logL_for_parallelization, job_iter)))

################## Generative model ##########################################

def generate(msd_def, T, n=1):
    """
    Sample trajectories from a given MSD / fitparameters

    Parameters
    ----------
    msd_def : 2-tuple
        one of two options:
         + ``(fit, res)`` where `!fit` is an instance of `Fit` (e.g. one of the
           classes from `lib`) and ``res`` is the result of a previous run of
           `!fit` (i.e. a dict with ``'params'`` and ``'logL'`` entries; the
           latter is not used here)
         + ``(msdfun, ss_order, d)`` where ``msdfun`` is a callable (ideally
           wrapped with the `MSDfun` decorator), ``ss_order in {0, 1}`` (see
           module doc), and `!d` is the spatial dimension of the trajectories
    T : int
        the length of the trajectories to be generated, in frames
    n : int, optional
        the number of trajectories to generate

    Returns
    -------
    TaggedSet
        a data set containing the drawn trajectories
    """
    from .fit import Fit # bad style, but we're just instance checking
    if isinstance(msd_def[0], Fit):
        fit, res = msd_def
        ss_order = fit.ss_order

        msdm = fit.params2msdm(res['params'])
        ms = np.array([m for _, m in msdm])
        Cs = [msd2C_fun(msd, np.arange(T), ss_order=ss_order) for msd, _ in msdm]
        Ls = [linalg.cholesky(C, lower=True) for C in Cs]
        steps = np.array([L @ np.random.normal(size=(T-ss_order, n)) for L in Ls])
        steps = np.swapaxes(steps, 0, 2) # (n, T, d)
    else:
        msdfun, ss_order, d = msd_def
        ms = np.zeros(d) # could implement this at some point, but so far it seems useless

        C = msd2C_fun(msdfun, np.arange(T), ss_order=ss_order) / d
        L = linalg.cholesky(C, lower=True)
        steps = L @ np.random.normal(size=(n, T-ss_order, d))
        # Note that matmul acts on dimensions (-1, -2) for the two arguments,
        # NOT (-1, 0) as one might assume. In this case this is quite handy,
        # since we want the (n, T, d) order of dimensions.

    if ss_order == 0:
        return TaggedSet((Trajectory(mysteps + ms[None, :]) for mysteps in steps), hasTags=False)
    elif ss_order == 1:
        steps = np.insert(steps, 0, -ms[None, :], axis=1) # all trajectories (via cumsum) start at zero
        return TaggedSet((Trajectory(np.cumsum(mysteps + ms[None, :], axis=0)) for mysteps in steps), hasTags=False)
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")
