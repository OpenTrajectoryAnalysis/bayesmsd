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
bayesmsd, ds_logL, generate, generate_dataset_like
"""
import itertools
import inspect

import numpy as np
from scipy import linalg

from noctiluca import Trajectory, TaggedSet
from noctiluca import parallel

################## Covariance in terms of MSD #################################

from .cython_imports import GP_msd2C_ss0 as msd2C_ss0
from .cython_imports import GP_msd2C_ss1 as msd2C_ss1

def msd2C_fun(msd, ti, ss_order):
    """
    msd2C for MSDs expressed as functions / non-integer times.

    Parameters
    ----------
    msd : callable, use the `MSDfun <bayesmsd.deco.MSDfun>` decorator
        note that for ``ss_order < 1`` we expect ``msd(np.inf)`` to be
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
    MSDfun <bayesmsd.deco.MSDfun>, bayesmsd
    """
    if ss_order < 1:
        return 0.5*( msd(np.inf) - msd(ti[:, None] - ti[None, :]) )
    elif ss_order == 1:
        return 0.5*(  msd(ti[1:, None] - ti[None,  :-1]) + msd(ti[:-1, None] - ti[None, 1:  ])
                    - msd(ti[1:, None] - ti[None, 1:  ]) - msd(ti[:-1, None] - ti[None,  :-1]) )
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")

################## Gaussian Process likelihood ###############################

class GP: # just a namespace

    from .cython_imports import GP_BadCovarianceError as BadCovarianceError
    from .cython_imports import GP_logL               as logL

    def _logL_for_parallelization(params):
        # just unpacking arguments
        return GP.logL(*params)

    @parallel.chunky('chunksize', -1)
    def ds_logL(data, ss_order, msd_ms):
        """
        Gaussian process likelihood on a data set.

        Parameters
        ----------
        data : a `TaggedSet` of `Trajectory`
        ss_order : {0, 0.5, 1}
            steady state order; see module documentation.
        msd_ms : list of tuples (msd, mean)
            this should be a list with one entry for each spatial dimension of the
            data set. The first entry of each tuple is a function, ideally
            decorated with `MSDfun`. The second should be a float giving the
            mean/drift for the process (often zero).
        chunksize : int
            controls chunk size for parallel processing, when running within
            `!noctiluca.Parallel` context. Set to ``< 0`` to prevent
            parallelization, ``0`` to submit everything into one process, ``>
            0`` for chunked processing

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

        # For parallelization we need MSD as array, not callable (not pickleable)
        # If we do this per trajectory (dimension, in fact), we can allow
        # different time lags for different trajectories; for this, we reserve
        # the meta entry 'Δt'; let's consider it unlikely that users would
        # unwittingly use 'Δt' otherwise (due to unicode Δ).
        dts = np.ones(len(data), dtype=float)
        for i, traj in enumerate(data):
            try:
                dts[i] = traj.meta['Δt']
            except KeyError:
                continue

        dts_u = np.unique(dts)
        Ts = {dt : 0 for dt in dts_u}
        for traj, dt in zip(data, dts):
            Ts[dt] = max(Ts[dt], len(traj))

        if ss_order == 1:
            msd_m_by_dt = {dt : [(msd(dt*np.arange(Ts[dt])), m*dt)
                                 for msd, m in msd_ms]
                           for dt in dts_u}
        else: # ss_order < 1
            msd_m_by_dt = {dt : [(msd(np.append(dt*np.arange(Ts[dt]), np.inf)), m)
                                 for msd, m in msd_ms]
                           for dt in dts_u}

        todo = len(data)*d*[None]
        for i, (traj, dt) in enumerate(zip(data, dts)):
            msd_m = msd_m_by_dt[dt]
            for dim, (msd, m) in enumerate(msd_m):
                trace = traj[:][:, dim]
                my_msd = msd[:len(trace)]
                if ss_order < 1:
                    my_msd = np.append(my_msd, msd[-1])

                todo[d*i+dim] = (trace, ss_order, my_msd, m)

        assert not any(td is None for td in todo) 
        return np.sum(list(parallel._map(GP._logL_for_parallelization, todo)))

################## Generative model ##########################################

def generate(msd_def, T, n=1):
    """
    Sample trajectories from a given MSD / fitparameters

    Parameters
    ----------
    msd_def : tuple
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
    if len(msd_def) == 2:
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

    if ss_order < 1:
        return TaggedSet((Trajectory(mysteps + ms[None, :]) for mysteps in steps), hasTags=False)
    elif ss_order == 1:
        steps = np.insert(steps, 0, -ms[None, :], axis=1) # all trajectories (via cumsum) start at zero
        return TaggedSet((Trajectory(np.cumsum(mysteps + ms[None, :], axis=0)) for mysteps in steps), hasTags=False)
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")

def generate_dataset_like(data, msd_def):
    """
    Create a dataset like the reference, but from given MSD.

    The returned TaggedSet contains trajectories that match the input dataset
    in number, length, and missing frames, but are sampled from the MSD defined
    by `!msd_def` (which is usually a fit to these data).

    Parameters
    ----------
    data : TaggedSet of Trajectories
        the input dataset. Note that the actual data will not be used, just
        meta info like number and length of trajectories, and missing frames
    msd_def : (fit, res) or (msd_fun, ss_order, d)
        the definition of the MSD to sample from; see `generate`.

    Returns
    -------
    TaggedSet of Trajectories

    Notes
    -----
    This function assumes N = 1 (single particle) for all trajectories; d
    (number of spatial dimensions) should be consistent across the dataset.
    """
    T = max(map(len, data))
    new_data = generate(msd_def, T, n=len(data))
    for (traj, tags), (new_traj, new_tags) in zip(data(giveTags=True),
                                                  new_data(giveTags=True)):
        new_traj.data = new_traj.data[:, :len(traj)]
        new_traj.data[:, np.any(np.isnan(traj.data), axis=(0, 2)), :] = np.nan
        new_tags |= tags

    return new_data
