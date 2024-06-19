import numpy as np
from scipy import linalg

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

################## Gaussian Process likelihood ###############################

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)

class BadCovarianceError(RuntimeError):
    pass

def _core_logL(C, x):
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
        vprint(3, f"Problem when inverting covariance; slogdet = ({s}, {logdet})")
        vprint(3, type(err), err)
        raise BadCovarianceError("Inverting covariance did not work")

    return -0.5*(xCx + logdet) - len(C)*LOG_SQRT_2_PI

def logL(trace, ss_order, msd, mean=0):
    """
    Gaussian process likelihood for a given trace

    Parameters
    ----------
    trace : (T,) np.ndarray
        the data. Should be recorded at constant time lag; missing data are
        indicated with ``np.nan``.
    ss_order : {0, 0.5, 1}
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
    
    if ss_order < 1:
        X = trace[ti] - mean
        C = msd2C_ss0(msd, ti)

        if ss_order == 0.5: # condition on first data point
            mu = X[0] * C[1:, 0]/C[0, 0]
            X = X[1:] - mu
            C = C - C[:, [0]]*C[[0], :]/C[0, 0]
            C = C[1:, 1:]

    elif ss_order == 1:
        X = np.diff(trace[ti]) - mean*np.diff(ti)
        C = msd2C_ss1(msd, ti)

    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")

    return _core_logL(C, X)
