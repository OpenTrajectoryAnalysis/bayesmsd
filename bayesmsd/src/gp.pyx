## cython: profiling=True
# ^ remove second # to compile for profiler
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
# ^ remove second # to compile for line_profiler
# ^^ But ATTENTION: line_profiler can massively distort runtime; e.g. for logL
#       it claims ~90% of time are spent *assembling* the covariance matrix,
#       only ~10% on computation; more realistic estimate from manual debugging
#       & profiling with %timeit is almost exactly the reverse: 10% on
#       assembling, 90% on computation

import numpy as np
cimport numpy as np
np.import_array()

import cython

from libc.stdlib cimport abs
from libc.math cimport log, fabs
from scipy.linalg.cython_blas cimport dtrsm, ddot
from scipy.linalg.cython_lapack cimport dpotrf

from ..src.gp_py import BadCovarianceError

ctypedef double FLOAT_t
ctypedef unsigned long SIZE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void msd2C_ss0(FLOAT_t[::1]    msd,
                    SIZE_t[::1]     ti,
                    FLOAT_t[:, ::1] C,
                   ):
    """
    0th order steady state covariance from MSD.

    Parameters
    ----------
    msd : (T,) memoryview / array
        msd[T-1] should be MSD(∞)
    ti : (N,) memoryview / array (integer)
        times at which there are data in the trajectory; ti[n] < T-1 forall n
    C : (N, N) memoryview
        will be overwritten with the covariance matrix; only upper triangular
        part and diagonal (i.e. FORTRAN lower triangular)
    """
    cdef SIZE_t m, n, T=msd.shape[0], maxti=0

    for m in range(ti.shape[0]):
        if ti[m] > maxti:
            maxti = ti[m]

    assert C.shape[0] >= ti.shape[0]
    assert C.shape[1] >= ti.shape[0]
    assert T >= maxti+1

    for m in range(ti.shape[0]):
        C[m, m] = 0.5*msd[T-1]
        for n in range(m+1, ti.shape[0]):
            C[m, n] = 0.5*( msd[T-1] - msd[abs(ti[m] - ti[n])] )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void msd2C_ss1(FLOAT_t[::1]    msd,
                    SIZE_t[::1]     ti,
                    FLOAT_t[:, ::1] C,
                   ):
    """
    1st order steady state covariance from MSD. For internal use.

    Parameters
    ----------
    msd : (T,) memoryview / array
    ti : (N,) memoryview / array (integer)
        times at which there are data in the trajectory; ti[n] < T forall n
    C : (N-1, N-1) memoryview
        will be overwritten with the covariance matrix; only upper triangular
        part and diagonal (i.e. FORTRAN lower triangular)
    """
    cdef SIZE_t m, n, maxti=0

    for m in range(ti.shape[0]):
        if ti[m] > maxti:
            maxti = ti[m]

    assert C.shape[0] >= ti.shape[0]-1
    assert C.shape[1] >= ti.shape[0]-1
    assert msd.shape[0] > maxti

    # ti should be ordered
    # populate only upper triangle
    # benchmarking showed no speedup for precalculating msd[ti-tj]
    for m in range(ti.shape[0]-1):
        C[m, m] = msd[abs(ti[m+1]-ti[m])]
        for n in range(m+1, ti.shape[0]-1):
            C[m, n] = 0.5*(  msd[abs(ti[n+1]-ti[m])]
                           + msd[abs(ti[n]  -ti[m+1])]
                           - msd[abs(ti[n+1]-ti[m+1])]
                           - msd[abs(ti[n]  -ti[m])]
                          )

################## Gaussian Process likelihood ###############################

# Write only one triangle of C when assembling
# use dscal for 0.5*

cdef FLOAT_t LOG_2PI = np.log(2*np.pi)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef FLOAT_t _core_logL(FLOAT_t[:, ::1] C, # will be overwritten
                        FLOAT_t[::1]    x,
                        FLOAT_t[::1]    y, # copy of x
                       ) except? 1e300:   # return +inf on error
    # Implementation notes
    # - (slogdet, solve) is faster than eigendecomposition (~3x) in python;
    #   test in C
    # - don't check positive definiteness here, that should be done before—if
    #   necessary; c.f. Fit.constraint_Cpositive
    cdef FLOAT_t logdet, xCx
    cdef int i, N=C.shape[0], info, inc=1
    cdef double one=1.

    # Cholesky factorization
    dpotrf('l', &N, &C[0, 0], &N, &info)

    # logdet from cholesky
    if info == 0:
        logdet = 0.
        for i in range(N):
            logdet += log(fabs(C[i, i]))
    else:
        raise BadCovarianceError(f"Cholesky factorization failed, dpotrf returned {info}")

    logdet *= 2

    # Compute xCx = x @ inv(C) @ x = x @ y with C @ y = x
    # solve with cholesky: L @ L.T @ y = x
    # dcopy(&x[0], &inc, &y[0], &inc) # copy x --> y # is now required as parameter
    dtrsm('l', 'l', 'n', 'n',       # L @ b = x
          &N, &inc, &one,
          &C[0, 0], &N,
          &y[0], &N,
         )
    dtrsm('l', 'l', 't', 'n',       # L.T @ y = b
          &N, &inc, &one,
          &C[0, 0], &N,
          &y[0], &N,
         )
    xCx = <FLOAT_t> ddot(&N, &x[0], &inc, &y[0], &inc)

    return -0.5*( xCx + logdet + N*LOG_2PI )

def logL(trace, ss_order, msd, mean=0):
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
    cdef FLOAT_t[:, ::1] C, c
    cdef FLOAT_t[::1]    X, x

    ti = np.nonzero(~np.isnan(trace))[0].astype(np.uint)
    cdef SIZE_t N = len(ti)

    if ss_order < 1:
        X = trace[ti] - mean
        C = np.empty((N, N), dtype=float)
        msd2C_ss0(msd, ti, C)

        if ss_order == 0.5: # condition on first data point
            c = np.empty((N-1, N-1), dtype=float)
            x = np.empty(N-1, dtype=float)

            for i in range(N-1):
                x[i] = X[i+1] - X[0] * C[0, i+1]/C[0, 0]
                for j in range(i, N-1):
                    c[i, j] = C[i+1, j+1] - C[0, i+1]*C[0, j+1]/C[0, 0]

            C = c
            X = x

    elif ss_order == 1:
        X = np.diff(trace[ti]) - mean*np.diff(ti)
        C = np.empty((N-1, N-1), dtype=float)
        msd2C_ss1(msd, ti, C)

    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")

    return _core_logL(C, X, X.copy())
