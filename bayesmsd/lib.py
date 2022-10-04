"""
A library of `Fit <bayesmsd.fit.Fit>` implementations

This module provides some commonly used MSD fits. Most notably:

+ Fitting a powerlaw through `NPXFit`: NPX stands for Noise + Powerlaw + X,
  i.e. this fit class has the capability of fitting a powerlaw MSD to your data
  and supplement it with standard imaging artifacts (localization error and
  motion blur) as well as a freeform extension at long times (we use cubic
  splines). The latter is disabled (``n = 0``) by default, thus providing a
  standard powerlaw fit.
+ Fitting cubic splines to your MSD curve via `SplineFit`. Doing this for
  varying number ``n`` of spline points allows to discern which features of an
  empirical MSD curve are significant and which are just noise. To compare fits
  with different number of spline points, you can use AIC; the number of free
  parameters for each fit can be obtained from `Fit.number_of_fit_parameters
  <bayesmsd.fit.Fit.number_of_fit_parameters>`.
"""
from copy import deepcopy

import numpy as np
from scipy import optimize, interpolate

import rouse
from noctiluca.analysis.p2 import MSD

from . import deco
from .fit import Fit
from .parameters import Parameter, Linearize

class SplineFit(Fit):
    """
    Fit a spline MSD

    The MSD in this case is parametrized by the positions of a few spline
    points, between which we interpolate with cubic splines. The boundary
    conditions for the splines are set such that the fitted MSD extrapolates
    beyond the data as a powerlaw, except for large times in the ``ss_order ==
    0`` case, in which the MSD converges to a constant at infinite time. To
    achieve this within the spline fits, the time coordinate is compactified
    from ``t in [1, T]`` to ``x in [0, 1]``:

    - if ``ss_order == 0``, we need ``t = inf`` to be accessible. We therefore
      choose the compactification ``x = 4/π*arctan(log(t/T))``, such that ``t =
      inf`` corresponds to ``x = 2``.
    - if ``ss_order == 1``, we simply work in log-space, and normalize: ``x =
      log(t)/log(T)``.

    These compactifications are of course applied only internally.

    For the y-coordinate of the spline we apply a simple log-transform, ``y =
    log(MSD)``. We then use the boundary condition that the second derivative
    of the spline vanishes, meaning the fitted MSDs can be extrapolated quite
    naturally by powerlaws.

    Parameters
    ----------
    data : noctiluca.TaggedSet, pandas.DataFrame, list of numpy.ndarray
        the data to fit. See `Fit`.
    ss_order : {0, 1}
        the steady state order to assume
    n : int >= 2
        the number of spline points
    previous_spline_fit_and_result : tuple (SplineFit, dict)
        this can be used to initialize the current fit from the resulting
        spline of a previous one. Very useful when running the above-mentioned
        model selection task over spline knots. First entry should be the
        `Fit` object used, second one should be the resulting dict with keys
        ``'params'`` and ``'logL'``.

    Notes
    -----
    Clearly the number of spline points controls the goodness of fit and the
    degree of overfitting. It is thus recommended to use some information
    criterion (e.g. AIC) to determine a reasonable level of detail given the
    data. This approach turns out to be pretty useful in understanding which
    features of an "empirical MSD" (like calculated by
    `!noctiluca.analysis.MSD`) are reliable, and which ones are just noise.

    The parameters for this fit are the coordinates of the spline knots, in the
    transformed coordinate system (``(dt, MSD) --> (x, y) == (compactify(dt),
    log(MSD))``). Since the total extent of the data along the time axis is
    fixed, we also fix the first and last x-coordinates, such that ultimately
    the free parameters are ``(n-2)*[x] + n*[y]``.

    See also
    --------
    Fit, msdfit
    """
    def __init__(self, data, ss_order, n,
                 previous_spline_fit_and_result=None,
                 motion_blur_f=0,
                ):
        super().__init__(data)
        self.motion_blur_f = motion_blur_f

        if n < 2: # pragma: no cover
            raise ValueError(f"SplineFit with n = {n} < 2 does not make sense")
        self.n = n
        
        self.ss_order = ss_order
        self.constraints = [self.constraint_dx,
                            self.constraint_logmsd,
                            self.constraint_Cpositive,
                           ]

        self.x_first = 0
        if self.ss_order == 0:
            self.bc_type = ('natural', (1, 0.0))
            self.x_last = 2
        elif self.ss_order == 1:
            self.bc_type = 'natural'
            self.x_last = 1
        else: # pragma: no cover
            raise ValueError(f"Did not understand ss_order = {ss_order}")

        # x lives in the compactified interval [0, 1] (or [0, 2]), while y =
        # log(MSD) can be any real number.
        self.parameters = {}
        for i in range(self.n):
            self.parameters[f"x{i}"] = Parameter((self.x_first, self.x_last),
                                                 linearization=Linearize.Bounded(),
                                                 )
            self.parameters[f"y{i}"] = Parameter((-np.inf, np.inf),
                                                 linearization=Linearize.Exponential(),
                                                 )

        # x0 = x_first and x(n-1) = x_last are fixed
        del self.parameters["x0"]
        del self.parameters[f"x{self.n-1}"]

        self.prev_fit = previous_spline_fit_and_result # for (alternative) initialization

    def compactify(self, dt):
        """
        Compactification used for the current fit

        Parameters
        ----------
        dt : np.array, dtype=float
            the time lags to calculate compactification for. Should be ``> 0``
            but might include ``np.inf``.

        See also
        --------
        SplineFit
        """
        x = np.log(dt) / np.log(self.T)
        if self.ss_order == 0:
            x = (4/np.pi)*np.arctan(x)
        return x
            
    def decompactify_log(self, x):
        """
        Decompactify spline points (convenience function)

        Parameters
        ----------
        x : np.array, dtype=float
            the compactified x-coordinates

        Returns
        -------
        log_dt : np.array
            the corresponding ``log(dt [frames])`` values
        """
        if self.ss_order == 0:
            x = np.tan(np.pi/4*x)
        x[x == np.tan(np.pi/2)] = np.inf # patch np.pi precision (np.tan(np.arctan(np.inf)) = 1.633e16 != np.inf)
        return x * np.log(self.T)

    def _params2csp(self, params):
        """
        Calculate the cspline from the current parameters

        This spline lives in the compactified x-y-space. It is thus of limited
        use outside of this class and therefore designated for internal use
        mostly.

        Parameters
        ----------
        params : np.array, dtype=float
            the current parameter set

        See also
        --------
        params2msdm
        """
        x = np.array([self.x_first]
                   + [params[f"x{i}"] for i in range(1, self.n-1)]
                   + [self.x_last]
                    )
        y = [params[f"y{i}"] for i in range(self.n)]
        return interpolate.CubicSpline(x, y, bc_type=self.bc_type)

    def params2msdm(self, params):
        """
        Calculate the current spline MSD

        See also
        --------
        Fit.params2msdm
        """
        csp = self._params2csp(params)

        # Calculate powerlaw scaling extrapolating to short times
        alpha0 = csp(0, nu=1) / np.log(self.T)
        if self.ss_order == 0:
            alpha0 *= 4/np.pi

        @deco.MSDfun
        @deco.imaging(f=self.motion_blur_f, alpha0=alpha0)
        def msd(dt, csp=csp):
            # dt == 0 is filtered out by MSDfun
            return np.exp(csp(self.compactify(dt))) / self.d

        return self.d*[(msd, 0)]
            
    def initial_params(self):
        """
        Give suitable initial parameters for the spline

        To find proper initial parameters, we perform a simple powerlaw fit to
        the empirical MSD. In the ``ss_order == 0`` case this is just used as
        boundary condition for a two-point spline between the first time lag
        and infinity (where we use the empirical steady state variance as
        initial value). If ``ss_order == 1`` this fitted powerlaw is the
        initial MSD.
        
        Returns
        -------
        params : np.ndarray, dtype=float
            the inital spline knots, in the internal x-y-coordinates (i.e.
            compactified)

        See also
        --------
        Fit.initial_params
        """
        x_init = np.linspace(self.x_first, self.x_last, self.n)

        # If we have a previous fit (e.g. when doing model selection), use that
        # for initialization
        if self.prev_fit is not None:
            fit, res = self.prev_fit
            y_init = fit._params2csp(res['params'])(x_init)
        else:
            # Fit linear (i.e. powerlaw), which is useful in both cases.
            # For ss_order == 0 we will use it as boundary condition,
            # for ss_order == 1 this will be the initial MSD
            e_msd = MSD(self.data)
            dt_valid = np.nonzero(~np.isnan(e_msd))[0][1:] # skip msd[0] = 0
            (A, B), _ = optimize.curve_fit(lambda x, A, B : A*x + B,
                                           self.compactify(dt_valid),
                                           np.log(e_msd[dt_valid]),
                                           p0=(1, np.log(e_msd[dt_valid[0]])),
                                           bounds=([0, -np.inf], np.inf),
                                          )
                
            if self.ss_order == 0:
                # interpolate along 2-point spline
                ss_var = np.nanmean(np.concatenate([traj.abs()[:][:, 0]**2 for traj in self.data]))
                csp = interpolate.CubicSpline(np.array([0, 2]),
                                              np.log([e_msd[dt_valid[0]], 2*ss_var]),
                                              bc_type = ((1, A), (1, 0.)),
                                             )
                y_init = csp(x_init)
            elif self.ss_order == 1:
                y_init = A*x_init + B
            else: # pragma: no cover
                raise ValueError

        out = {f"x{i}" : x for i, x in enumerate(x_init[1:-1], start=1)}
        out.update({f"y{i}" : y for i, y in enumerate(y_init)})
        return out

    def initial_offset(self):
        """
        Used when starting from a previous `SplineFit`

        See also
        --------
        Fit.initial_offset
        """
        if self.prev_fit is None:
            return 0
        else:
            return -self.prev_fit[1]['logL']
        
    # ensure proper ordering of the x positions
    # if x's cross, the spline might diverge
    def constraint_dx(self, params):
        """
        Make sure the spline points are properly ordered in x

        We impose this constraint mainly to avoid crossing of spline points,
        which usually leads to the spline diverging. On top of that,
        conceptually this makes the solution well-defined.

        Parameters
        ----------
        params : np.ndarray, dtype=float
            the current fit parameters

        Returns
        -------
        float
            the constraint score

        See also
        --------
        Fit
        """
        min_step = 1e-7 # x is compactified to (0, 1)
        x = np.array([self.x_first]
                   + [params[f"x{i}"] for i in range(1, self.n-1)]
                   + [self.x_last]
                    )
        return np.min(np.diff(x))/min_step
    
    # this should (!) be taken care of by the Cpositive constraint
#     def constraint_dy(self, params):
#         # Ensure monotonicity in the MSD. This makes sense intuitively, but is it technically a condition?
#         min_step = 1e-7
#         y = params[(self.n-2):]
#         return np.min(np.diff(y))/min_step
    
    # make sure that the MSD stays numerically finite
    def constraint_logmsd(self, params):
        """
        Make sure the Spline does not diverge

        Parameters
        ----------
        params : np.ndarray, dtype=float
            the current fit parameters

        Returns
        -------
        float
            the constraint score

        See also
        --------
        Fit
        """
        start_penalizing = 200
        full_penalty = 500
        
        csp = self._params2csp(params)
        x_full = self.compactify(np.arange(1, self.T))
        return (full_penalty - np.max(np.abs(csp(x_full))))/start_penalizing

class NPXFit(Fit): # NPX = Noise + Powerlaw + X (i.e. spline)
    def __init__(self, data, ss_order, n=0,
                 previous_NPXFit_and_result=None,
                 motion_blur_f=0,
                ):
        super().__init__(data)
        self.motion_blur_f = motion_blur_f

        if n == 0 and ss_order == 0:
            raise ValueError("Incompatible assumptions: pure powerlaw (n=0) and trajectory steady state (ss_order=0)")
        self.n = n
        self.ss_order = ss_order

        # Set up parameters
        # Populate with templates -> expand dimensions -> remove templates
        # The powerlaw stops being positive definite at α = 2, so stay away from that
        self.parameters = {
            'log(σ²)' : Parameter((-np.inf, np.inf),
                                  linearization=Linearize.Exponential()),
            'log(Γ)'  : Parameter((-np.inf, np.inf),
                                  linearization=Linearize.Exponential()),
            'α'       : Parameter((0, 2-1e-10), # stay away from upper bound
                                  linearization=Linearize.Bounded()),
        }

        x_bounds = (0, 2) if self.ss_order == 0 else (0, 1)
        for i in range(self.n+1):
            self.parameters[f"x{i}"] = Parameter(x_bounds,
                                                 linearization=Linearize.Bounded())
            self.parameters[f"y{i}"] = Parameter((-np.inf, np.inf),
                                                 linearization=Linearize.Exponential())

        # For the spline, y0 and xn are fixed
        del self.parameters["y0"]
        del self.parameters[f"x{self.n}"]

        # Expand dimensions and remove templates
        # Fix higher dimensions to dim 0, except for localization error
        param_names = list(self.parameters.keys()) # keys() itself is mutable
        for name in param_names:
            for dim in range(self.d):
                dim_name = f"{name} (dim {dim})"
                self.parameters[dim_name] = deepcopy(self.parameters[name])

                if name != 'log(σ²)' and dim > 0:
                    self.parameters[dim_name].fix_to = f"{name} (dim 0)"

            del self.parameters[name]

        # Set up constraints
        self.constraints = [self.constraint_dx,
                            self.constraint_logmsd,
                            self.constraint_Cpositive,
                           ]
        if self.n == 0: # pure powerlaw; don't need constraints
            self.constraints = []

        # Compactification & splines
        self.logT = np.log(self.T)
        if self.ss_order == 0:
            # Fit in 4/π*arctan(log) space and add point at infinity, i.e. x = 4/π*arctan(log(∞)) = 2
            self.upper_bc_type = (1, 0.0)
            self.x_last = 2
        elif self.ss_order == 1:
            # Simply fit in log space, with natural boundary conditions
            self.upper_bc_type = 'natural'
            self.x_last = 1
        else: # pragma: no cover
            raise ValueError(f"Did not understand ss_order = {ss_order}")

        self.prev_fit = previous_NPXFit_and_result # for (alternative) initialization
        if self.prev_fit is not None and not self.prev_fit[0].d == self.d:
            raise ValueError(f"Previous NPXFit has different number of dimensions ({self.prev_fit[0].d}) from the current data set ({self.d}).")

    def compactify(self, dt):
        x = np.log(dt) / self.logT
        if self.ss_order == 0:
            x = (4/np.pi)*np.arctan(x)
        return x

    def decompactify_log(self, x):
        if self.ss_order == 0:
            x = np.tan(np.pi/4*x)
        return x * self.logT

    def _first_spline_point(self, x0, logG, alpha):
        logt0 = self.decompactify_log(x0)
        y0 = alpha*logt0 + logG

        # also need derivative for C-spline boundary condition
        if self.ss_order == 0:
            dcdx0 = alpha / ( 4/np.pi*self.logT/(self.logT**2 + logt0**2) )
        elif self.ss_order == 1:
            dcdx0 = alpha * self.logT
        else: # pragma: no cover
            raise ValueError

        return x0, logt0, y0, dcdx0

    def _params2csp(self, params):
        csps = self.d*[None]
        if self.n > 0:
            for dim in range(self.d):
                x0, logt0, y0, dcdx0 = self._first_spline_point(
                                            params[f"x0 (dim {dim})"],
                                            params[f"log(Γ) (dim {dim})"],
                                            params[f"α (dim {dim})"],
                                       )

                x = np.array(       [params[  f"x{i} (dim {dim})"] for i in range(self.n)] + [self.x_last]) 
                y = np.array([y0] + [params[f"y{i+1} (dim {dim})"] for i in range(self.n)])

                csps[dim] = interpolate.CubicSpline(x, y, bc_type=((1, dcdx0), self.upper_bc_type))

        return csps

    def params2msdm(self, params):
        csps = self._params2csp(params)
        msdm = []
        for dim, csp in enumerate(csps):
            with np.errstate(under='ignore'): # if noise == 0
                noise2 = np.exp(params[f"log(σ²) (dim {dim})"])
                G      = np.exp(params[ f"log(Γ) (dim {dim})"])
            alpha = params[f"α (dim {dim})"]

            # Define "naked" MSD function
            if self.n == 0:
                def msd(dt, G=G, alpha=alpha):
                    return G*(dt**alpha)
            else:
                t0 = np.exp(self.decompactify_log(params[f"x0 (dim {dim})"]))

                def msd(dt, G=G, alpha=alpha, csp=csp, t0=t0):
                    out = G*(dt**alpha)
                    ind = dt > t0
                    if np.any(ind):
                        x = self.compactify(dt[ind])
                        out[ind] = np.exp(csp(x))
                    return out

            # Apply MSD function decorators
            msd = deco.imaging(noise2=noise2, f=self.motion_blur_f, alpha0=alpha)(msd)
            msd = deco.MSDfun(msd)

            msdm.append((msd, 0))
        return msdm
            
    def initial_params(self):
        params = {}

        if self.prev_fit:
            fit, res = self.prev_fit
            csps = fit._params2csp(res['params'])

            for dim in range(self.d):
                for name in ['log(σ²)', 'log(Γ)', 'α']:
                    params[f"{name} (dim {dim})"] = res['params'][f"{name} (dim {dim})"]

                if self.n > 0:
                    # Note order of conditions: x0 exists only if fit.n > 0
                    if fit.n == 0 or res['params'][f"x0 (dim {dim})"] >= self.x_last:
                        logG  = res['params'][f"log(Γ) (dim {dim})"]
                        alpha = res['params'][     f"α (dim {dim})"]

                        x_init = np.linspace(0.5, self.x_last, self.n+1)
                        y_init = alpha*self.decompactify_log(x_init) + logG
                    else:
                        x0 = res['params'][f"x0 (dim {dim})"]
                        x_init = np.linspace(x0, self.x_last, self.n+1)
                        y_init = csps[dim](x_init)

                    params.update({f"x{i} (dim {dim})" : x for i, x in enumerate(x_init[:-1])})
                    params.update({f"y{i} (dim {dim})" : y for i, y in enumerate(y_init[1:], start=1)})

        else:
            # Fit linear (i.e. powerlaw), which is useful in both (ss_order) cases.
            # For ss_order == 0 we will use it as boundary condition,
            # for ss_order == 1 this will be the initial MSD
            e_msd = MSD(self.data)/self.d
            dt_valid = np.nonzero(~np.isnan(e_msd))[0][1:]
            (alpha, logG), _ = optimize.curve_fit(lambda x, alpha, logG : alpha*x + logG,
                                                  np.log(dt_valid),
                                                  np.log(e_msd[dt_valid]),
                                                  p0=(1, 0),
                                                  bounds=([0, -np.inf], [2, np.inf]),
                                              )

            for dim in range(self.d):
                params[f"log(σ²) (dim {dim})"] = np.log(e_msd[dt_valid[0]]/2)
                params[ f"log(Γ) (dim {dim})"] = logG
                params[      f"α (dim {dim})"] = alpha

            if self.n > 0:
                x0, logt0, y0, dcdx0 = self._first_spline_point(0.5, logG, alpha)
                x_init = np.linspace(x0, self.x_last, self.n+1)

                if self.ss_order == 0:
                    # interpolate along 2-point spline
                    ss_var = np.nanmean(np.concatenate([np.sum(traj[:]**2, axis=1) for traj in self.data]))/self.d
                    csp = interpolate.CubicSpline(np.array([x0, 2]),
                                                  np.array([y0, np.log(2*ss_var)]),
                                                  bc_type = ((1, dcdx0), (1, 0.)),
                                                 )
                    y_init = csp(x_init)
                elif self.ss_order == 1:
                    y_init = alpha*self.decompactify_log(x_init) + logG
                else: # pragma: no cover
                    raise ValueError

                for dim in range(self.d):
                    params.update({f"x{i} (dim {dim})" : x for i, x in enumerate(x_init[:-1])})
                    params.update({f"y{i} (dim {dim})" : y for i, y in enumerate(y_init[1:], start=1)})

        return params

    def initial_offset(self):
        if self.prev_fit is None:
            return 0
        else:
            # Technically the likelihoods of two NPXFits are not comparable
            # when ss_order is different (which is presumably rare, but might
            # happen. However, we can assume that they are roughly the same
            # order of magnitude, such that setting this as initial offset is
            # probably a better guess than 0.
            return -self.prev_fit[1]['logL']
        
    def constraint_dx(self, params):
        # constraints are not applied if n == 0, so we can safely assume n > 0
        min_step = 1e-7 # x is compactified to (0, 1)
        x = np.array([[params[f"x{i} (dim {dim})"] for i in range(self.n)] + [self.x_last]
                      for dim in range(self.d)])
        return np.min(np.diff(x, axis=-1))/min_step
    
    def constraint_logmsd(self, params):
        # constraints are not applied if n == 0, so we can safely assume n > 0
        start_penalizing = 200
        full_penalty = 500

        csps = self._params2csp(params)
        x_full = self.compactify(np.arange(1, self.T))
        xs = [x_full[x_full >= params[f"x0 (dim {dim})"]] for dim in range(self.d)]

        if all(len(x) == 0 for x in xs): # pragma: no cover
            return np.inf

        logmsd = np.concatenate([csp(x) for csp, x in zip(csps, xs)])
        return (full_penalty - np.max(np.abs(logmsd)))/(full_penalty - start_penalizing)

class TwoLocusRouseFit(Fit):
    """
    Fit a Rouse model for two loci on a polymer at fixed separation

    This class implements a fit for two loci at fixed separation, but on the
    same polymer. A simple model for these dynamics is given by the infinite
    continuous Rouse model, which gives an analytical expression for the MSD
    (implemented in `!rouse.twoLocusMSD`). Here we provide an implementation to
    fit this MSD to data.

    The parameters for this MSD are

    - the (squared) localization error ``noise2``
    - the Rouse scaling prefactor Γ of a single locus on the polymer
    - the steady state variance J of the distance between the loci. This
      encodes the chain length between them

    The parameter vector for this fit is thus ``d*[log(noise2), log(Γ),
    log(J)]``.

    Since all of these are positive quantities with units, it seems natural to
    perform the fit in log-space. Thus we (effectively) place a 1/x prior on
    all of them.

    This class allows for separate fits to all the spatial dimensions of the
    data. In a standard setting this makes sense only for the localization
    error, so by default we fix Γ and J to be the same for all dimensions.
    """
    def __init__(self, data, k=1,
                 motion_blur_f=0,
                ):
        super().__init__(data)
        self.motion_blur_f = motion_blur_f
        
        self.ss_order = 0
        
        self.parameters = {}
        for name in ['log(σ²)', 'log(Γ)', 'log(J)']:
            for dim in range(self.d):
                dim_name = f"{name} (dim {dim})"
                self.parameters[f"{name} (dim {dim})"] = Parameter((-np.inf, np.inf),
                                                                   linearization=Linearize.Exponential())

                if name != 'log(σ²)' and dim > 0:
                    self.parameters[dim_name].fix_to = f"{name} (dim 0)"

        self.constraints = [] # Don't need to check Cpositive, will always be true for Rouse MSDs
        
    def params2msdm(self, params):
        """
        Give an MSD function (and mean = 0) for given parameters

        Uses ``rouse.twoLocusMSD``
        """
        msdm = []
        for dim in range(self.d):
            with np.errstate(under='ignore'): # if noise == 0
                noise2 = np.exp(params[f"log(σ²) (dim {dim})"])
                G      = np.exp(params[ f"log(Γ) (dim {dim})"])
                J      = np.exp(params[ f"log(J) (dim {dim})"])

            @deco.MSDfun
            @deco.imaging(noise2=noise2, f=self.motion_blur_f, alpha0=0.5)
            def msd(dt, G=G, J=J):
                return rouse.twoLocusMSD(dt, G, J)

            msdm.append((msd, 0))
        return msdm
        
    def initial_params(self):
        """
        Initial parameters from empirical MSD
        """
        e_msd = MSD(self.data) / self.d
        dt_valid = np.nonzero(~np.isnan(e_msd))[0][1:]
        dt_valid_early = dt_valid[:min(5, len(dt_valid))]

        J = np.nanmean(np.concatenate([traj[:]**2 for traj in self.data], axis=0))
        G = np.nanmean(e_msd[dt_valid_early]/np.sqrt(dt_valid_early))
        noise2 = e_msd[dt_valid[0]]/2

        params = {}
        for dim in range(self.d):
            params[f"log(σ²) (dim {dim})"] = np.log(noise2)
            params[ f"log(Γ) (dim {dim})"] = np.log(G)
            params[ f"log(J) (dim {dim})"] = np.log(J)

        return params
