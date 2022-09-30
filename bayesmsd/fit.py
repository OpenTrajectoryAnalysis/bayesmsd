"""
Implementation of `Fit` base class

See also
--------
bayesmsd, Fit, Profiler <bayesmsd.profiler.Profiler>
"""
from abc import ABCMeta, abstractmethod
from copy import deepcopy

from tqdm.auto import tqdm

import numpy as np
from scipy import optimize

from noctiluca import make_TaggedSet

from . import gp
from .deco import method_verbosity_patch

class Fit(metaclass=ABCMeta):
    """
    Abstract base class for MSD fits. Backbone of bayesmsd.

    Subclass this to implement fitting of a specific functional form of the
    MSD. See also the existing library of fits in `bayesmsd.lib`.

    Parameters
    ----------
    data : noctiluca.TaggedSet, pandas.DataFrame, list of numpy.ndarray
        the data to run the fit on.

        This input is handled by the `userinput.make_TaggedSet()
        <https://noctiluca.readthedocs.io/en/latest/noctiluca.util.html#noctiluca.util.userinput.make_TaggedSet>`_
        function of the `noctiluca` package and thus accepts a range of
        formats. A dataset of trajectories with ``N`` loci, ``T`` frames, and
        ``d`` spatial dimensions can be specified as list of numpy arrays with
        dimensions ``(N, T, d)``, ``(T, d)``, ``(T,)``, where ``T`` can vary
        between trajectories but ``N`` and ``d`` should be the same. A pandas
        dataframe should have columns ``(particle, frame, x1, y1, z1, x2, y2,
        z2, ...)``, where ``particle`` identifies which trajectory each entry
        belongs to, which ``frame`` is the frame number in which it was
        detected. For precise specs see `noctiluca.util.userinput
        <https://noctiluca.readthedocs.io/en/latest/noctiluca.util.html#module-noctiluca.util.userinput>`_.

    Attributes
    ----------
    data : `TaggedSet` of `Trajectory`
        the data to use. Note that the current selection in the data is saved
        internally
    d : int
        spatial dimension of trajectories in the data
    T : int
        maximum length of the trajectories, in frames
    ss_order : {0, 1}
        steady state order. Often this will be a fixed constant for a
        particular `Fit` subclass.
    bounds : list of (lb, ub)
        bound for each of the parameters in the fit. This is everything the
        class "knows" about your choice of parameters internally. The length of
        this list is also used internally to count number of parameters, so it
        is important that every parameter has bounds. Use ``np.inf`` and
        ``-np.inf`` for unbounded parameters.
    fix_values : list of (i, fix_to)
        allows to fix some parameter values to constant or values of other
        parameters, e.g. to allow for different behavior in different
        dimensions, but fixing it to be equal by default. See Notes section.
    constraints : list of constraint functions
        allows to specify constraints on the parameters that will be
        implemented as smooth penalty on the likelihood. Can also take care of
        feasibility constraints: by default, there is a constraint checking
        that the covariance matrix given by the current MSD is positive
        definite (otherwise we could not even evaluate the likelihood). See
        Notes section.
    max_penalty : float
        constant cutoff for the penalty mechanism. Infeasible sets of
        parameters (where the likelihood function is not even well-defined)
        will be penalized with this value. For any set of parameters penalized
        with at least this value, likelihood evaluation is skipped and the
        value ``max_penalty`` assigned as value of the minimization target.
        Default value is ``1e10``, there should be little reason to change
        this.
    verbosity : {0, 1, 2, 3}
        controls output during fitting. 0: no output; 1: error messages only;
        2: informational; 3: debugging

    Notes
    -----
    The `!fix_values` mechanism allows to keep some parameters fixed, or
    express them as function of others. ``Fit.fix_values`` is a list of tuples
    ``(i, fix_to)``, where ``i`` is the index of the parameter you want to fix,
    c.f. `!bounds`. ``fix_to`` is either just a constant value, or a function
    ``fix_to(params) --> float``, where ``params`` are the current parameter
    values. Note that in this function you should not rely on any parameters
    that are themselves to be fixed. (It would get impossible to resolve all
    the dependencies).

    `!constraints` are functions with signature ``constraint(params) -->
    float``. The output is interpreted as follows:
    - x <= 0 : infeasible; maximum penalization
    - 0 <= x <= 1 : smooth penalization: ``penalty = exp(1/tan(pi*x))``
    - 1 <= x : feasible; no penalization
    Thus, if e.g. some some function ``fun`` of the parameters should be
    constrained to be positive, you would use ``fun(params)/eps`` as the
    constraint, with ``eps`` some small value setting the tolerance region. If
    there are multiple constraints, always the strongest one is used. For
    infeasible parameters, the likelihood function is not evaluated, but the
    "likelihood" is just set to ``-Fit.max_penalty``.

    Note that there is a default constraint checking positivity of the
    covariance matrix. If your functional form of the MSD is guaranteed to
    satisfy this (e.g. for a physical model), you can remove this constraint
    for performance.

    Upon subclassing, it is highly recommended to initialize the base class
    first thing:

    >>> def SomeFit(Fit):
    ...     def __init__(self, data, *other_args):
    ...         super().__init__(data) # <--- don't forget!

    This class uses ``scipy.optimize.minimize`` to find the MAP parameter
    estimate (or MLE if you leave `logprior` flat). When running the fit, you
    can choose between the simplex (Nelder-Mead) algorithm or gradient descent
    (L-BFGS-B). The latter uses the stopping criterion ``f^k -
    f^{k+1}/max{|f^k|,|f^{k+1}|,1} <= ftol``, which is inappropriate for
    log-likelihoods (which should be optimized to fixed accuracy of O(0.1)
    independent of the absolute value, which might be very large). We therefore
    use Nelder-Mead by default, which does not depend on derivatives and thus
    also has an absolute stopping criterion.

    If this function runs close to a maximum, e.g. in `Profiler` or when using
    successive optimization steps, we can fix the problem with gradient-based
    optimization by removing the large absolute value offset from the
    log-likelihood. This functionality is also exposed to the end user, who can
    overwrite the ``initial_offset`` method to give a non-zero offset together
    with the initial values provided via `initial_params`.
    """
    def __init__(self, data):
        self.data = make_TaggedSet(data)
        self.data_selection = self.data.saveSelection()
        self.d = self.data.map_unique(lambda traj: traj.d)
        self.T = max(map(len, self.data))

        # Fit properties
        self.ss_order = 0
        self.bounds = [] # List of (lb, ub), to be passed to optimize.minimize. len(bounds) is used to count parameters!
        self.fix_values = [] # will be appended to the list passed to run()

        # Each constraint should be a callable constr(params) -> x. We will apply:
        #   x <= 0                : infeasible. Maximum penalization
        #        0 <= x <= 1      : smooth crossover: np.exp(1/tan(pi*x))
        #                  1 <= x : feasible. No penalization   
        self.constraints = [self.constraint_Cpositive]
        self.max_penalty = 1e10
        
        self.verbosity = 1
        
    def vprint(self, v, *args, **kwargs):
        """
        Prints only if ``self.verbosity >= v``.
        """
        if self.verbosity >= v:
            print("[bayesmsd.Fit]", (v-1)*'--', *args, **kwargs)
        
    ### To be overwritten / used upon subclassing ###

    @property # should declare @property when overriding
    @abstractmethod
    def parameter_names(self):
        """
        A list of names for the fit parameters, as defined by `!bounds`.
        """
        raise NotImplementedError # pragma: no cover
    
    @abstractmethod
    def params2msdm(self, params):
        """
        Definition of MSD in terms of parameters

        This is the core of the fit definition. It should give a list of tuples
        as required by `gp.ds_logL`

        Parameters
        ----------
        params : np.ndarray
            the current parameters

        Returns
        -------
        list of tuples (msd, m)
            for the definition of `!msd`, use of the `MSDfun` decorator is
            recommended.

        See also
        --------
        gp.ds_logL, MSDfun, imaging
        """
        raise NotImplementedError # pragma: no cover

    def logprior(self, params):
        """
        Prior over the parameters

        Use this function if you want to specify a prior over the parameters
        that will be added to the log-likelihood.  Default is a flat prior,
        i.e. ``return 0``.

        Parameters
        ----------
        params : np.ndarray
            the current parameters

        Returns
        -------
        float
        """
        return 0
    
    @abstractmethod
    def initial_params(self):
        """
        Give initial values for the parameters

        You can use ``self.data`` to perform some ad hoc estimation (e.g. from
        the empirical MSD, using `!noctiluca.analysis.MSD`) or just return
        constants.

        Returns
        -------
        params : np.ndarray
            initial values for the parameters. Should satisfy ``len(params) ==
            len(self.bounds)``.
        """
        raise NotImplementedError # pragma: no cover

    def initial_offset(self):
        """
        Log-likelihood offset associated with initial parameters

        See Notes section of class documentation.

        Returns
        -------
        float

        See also
        --------
        Fit
        """
        return 0
                
    def constraint_Cpositive(self, params):
        """
        Constraint for positive definiteness of covariance matrix

        This can serve as an example of a non-trivial constraint. Note that you
        may not want to use it if positivity is already guaranteed by the
        functional form of your MSD. See also Notes section of class doc.

        Returns
        -------
        float

        Notes
        -----
        This function checks whether the spatial components are identical using
        python's ``is``. So if you are implementing an MSD with identical
        spatial components, you should return the final list as ``self.d*[(msd,
        mean)]``, such that this constraint checks positivity only once.

        See also
        --------
        Fit
        """
        min_ev_okay = 1 - np.cos(np.pi/self.T) # white noise min ev

        scores = []
        done = []
        for msd, _ in self.params2msdm(params):
            if msd not in done:
                min_ev = np.min(np.linalg.eigvalsh(gp.msd2C_fun(msd, np.arange(self.T), self.ss_order)))
                scores.append(min_ev / min_ev_okay)
                done.append(msd)

        return min(scores)
    
    ### General machinery, usually won't need overwriting ###

    def parameter_index(self, name):
        """
        Give the index for a name from `!parameter_names`
        """
        try:
            return np.nonzero(np.array(self.parameter_names) == name)[0][0]
        except IndexError:
            raise ValueError(f"Did not find name '{name}' in self.parameter_names = {self.parameter_names}")

    def MSD(self, fitres, dt=None):
        """
        Return the MSD evaluated at dt. Convenience function

        Parameters
        ----------
        fitres : dict
            the output of a previous fit run. Specifically, this should be a
            dict having an entry ``'params'`` that is a 1d ``np.array``
            containing the parameter values.
        dt : array-like
            the time lags (in frames) at which to evaluate the MSD. If left
            unspecified, we return a callable MSD function
        kwargs : additional keyword arguments
            are forwarded to the MSD functions returned by `params2msdm`.

        Returns
        -------
        callable or np.array
            the MSD function (summed over all dimensions), evaluated at `!dt` if provided.
        """
        def msdfun(dt, params=fitres['params'], **kwargs):
            msdm = self.params2msdm(params)
            return np.sum([msd(dt, **kwargs) for msd, m in msdm], axis=0)

        if dt is None:
            # Write proper signature to docstring
            single_msdfun = self.params2msdm(fitres['params'])[0][0]
            if hasattr(single_msdfun, '_kwargstring'):
                msdfun.__doc__ = f"""full signature:

msdfun(dt,
       params={fitres['params']},
       {single_msdfun._kwargstring},
      )

`params` overrides parameters that are later given as keywords.
"""
            return msdfun
        else:
            return msdfun(dt)
        
    def _penalty(self, params):
        """
        Gives penalty for given parameters. Internal use.

        Parameters
        ----------
        params : np.ndarray

        Returns
        -------
        float
        """
        x = np.inf
        for constraint in self.constraints:
            x = min(constraint(params), x)
            if x <= 0:
                return -1 # unfeasible

        if x >= 1:
            return 0
        else:
            with np.errstate(over='raise', under='ignore'):
                try:
                    return min(np.exp(1/np.tan(np.pi*x)), self.max_penalty)
                except FloatingPointError:
                    return self.max_penalty

    def expand_fix_values(self, fix_values=None):
        """
        Preprocessing for fixed parameters. Mostly internal use.

        This function concatenates the internal ``self.fix_values`` and the
        given ``fix_values``, and makes sure that they all conform to the
        format ``(i, function)``, i.e. it converts fixed values given as
        constants to (trivial) functions.

        Parameters
        ----------
        fix_values : list of tuples (i, fix_to), optional
            values to fix, beyond what's already in ``self.fix_values``

        Returns
        -------
        fix_values : list
            same as input, plus internal ``self.fix_values`` and constants
            resolved

        See also
        --------
        Fit.fix_values, get_value_fixer
        """
        # Converts fix_values to useable format (i, function)
        # Also appends self.fix_values
        if fix_values is None:
            fix_values = []
        fix_values = fix_values + self.fix_values
        
        ifix = [i for i, _ in fix_values]
        _, ind = np.unique(ifix, return_index=True)
        full = []
        for i in ind:
            ip, fun = fix_values[i]
            if not callable(fun):
                fun = lambda x, val=fun : val # `fun` is (initially) of course a misnomer in this case

            full.append((ip, fun))
        return full

    def number_of_fit_parameters(self, fix_values=None):
        """
        Get the number of independent fit parameters

        The number of independent fit parameters is the full number of
        parameters (``len(params)``) minus the number of parameters that are
        fixed via the ``fix_values`` mechanism. This function is provided for
        convenience.

        Parameters
        ----------
        fix_values : list of tuples (i, fix_to), optional
            should be the same as was / will be handed to `run`

        Returns
        -------
        int
            the number of fit parameters
        """
        n_fixed = len(self.expand_fix_values(fix_values)) # also ensures uniqueness of indices
        return len(self.bounds) - n_fixed
                
    def get_value_fixer(self, fix_values=None):
        """
        Create a function that resolves fixed parameters

        Parameters
        ----------
        fix_values : list of tuples (i, fix_to), optional
            values to fix, beyond what's already in ``self.fix_values``

        Returns
        -------
        fixer : callable
            a function with signature ``fixer(params) --> params``, where in
            the output array all parameters fixes are applied.
        """
        fix_values = self.expand_fix_values(fix_values)

        n_params = len(self.bounds)
        is_fixed = np.zeros(n_params, dtype=bool)
        for i, _ in fix_values:
            is_fixed[i] = True
        
        def value_fixer(params, fix_values=fix_values, is_fixed=is_fixed, n_params=n_params):
            if len(params) == n_params:
                fixed_params = params.copy()
            elif len(params) + np.sum(is_fixed) == n_params:
                fixed_params = np.empty(n_params, dtype=float)
                fixed_params[:] = np.nan
                fixed_params[~is_fixed] = params
            else: # pragma: no cover
                raise RuntimeError(f"Did not understand how to fix up parameters: len(params) = {len(params)}, sum(is_fixed) = {np.sum(is_fixed)}, n_params = {n_params}")

            for ip, fixfun in fix_values:
                fixed_params[ip] = fixfun(fixed_params)
            
            return fixed_params

        return value_fixer
    
    def get_min_target(self, offset=0, fix_values=None, do_fixing=True):
        """
        Define the minimization target (negative log-likelihood)

        Parameters
        ----------
        offset : float
            constant to subtract from log-likelihood. See Notes section of
            class doc.
        fix_values : list of tuples (i, fix_to)
            values to fix, beyond what's already in ``self.fix_values``
        do_fixing : bool
            set to ``False`` to prevent the minimization target from resolving
            any of the parameter fixes (by default). Might be useful when
            exploring parameter space.

        Returns
        -------
        min_target : callable
            function with signature ``min_target(params) --> float``.

        Notes
        -----
        The returned ``min_target`` takes additional keyword arguments:

        - ``just_return_full_params`` : bool, ``False`` by default. If
          ``True``, don't calculate the actual target function, just return the
          parameter values after fixing
        - ``do_fixing`` : bool, defaults to `!do_fixing` as given to
          `get_min_target`.
        - fixer, offset : just handed over as arguments for style (scoping)

        See also
        --------
        run
        """
        fixer = self.get_value_fixer(fix_values)
        
        def min_target(params, just_return_full_params=False,
                       do_fixing=do_fixing, fixer=fixer, offset=offset,
                       ):
            if do_fixing:
                params = fixer(params)
            if just_return_full_params:
                return params

            penalty = self._penalty(params)
            if penalty < 0: # infeasible
                return self.max_penalty
            else:
                return -gp.ds_logL(self.data,
                                   self.ss_order,
                                   self.params2msdm(params),
                                  ) \
                       - self.logprior(params) \
                       + penalty \
                       - offset

        return min_target

    @method_verbosity_patch
    def run(self,
            init_from = None,
            optimization_steps=('simplex',),
            maxfev=None,
            fix_values = None,
            full_output=False,
            show_progress=False,
           ):
        """
        Run the fit

        Parameters
        ----------
        init_from : dict
            initial point for the fit, as a dict with fields ``'params'`` and
            ``'logL'``, like the ones this function returns. If you just want
            to initialize from a certain parameter point, you can set
            ``init_from['logL'] = 0``.
        optimization_steps : tuple
            successive optimization steps to perform. Entries should be
            ``'simplex'`` for Nelder-Mead, ``'gradient'`` for gradient descent,
            or a dict whose entries will be passed to
            ``scipy.optimize.minimize`` as keyword arguments.
        maxfev : int or None
            limit on function evaluations for ``'simplex'`` or ``'gradient'``
            optimization steps
        fix_values : list of tuples (i, fix_to)
            can be used to keep some parameter values fixed or express them as
            function of the other parameters. See class doc for more details.
        full_output : bool
            Set to ``True`` to return the output dict (c.f. Returns) and the
            full output from ``scipy.optimize.minimize`` for each optimization
            step. Otherwise (``full_output == False``, the default) only the
            output dict from the ultimate run is returned.
        show_progress : bool
            display a `!tqdm` progress bar while fitting
        verbosity : {None, 0, 1, 2, 3}
            if not ``None``, overwrites the internal ``self.verbosity`` for
            this run. Use to silence or get more details of what's happening

        Returns
        -------
        dict
            with fields ``'params'``, a complete set of parameters; ``'logL'``,
            the associated value of the likelihood (or posterior, if the prior
            is non-trivial).
        """
        self.data.restoreSelection(self.data_selection)

        for step in optimization_steps:
            assert type(step) is dict or step in {'simplex', 'gradient'}
        
        # Initial values
        if init_from is None:
            p0 = self.initial_params()
            total_offset = self.initial_offset()
        else:
            p0 = deepcopy(init_from['params'])
            total_offset = -init_from['logL']
        
        # Adjust for fixed values
        ifix = [i for i, _ in self.expand_fix_values(fix_values)]
        bounds = deepcopy(self.bounds)
        for i in sorted(ifix, reverse=True):
            del bounds[i]
            p0[i] = np.nan
        p0 = p0[~np.isnan(p0)]
        
        # Set up progress bar
        bar = tqdm(disable = not show_progress)
        def callback(x):
            bar.update()
        
        # Go!
        all_res = []
        with np.errstate(all='raise'):
            for istep, step in enumerate(optimization_steps):
                if step == 'simplex':
                    # The stopping criterion for Nelder-Mead is max |f_point -
                    # f_simplex| < fatol AND max |x_point - x_simplex| < xatol.
                    # We don't care about the precision in the parameters (that
                    # should be determined by running the Profiler), so we
                    # switch off the xatol mechanism and use only fatol.
                    # Note that fatol is "just" a threshold on the decrease
                    # relative to the last evaluation, not a strict bound. So
                    # the Profiler might actually still find a better estimate,
                    # even though it uses the same tolerance (1e-3).
                    options = {'fatol' : 1e-3, 'xatol' : np.inf}

                    if maxfev is not None:
                        options['maxfev'] = maxfev
                    kwargs = dict(method = 'Nelder-Mead',
                                  options = options,
                                  bounds = bounds,
                                  callback = callback,
                                 )
                elif step == 'gradient':
                    options = {}
                    if maxfev is not None:
                        options['maxfun'] = maxfev
                    kwargs = dict(method = 'L-BFGS-B',
                                  options = options,
                                  bounds = bounds,
                                  callback = callback,
                                 )
                else:
                    kwargs = dict(callback = callback,
                                  bounds = bounds,
                                 )
                    kwargs.update(step)
                    
                min_target = self.get_min_target(offset=total_offset, fix_values=fix_values)
                try:
                    fitres = optimize.minimize(min_target, p0, **kwargs)
                except gp.BadCovarianceError as err: # pragma: no cover
                    self.vprint(2, "BadCovarianceError:", err)
                    fitres = lambda: None
                    fitres.success = False
                
                if not fitres.success:
                    self.vprint(1, f"Fit (step {istep}: {step}) failed. Here's the result:")
                    self.vprint(1, '\n', fitres)
                    raise RuntimeError("Fit failed at step {:d}: {:s}".format(istep, step))
                else:
                    all_res.append(({'params' : min_target(fitres.x, just_return_full_params=True),
                                     'logL' : -(fitres.fun+total_offset),
                                    }, fitres))
                    p0 = fitres.x
                    total_offset += fitres.fun
                    
        bar.close()
        
        if full_output:
            return all_res
        else:
            return all_res[-1][0]
