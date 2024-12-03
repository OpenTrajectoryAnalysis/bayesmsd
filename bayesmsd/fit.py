"""
Implementation of `Fit` base class

See also
--------
bayesmsd, Fit, Profiler <bayesmsd.profiler.Profiler>
"""
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy

from tqdm.auto import tqdm

import numpy as np
from scipy import optimize, special, stats

from noctiluca import make_TaggedSet, parallel

from .gp import GP, msd2C_fun
from .deco import method_verbosity_patch
from .profiler import Profiler
from .parameters import Parameter, Linearize

_MARGINALIZE = '<marginalize>'

class Fit(metaclass=ABCMeta):
    """
    Abstract base class for MSD fits; the backbone of bayesmsd.

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
    parameters : dict
        the parameters for this fit. Each entry should carry a sensible name
        and be an instance of `bayesmsd.Parameter
        <bayesmsd.parameters.Parameter>`. By default, this is populated with a
        parameter ``m1`` for each dimension, which is fixed to 0. This can be
        used for the first moment (mean/drift) in the fits, or
        removed/overridden.
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
    likelihood_chunksize : int
        controls chunking of parallelization (if running in
        `!noctiluca.Parallelize` context): ``< 0`` prevents any
        parallelization; ``0`` submits the whole likelihood calculation into
        one process; ``> 0`` chunks the likelihood calculation. A chunk size of
        ``1`` means that for each dimension of each trajectory we submit a
        separate task; this usually leads to (way) too much overhead, so higher
        chunk sizes are recommended.
    maxfev : float
        maximum number of function evaluations per fitting run. Defaults to
        "practically infinite", 1e10. How this works is that any `!MinTarget`
        of this fit will admit only `!maxfev` calls; this means that whenever a
        new `!MinTarget` is initialized (i.e. for a new fit run; using
        `!logL()`; etc.), this counter is "reset".

    Notes
    -----
    `!constraints` are functions with signature ``constraint(params) -->
    float``. The output is interpreted as follows:

     - x <= 0 : infeasible; maximum penalization
     - 0 <= x <= 1 : smooth penalization: ``penalty = exp(cot(pi*x))``
     - 1 <= x : feasible; no penalization

    Thus, if e.g. some some function ``fun`` of the parameters should be
    constrained to be positive, you would use ``fun(params)/eps`` as the
    constraint, with ``eps`` some small value setting the tolerance region. If
    there are multiple constraints, always the strongest one is used. For
    infeasible parameters, the likelihood function is not evaluated; instead
    the "likelihood" is just set to ``-Fit.max_penalty``.

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
    independent of the absolute value, which in turn might be very large). We
    therefore use Nelder-Mead by default, which does not depend on derivatives
    and thus also has an absolute stopping criterion.

    If this function runs close to a maximum, e.g. in `Profiler
    <bayesmsd.profiler.Profiler>` or when using successive optimization steps,
    we can fix the problem with gradient-based optimization by removing the
    large absolute value offset from the log-likelihood. This functionality is
    also exposed to the end user, who can overwrite the ``initial_offset``
    method to give a non-zero offset together with the initial values provided
    via `initial_params`.

    The rationale for pre-populating `!parameters` with the first moment
    parameters is that these are rarely interesting when writing a new fit, so
    wouldn't be implemented by a lazy developer (yours truly) by default. Just
    carrying them through, however, is completely trivial and exposes the trend
    fitting functionality to the end user by simply "unfixing" the
    corresponding parameters (i.e. setting ``.fix_to = None``).
    """
    def __init__(self, data):
        self.data = make_TaggedSet(data)
        self.data_selection = self.data.saveSelection()
        self.d = self.data.map_unique(lambda traj: traj.d)
        self.T = max(map(len, self.data))

        # Fit properties
        self.ss_order = None
        self.parameters = {f'm1 (dim {dim})' : Parameter((-np.inf, np.inf),
                                                         linearization=Linearize.Exponential(),
                                                         fix_to=0,
                                                         )
                           for dim in range(self.d)
                           } # to be extended / modified when subclassing
        # `params` arguments to `params2msdm` (and others) will be dict with same keys

        # Each constraint should be a callable constr(params) -> x. We will apply:
        #   x <= 0                : infeasible. Maximum penalization
        #        0 <= x <= 1      : smooth crossover: np.exp(1/tan(pi*x))
        #                  1 <= x : feasible. No penalization   
        self.constraints = [self.constraint_Cpositive]
        self.max_penalty = 1e10
        
        self.verbosity = 1
        self.likelihood_chunksize = -1
        self.maxfev = 1e10

        # List of those parameter names that have improper priors
        # Remember to define ``self.logprior`` for those that have proper priors!
        self.improper_priors = []

    def vprint(self, v, *args, **kwargs):
        """
        Prints only if ``self.verbosity >= v``.
        """
        if self.verbosity >= v: # pragma: no cover
            print("[bayesmsd.Fit]", (v-1)*'--', *args, **kwargs)
        
    ### To be overwritten / used upon subclassing ###
    
    @abstractmethod
    def params2msdm(self, params):
        """
        Definition of MSD in terms of parameters

        This is the core of the fit definition. It should give a list of tuples
        as required by `GP.ds_logL <bayesmsd.gp.GP.ds_logL>`

        Parameters
        ----------
        params : dict
            the current parameter values

        Returns
        -------
        list of tuples (msd, m)
            for the definition of `!msd`, use of the `MSDfun
            <bayesmsd.deco.MSDfun>` decorator (and potentially also `imaging
            <bayesmsd.deco.imaging>`) is recommended.

        See also
        --------
        GP.ds_logL <bayesmsd.gp.GP.ds_logL>, MSDfun <bayesmsd.deco.MSDfun>,
        imaging <bayesmsd.deco.imaging>
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
        params : dict
            the current parameter values; note that only the free parameters
            are given

        Returns
        -------
        float
        """
        # Since we included the possibility to calculate evidences, model
        # implementations need to actually take care of the prior
        raise NotImplementedError # pragma: no cover
    
    @abstractmethod
    def initial_params(self):
        """
        Give initial values for the parameters

        You can use ``self.data`` to perform some ad hoc estimation (e.g. from
        the empirical MSD, using `!noctiluca.analysis.MSD`) or just return
        constants.

        Returns
        -------
        params : dict
            initial values for the parameters. Should have the same keys as
            ``self.parameters``.
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
                min_ev = np.min(np.linalg.eigvalsh(msd2C_fun(msd, np.arange(self.T), self.ss_order))) / msd(1)
                scores.append(min_ev / min_ev_okay)
                done.append(msd)

        return min(scores)
    
    ### General machinery, usually won't need overwriting ###

    def logL(self, params):
        """
        Evaluate log-likelihood (convenience function)

        This function evaluates the log-likelihood for the given parameter
        values. Note that this is purely a convenience function; it is
        equivalent to

        >>> mt = fit.MinTarget(fit)
        ... logL = -mt(mt.params_dict2array(params))

        Parameters
        ----------
        params : dict
            the parameter values. Has to contain values for independent
            parameters, can contain other values (which would be ignored)

        Returns
        -------
        float
        """
        mt = self.MinTarget(self)
        return -mt(mt.params_dict2array(params))

    def fill_dependent_params(self, params, fix_values=None):
        """
        Fill in dependent parameters (convenience function)

        Parameters
        ----------
        params_independent : dict
            a dictionary containing values for all independent parameters
        fix_values : dict
            additional fixed values (same as for e.g. `!run()`).

        Returns
        -------
        dict
            with keys for all parameters of the fit, filled in according to
            fixed values
        """
        mt = self.MinTarget(self, fix_values=fix_values)
        return mt.params_array2dict(mt.params_dict2array(params))

    def MSD(self, params, dt=None):
        """
        Return the (trend-free) MSD evaluated at dt. Convenience function.

        Parameters
        ----------
        params : dict
            the current parameter values. Like the ``'params'`` entry of the
            output dict from any fit.
        dt : array-like
            the time lags (in frames) at which to evaluate the MSD. If left
            unspecified, we return a callable MSD function
        kwargs : additional keyword arguments
            are forwarded to the MSD functions returned by `params2msdm`.

        Returns
        -------
        callable or np.array
            the MSD function (summed over all dimensions), evaluated at `!dt` if provided.

        Notes
        -----
        This function returns the "trend-free" MSD; meaning we ignore potential
        values for the mean/trend parameters `!m1`. For `!ss_order == 0` this
        is correct behavior either way; for `!ss_order == 1` the trend term can
        be added as ``(m1*Δt)^2``.
        """
        def msdfun(dt, params=params, **kwargs):
            msdm = self.params2msdm(params)
            return np.sum([msd(dt, **kwargs) for msd, m in msdm], axis=0)

        if dt is None:
            # Write proper signature to docstring
            single_msdfun = self.params2msdm(params)[0][0]
            if hasattr(single_msdfun, '_kwargstring'):
                msdfun.__doc__ = f"""full signature:

msdfun(dt,
       params={params},
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
        params : dict

        Returns
        -------
        float
            anything < 0 indicates infeasibility, so don't even evaluate the
            likelihood
        """
        x = np.inf

        # Check parameter bounds
        # We can just detect infeasibility; if a soft bound is needed, implement
        # a constraint
        for name, P in self.parameters.items():
            if (   P.bounds[0] > params[name]
                or P.bounds[1] < params[name]):
                return -1 # infeasible

        # Check inequality constraints
        for constraint in self.constraints:
            x = min(constraint(params), x)
            if x <= 0:
                return -1 # infeasible

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
        Assemble full list of values to fix (internal + given)

        Merge the given `!fix_values` with the parameter-level specifications
        and ensure that all entries are appropriate callables (can also be
        specified as numerical constants or other parameter names)

        Parameters
        ----------
        fix_values : dict, optional
            values to fix, beyond what's already in
            ``self.parameters[...].fix_to``

        Returns
        -------
        fix_values : dict
            same as input, plus internals, and constants resolved

        Notes
        -----
        Parameter resolution order:

         + (marginalization)
         + free parameters
         + fixed to constant
         + fixed to other parameter by name
         + fixed to callable

        So you cannot fix to another parameter that is itself fixed to a
        callable. You can use ``parameter.fix_to = other_parameter.fix_to``
        instead to just copy the function over instead of fixing by name. This
        helps prevent undetectable cycles in the fixes.
        """
        # Merge input and fit-internal fix_values (input takes precedence)
        # Note: using the dict here ensures uniqueness (only one fix per
        # parameter)
        fix_values_in = fix_values if fix_values is not None else {}
        fix_values = {name : param.fix_to
                      for name, param in self.parameters.items()
                      }
        fix_values.update(fix_values_in)
        fix_values = {name : val for name, val in fix_values.items() if val is not None}

        # Keep track of which parameter is resolved how, such that we can give
        # a good resolution order later
        marginalized = [] # just names
        unfixed      = [] # just names
        to_constant  = [] # (name, constant)
        to_other     = [] # (name, name2)
        to_callable  = [] # (name, callable)

        for name in self.parameters:
            try:
                fix_to = fix_values[name]
            except KeyError:
                unfixed.append(name)
                continue

            if fix_to == _MARGINALIZE:
                marginalized.append(name)
            elif callable(fix_to):
                to_callable.append((name, fix_to))
            elif fix_to in self.parameters:
                to_other   .append((name, fix_to))
            else:
                to_constant.append((name, fix_to))

        # Sort out the order for fixing to other parameters
        determined = marginalized + unfixed + [name for name, _ in to_constant]
        to_other_ordered = []
        while len(to_other) > 0:
            cache_len = len(to_other)
            i = 0
            while i < len(to_other):
                if to_other[i][1] in determined:
                    name, fix_to = to_other.pop(i)
                    to_other_ordered.append((name, fix_to))
                    determined.append(name)
                else:
                    i += 1

            if len(to_other) == cache_len:
                # Could not insert any more of the identifications, so there
                # must be a cycle in the remaining fixes (or something depends
                # on a parameter that's fixed by callable)
                self.vprint(1, "Left-overs from resolution order determination:\n"
                              f"      determined = {determined}\n"
                              f"        to_other = {to_other}\n"
                              f"     to_callable = {to_callable}\n")
                raise RuntimeError("Could not determine resolution order of parameter fixes.")

        assert len(determined) + len(to_callable) == len(self.parameters)
        return marginalized, unfixed, to_constant, to_other_ordered, to_callable

    def independent_parameters(self, fix_values=None):
        """
        Return the names of the independent (not fixed) parameters

        Parameters
        ----------
        fix_values : dict
            should be the same as was / will be handed to `run`

        Returns
        -------
        list of str
            a list with the names of the independent parameters, i.e. those
            that are not fixed to some other value
        """
        if fix_values is None:
            fix_values = {}

        def is_free(name):
            try:
                return fix_values[name] is None
            except KeyError:
                return self.parameters[name].fix_to is None

        return [name for name in self.parameters if is_free(name)]
                
    class MinTarget:
        """
        Helper class; acts as cost function for optimization.

        Beyond acting as the actual cost function through implementation of the
        ``()`` operator, this class also defines the conversion between
        parameter dicts (where parameters are named and unordered) and
        parameter arrays (which are used for optimization). Most of this is for
        internal use.

        Parameters
        ----------
        fit : Fit
            the `Fit` object that this target is associated with.
        fix_values : dict
            any additional fixes for parameters, beyond what's already in
            ``fit.parameters[...].fix_to``.
        offset : float
            the global offset to subtract; see Notes section of `Fit`.
        adjust_prior_for_fixed_values : bool
            if ``False``, pretend that the parameters in `fix_values` are still
            free. Important for the `Profiler`.

        Attributes
        ----------
        fit : Fit
            the `Fit` object that this target is associated with.
        fix_values : dict
            the complete set of values to keep fixed; keys are parameter names,
            values are (guaranteed to be) callables.
        fix_values_resolution_order : list of str
            list of parameter names. The fixes in `!fix_values` will be
            executed in this order.
        param_names : list of str
            list of parameter names that remain independent; this determines
            the order of entries when converting parameter dicts to arrays.
        offset : float
            constant to be subtracted from the target (i.e. added to the
            likelihood). Can be used to ensure that the minimum value of the
            target is close to 0.
        xatol : float
            set this to a finite value to make MinTarget check its past
            evaluations and report (suspected) convergence. For internal use.
        N_xatol : int
            how far to look back when assessing xatol-convergence. Default:
            2x(number of independent fit parameters)
        memo_evals : list
            evaluation history. Can come in handy when the fit aborts for
            whatever reason
        """
        def __init__(self, fit, *, fix_values=None, offset=0,
                     adjust_prior_for_fixed_values=True,
                     ):
            self.fit = fit
            self.likelihood_chunksize = self.fit.likelihood_chunksize
            self.maxfev = self.fit.maxfev

            # See class docstring
            fv = self.fit.expand_fix_values(fix_values)
            self.params_marginalized = fv[0]
            self.params_free         = fv[1]
            self.params_to_constant  = fv[2]
            self.params_to_other     = fv[3]
            self.params_to_callable  = fv[4]

            self.offset = offset

            self.xatol = np.nan # workaround for convergence; check Fit.run()
            self.N_xatol = max(10, 2*len(self.params_free))
            self.memo_evals = []
            
            self.paramnames_prior = self.params_free
            if not adjust_prior_for_fixed_values:
                self.paramnames_prior += list(fix_values.keys())

            if len(self.params_marginalized) > 0:
                self.margev_fit = copy(self.fit) # shallow, to prevent data copying
                self.marg_ev_mci = (-np.inf, None) # remember best evaluation

                self.margev_fit.parameters = deepcopy(fit.parameters) # so we can override
                for name in self.params_marginalized:
                    self.margev_fit.parameters[name].fix_to = None
                for name in self.params_free:
                    self.margev_fit.parameters[name].fix_to = np.nan
            else:
                self.margev_fit = None # no marginalization to do
                self.marg_ev_mci = (-np.inf, {})

        class xatolConverged(Exception):
            pass

        def params_array2dict(self, params_array):
            """
            Convert a parameter array to dict

            Parameters
            ----------
            params_array : np.ndarray

            Returns
            -------
            dict

            See also
            --------
            params_dict2array
            """
            params = dict(zip(self.params_free, params_array))
            for name in self.params_marginalized:
                params[name] = np.nan
            for name, val in self.params_to_constant:
                params[name] = val
            for name, other in self.params_to_other:
                params[name] = params[other]
            for name, fun in self.params_to_callable:
                try:
                    params[name] = fun(params, name=name)
                except TypeError: # 'name' not a kwarg
                    params[name] = fun(params)

            return params

        def params_dict2array(self, params):
            """
            Convert a parameter dict to array

            Parameters
            ----------
            params_dict : dict

            Returns
            -------
            np.ndarray

            See also
            --------
            params_array2dict
            """
            return np.array([params[name] for name in self.params_free])

        def eval_atomic(self, params_array):
            """
            Evaluate likelihood, once all parameters are known (no marginalization)

            This function evaluates penalty, prior, and likelihood for the
            current parameters and returns the negative of their sum. If
            ``penalty < 0``, instead of evaluating the likelihood, we just
            return the maximum penalization.

            Parameters
            ----------
            params_array : np.ndarray

            Returns
            -------
            float
            """
            params = self.params_array2dict(params_array)
            params_prior = {name : params[name] for name in self.paramnames_prior}

            for name, val in params.items():
                if np.isnan(val):
                    raise ValueError(f"Got NaN value for parameter '{name}'")

            penalty = self.fit._penalty(params)
            if penalty < 0: # infeasible
                return self.fit.max_penalty
            else:
                self.fit.data.restoreSelection(self.fit.data_selection)
                try:
                    logL = GP.ds_logL(self.fit.data,
                                      self.fit.ss_order,
                                      self.fit.params2msdm(params),
                                      chunksize=self.likelihood_chunksize,
                                      )
                except GP.BadCovarianceError as err:
                    # This should not really happen
                    # It means that the covariance matrix was not positive
                    # definite, and its Fit's job to ensure that it is.
                    # However, due to numerical inaccuracies (especially with
                    # evidence() or the profiler, who explore wider ranges of
                    # parameter space) or user error when implementing custom
                    # Fits, this can still happen sometimes. We can attempt to
                    # rescue it by just returning max_penalty.
                    self.fit.vprint(1, "BadCovarianceError:", err)
                    return self.fit.max_penalty

                return (- logL
                        - self.fit.logprior(params_prior)
                        + penalty
                        - self.offset
                        )

        def __call__(self, params_array):
            """
            Actual minimzation target

            Recurses if parameters are marginalized; otherwise just calls
            `eval_atomic`.

            Parameters
            ----------
            params_array : np.ndarray

            Returns
            -------
            float
            """
            if len(self.memo_evals) >= self.maxfev:
                raise RuntimeError(f"Exceeded maxfev = {self.maxfev} function evaluations")

            if self.margev_fit is None:
                out = self.eval_atomic(params_array)
            else:
                params = self.params_array2dict(params_array)
                params_prior = {name : params[name] for name in self.paramnames_prior}

                for name, val in zip(self.params_free, params_array):
                    self.margev_fit.parameters[name].fix_to = val

                ev_chunksize = self.likelihood_chunksize
                if len(self.params_marginalized) <= 1:
                    # switch off parallelization in evidence, since it would be
                    # at most 2 parallel evaluations anyways; this will allow
                    # margev_fit to parallelize its likelihood
                    ev_chunksize = -1
                ev, mci = self.margev_fit.evidence(likelihood_chunksize=ev_chunksize,
                                                   return_mci=True,
                                                   )

                if ev > self.marg_ev_mci[0]: # remember best evaluation
                    self.marg_ev_mci = (ev, mci)

                out = (- ev
                       - self.fit.logprior(params_prior)
                       - self.offset
                       )

            self.memo_evals.append((out, params_array))
            if np.isfinite(self.xatol) and len(self.memo_evals) >= self.N_xatol:
                history = np.array([p for _, p in self.memo_evals[-self.N_xatol:]])
                xmean = np.mean(history, axis=0)
                delta = np.abs(history - xmean[None])
                if np.max(delta) < self.xatol:
                    raise self.xatolConverged(f"Fit parameters converged to better than xatol = {self.xatol}. Check MinTarget.memo_evals for evaluations")

            return out

        def profile_marginalized_params(self, params):
            """
            Run the profiler over the marginalized parameters

            Parameters
            ----------
            params : dict
            
            Returns
            -------
            dict
                for each marginalized parameter: point estimate and 95%
                credible interval
            """
            if self.margev_fit is None:
                return dict()

            params_array = self.params_dict2array(params)
            for name, val in zip(self.params_free, params_array):
                self.margev_fit.parameters[name].fix_to = val

            profiler = Profiler(self.margev_fit,
                                profiling=len(self.params_marginalized) > 1,
                                ) # ^ suppress warning for single param
            return profiler.find_MCI()

    @method_verbosity_patch
    def run(self,
            init_from = None,
            optimization_steps=('simplex',),
            maxfev=None,
            xatol=1e-5,
            fix_values = None,
            adjust_prior_for_fixed_values=True,
            give_rough_marginal_mci=False,
            full_output=False,
            show_progress=False,
           ):
        """
        Run the fit

        Parameters
        ----------
        init_from : dict
            initial point for the fit, as a dict with fields ``'params'`` and
            ``'logL'``, like the ones this function returns. The ``'logL'``
            entry is optional; you can also leave it out or set to 0.
        optimization_steps : tuple
            successive optimization steps to perform. Entries should be
            ``'simplex'`` for Nelder-Mead, ``'gradient'`` for gradient descent,
            or a dict whose entries will be passed to
            ``scipy.optimize.minimize`` as keyword arguments.
        maxfev : int or None
            limit on function evaluations for ``'simplex'`` or ``'gradient'``
            optimization steps (overrides `!self.maxfev`)
        xatol : float
            convergence criterion in parameter space; should be relevant only
            in special cases. Can be set to ``np.nan`` to disable.
        fix_values : dict
            can be used to keep some parameter values fixed or express them as
            function of the other parameters. See class doc for more details.
        adjust_prior_for_fixed_values : bool
            whether to evaluate the model prior for the restricted parameter
            space resulting from `fix_values` or over the full parameter space.
            While the former is "correct" in most cases, the latter is
            important if we want to explore the likelihood landscape around a
            previously found optimum, as the `Profiler` does.
        give_rough_marginal_mci : bool
            when marginalizing parameters, instead of doing a full run of the
            profiler at the end, just return the profiler results from
            evidence(). Note that the credible intervals will be less precise
            on these; point estimates should be fine.
        full_output : bool
            Set to ``True`` to return the output dict (c.f. Returns) and the
            full output from ``scipy.optimize.minimize`` for each optimization
            step. Otherwise (``full_output == False``, the default) only the
            output dict from the final run is returned.
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
            initial_params = self.initial_params()
            initial_offset = self.initial_offset()
        else:
            initial_params = deepcopy(init_from['params'])
            try:
                initial_offset = -init_from['logL']
            except KeyError:
                initial_offset = 0

        # Set up the minimization target
        # also allows us to convert initial_params to appropriate array
        min_target = self.MinTarget(self,
                                    fix_values=fix_values,
                                    offset=initial_offset,
                                    adjust_prior_for_fixed_values=adjust_prior_for_fixed_values,
                                    )
        if maxfev is not None:
            min_target.maxfev = maxfev

        p0 = min_target.params_dict2array(initial_params)
        bounds = [self.parameters[name].bounds for name in min_target.params_free]
        
        # Set up progress bar
        bar = tqdm(disable = not show_progress, desc='fit iterations')
        def callback(x):
            bar.update()

        # Go!
        all_res = []
        with np.errstate(all='raise'):
            for istep, step in enumerate(optimization_steps):
                if step == 'simplex':
                    # The stopping criterion for Nelder-Mead is max |f_point -
                    # f_simplex| < fatol AND max |x_point - x_simplex| < xatol.
                    # 
                    # In principle we should switch off the xatol mechanism and
                    # just "listen" to fatol, i.e. variations in the
                    # log-likelihood. However, in some instances (e.g. with
                    # marginalized parameters, where "log-likelihood" in fact
                    # is numerically integrated evidence), the likelihood
                    # landscape might be somewhat noisy (in the cases that I
                    # studied, this was quite minor, with likelihoods
                    # fluctuating on the scale of 0.01). In those cases, the
                    # fit will usually converge to a decent parameter set, but
                    # is unable to pin down the likelihood to the required
                    # precision; it would thus run forever. For these cases, it
                    # makes sense to set xatol, but with the convergence
                    # criterion being fatol OR xatol. Thus the actual
                    # implementation goes through MinTarget. Also note that fit
                    # parameters should numerically all be O(1) anyways, so
                    # fixing an absolute value here is fine-ish; not the most
                    # beautiful solution, but a practical one.
                    # 
                    # Note that fatol is "just" a threshold on the decrease
                    # relative to the last evaluation, not a strict bound. So
                    # the Profiler might actually still find a better estimate,
                    # even though it uses the same tolerance (1e-3).
                    options = {'fatol' : 1e-3,
                               'xatol' : np.inf,
                               'maxfev' : 1e10, # taken care of by MinTarget
                               }
                    kwargs = dict(method = 'Nelder-Mead',
                                  options = options,
                                  bounds = bounds,
                                  callback = callback,
                                 )
                elif step == 'gradient':
                    options = {'maxfun' : 1e10}
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
                    
                min_target.xatol = xatol
                try:
                    if len(p0) > 0:
                        fitres = optimize.minimize(min_target, p0, **kwargs)
                    else:
                        fitres = lambda : None # same trick as below
                        fitres.fun = min_target(p0)
                        fitres.x = p0
                        fitres.success = True
                except GP.BadCovarianceError as err: # pragma: no cover
                    self.vprint(2, "BadCovarianceError:", err)
                    fitres = lambda: None # hack: lambdas allow free assignment of attributes
                    fitres.success = False
                except min_target.xatolConverged:
                    all_f = np.array([f for f, _ in min_target.memo_evals])
                    i_best = np.argmin(all_f)
                    fitres = lambda : None
                    fitres.fun = min_target.memo_evals[i_best][0]
                    fitres.x   = min_target.memo_evals[i_best][1]
                    fitres.success = True

                    df = np.mean(all_f[-min_target.N_xatol:]-fitres.fun)
                    self.vprint(1, f"Fit parameters converged (xatol = {min_target.xatol}); mean likelihood deviation over last {min_target.N_xatol} evaluations was {df:.3g}.")
                
                if not fitres.success: # pragma: no cover # got rare since we moved maxfev to Fit
                    self.vprint(1, f"Fit (step {istep}: {step}) failed. Here's the result:")
                    self.vprint(1, '\n', fitres)
                    raise RuntimeError("Fit failed at step {:d}: {:s}".format(istep, step))
                else:
                    params = min_target.params_array2dict(fitres.x)
                    all_res.append(({'params' : params,
                                     'marginalized' : (min_target.marg_ev_mci[1]
                                                       if give_rough_marginal_mci else
                                                       min_target.profile_marginalized_params(params)
                                                       ),
                                     'logL' : -(fitres.fun+min_target.offset),
                                    }, fitres))
                    p0 = fitres.x
                    min_target.offset += fitres.fun
                    
        bar.close()
        
        if full_output:
            return all_res
        else:
            return all_res[-1][0]

    @parallel.chunky('likelihood_chunksize', -1)
    def evidence(self, show_progress=False,
                 conf = 0.8,
                 conf_tol = 0.1,
                 n_cred = 10,
                 n_steps_per_cred = 2,
                 f_integrate = 0.99,
                 log10L_improper = 3,
                 return_mci=False,
                 return_evaluations=False,
                 init_from_params=None,
                ):
        """
        Estimate evidence for this model

        The parameters to this function are mostly technical and can remain at
        their default values for most use cases. It is possible that in some
        cases speedups can be achieved by tuning them, but generally the
        defaults should be reasonable enough.

        Parameters
        ----------
        show_progress : bool
            display progress bar(s)
        conf : float
            confidence level for the initial `Profiler` run. Will usually be
            some not-too-high value.
        conf_tol : float
            tolerance for `conf`. Usually relatively high.
        n_cred : float
            how far (at most) from the point estimate to evaluate the
            likelihood function, in multiples of the initial Profiler credible
            interval.
        n_steps_per_cred : float
            how many grid points to place between point estimate and boundary
            of initial credible interval.
        f_integrate : float
            proxy parameter to determine the threshold until which the
            likelihood will be evaluated. Can be interpreted as "if the
            posterior was Gaussian, we would cover at least a fraction f of its
            mass".
        log10L_improper : float
            width of surrogate prior for improper priors. See Notes.
        likelihood_chunksize : int
            see class description; here a chunksize of 1 corresponds to a
            single call to the fit likelihood.
        return_mci : bool
            return the results of the initial profiler run. Note that these are
            conditional posterior and with imprecise confidence settings (see
            `!conf` and `!conf_tol` below).
        return_evaluations : bool
            return the grid of evaluated likelihood points. If set, we return a
            tuple ``(xi, logL, logprior)`` where ``xi`` are the parameter
            vectors whose product forms the grid; ``logL`` are the evaluations
            of the fit "likelihood" and ``logprior`` is the associated
            parameter space. Note that the terms "likelihood" and "prior" are
            to be regarded with caution here, since proper priors are included
            in ``logL``, while improper priors are in ``logprior``. In any
            case, ``logL + logprior`` gives the unnormalized (log-)posterior
            whose integral is the evidence.
        init_from_params : dict or None
            start the initial fit from here. Can be useful when running on
            single (or few) trajectories, where the fit might not converge
            well; in those cases, run the profiler to find the initial point
            estimate and specify it here.

        Returns
        -------
        float
            estimated evidence for this model
        dict, optional
            the profiler result, if ``return_mci == True``
        tuple
            the evaluation grid, if ``return_evaluations == True``

        Notes
        -----
        This function estimates model evidence by integrating the likelihood
        over a non-uniform grid. The outline of the algorithm is as follows:

         + run the fit to get a point estimate
         + run a `Profiler` (in conditional posterior mode) to get order of
           magnitude estimates for the steps in different directions over which
           the likelihood changes. This run is controlled by `conf` and
           `conf_tol`; since we do not need precise estimates, but just an idea
           of how big steps should generally be, the default is to search with
           relatively low accuracy (high `conf_tol`) for a not-too-strong
           decrease in likelihood (medium values for `conf`).
         + given the profiler results, assemble a grid of parameter values over
           which the likelihood may be evaluated. This step is controlled by
           `n_cred` (how big to make the grid) and `n_steps_per_cred` (how fine
           to make the grid)
         + start by evaluating the grid completely within the bounds given by
           the profiler
         + grow the integration region iteratively: at each step, find the grid
           points that are adjacent to another one that a) has been evaluated
           already and b) has a value above the cutoff controlled by
           `f_integrate`. Evaluate the likelihood on all these candidate
           points; repeat
         + with the likelihood evaluated at all relevant grid points, integrate
           numerically by multiplying the voxel volume and summing.

        For the resulting evidence value to be accurate and comparable across
        different models, it is tantamount that the priors be properly
        normalized. This means that a) use this function only with models that
        implement the `logprior` function correctly; b) models with improper
        priors (e.g. for localization error we usually use a log-flat prior)
        are problematic. We sneak around the latter issue with a debatable
        method: for the cases where we have improper priors, we fix a suitable,
        broad, and importantly proper prior on the location of the point
        estimate. This approach is clearly suspicious, because we fix the
        *prior* to a result from the data. One might argue that at the end of
        the day we are basically "just" fixing the order of magnitude (think:
        units) of a given parameter, which we might also do by checking the
        experimental protocol. But at the end of the day, it remains a
        questionable method; any reader with better ideas for how to handle
        this case should email me.

        Having accepted the dubious approach to improper priors: the surrogate
        (proper) prior we finally use is a Gaussian with standard deviation
        ``log(10)*log10L_improper``. The idea here is that if the improper
        prior is log-flat (my most common use case), the parameter
        `log10L_improper` basically gives the number of valid digits in a
        parameter estimate that we would accept as "not fine tuned".
        Specifically: consider "fitting" a model without any parameters; if I
        can improve upon that fit by introducing one new parameter with the
        value 17, we might accept the second model as reasonable; if the fit
        only improves if I fix the new parameter to 17.2193427, we might say
        that the new model requires too much fine tuning. Upon personal
        reflection, ``log10L_improper = 3`` valid digits seems to be
        reasonable; reasonable people might differ.
        """
        names = self.independent_parameters()
        if len(names) == 0:
            # no free parameters; just return likelihood
            return self.MinTarget(self)(np.array([]))

        # Note: line_profiler shows that chi2.ppf() is quite slow; amounting to
        # ~5% of total runtime on an example trajectory
        DlogL = stats.chi2(df=len(names)).ppf(f_integrate)/2
        n_steps = np.round(n_steps_per_cred*n_cred).astype(int)
        sigma_improper = np.log(10)*log10L_improper

        # Run profiler
        profiler = Profiler(self, profiling=False, conf=conf, conf_tol=conf_tol)
        profiler.verbosity = 0 # suppress everything, would just be confusing anyways
        if init_from_params is not None:
            pe = {}
            pe['params'] = self.fill_dependent_params(init_from_params)
            pe['logL']   = self.logL(pe['params'])
            profiler.point_estimate = pe

        mci = profiler.find_MCI(show_progress=show_progress)
        assert set(mci.keys()) == set(names)
        
        # Adjust for improper priors
        def compactify(x, x0):
            return special.erf((x-x0)/sigma_improper)
        def decompactify(y, x0):
            return special.erfinv(y)*sigma_improper + x0

        impropers = [name for name in names if name in self.improper_priors]
        compactified_bounds = {} # bounds for impropers, that are guaranteed to be safe under decompactify
        for name in impropers:
            x0 = mci[name][0]
            bounds = compactify(self.parameters[name].bounds, x0)
            while decompactify(bounds[0], x0) < self.parameters[name].bounds[0]:
                bounds[0] *= 0.99
            while decompactify(bounds[1], x0) > self.parameters[name].bounds[1]:
                bounds[1] *= 0.99
            compactified_bounds[name] = bounds

            ci = compactify(mci[name][1], x0)
            ci[0] = max(bounds[0], ci[0])
            ci[1] = min(bounds[1], ci[1])

            mci[name+'_orig'] = mci[name]
            mci[name] = (0., ci)

        # Assemble parameter grid
        xi = []
        for name in names:
            x_point = mci[name][0]
            
            x_lo = x_point + n_cred*(mci[name][1][0]-x_point)
            x_min = self.parameters[name].bounds[0]
            if name in impropers:
                x_min = compactified_bounds[name][0]
            
            x_hi = x_point + n_cred*(mci[name][1][1]-x_point)
            x_max = self.parameters[name].bounds[1]
            if name in impropers:
                x_max = compactified_bounds[name][1]
            
            x = np.concatenate([np.linspace(x_lo, x_point, n_steps+1)[:-1],
                                [x_point],
                                np.linspace(x_point, x_hi, n_steps+1)[1:],
                               ])
            
            # make sure that we evaluate only in-bounds grid points
            # Note: for parameters with proper parameters, this is actually
            # unnecessary, since the likelihood will just return the penalty
            # term. But for parameters with improper priors we need to ensure
            # that we only use valid values; so let's just do it for all of
            # them together.
            # Note: it is important to use <= (not just <), for cases where the
            # point estimate sits on the boundary of the parameter range. In
            # those cases the whole positive half of x will be equal (to 0), so
            # delta == 0.
            delta = 0.1*(x[1]-x[0])
            ind = np.nonzero(x-x_min <= delta)[0] # too small
            if len(ind) > 0:
                x[ind] = np.nan
                x[ind[-1]] = x_min
                
            delta = 0.1*(x[-1]-x[-2])
            ind = np.nonzero(x_max-x <= delta)[0] # too large
            if len(ind) > 0:
                x[ind] = np.nan
                x[ind[0]] = x_max
                
            xi.append(x)

        # Calculate parameter space volume (i.e. prior) for each grid cell
        # We evaluate grid cells at x[i]. The associated volume is the sum of
        # half the distance to the previous grid point and half the distance to
        # the next; the grid points at the limits get only half a cell.
        #
        #   v              |           v              |           v              |   ...   |           v  
        #   X[0]===Δ+[0]===|===Δ-[1]===X[1]===Δ+[1]===|===Δ-[2]===X[2]===Δ+[2]===|===...===|===Δ-[N]===X[N]
        #   ^              |           ^              |           ^              |   ...   |           ^
        #   |             Δ[0]         |             Δ[1]         |             Δ[2] ...  Δ[N-1]       |
        #
        logprior = np.array(0)
        for i, (name, x) in enumerate(zip(names, xi)):
            dx = np.diff(x)
            dx_m = np.insert(0.5*dx, 0, 0)
            dx_p = np.append(0.5*dx, 0)

            # Make sure to correctly account for nan's (i.e. parameter bounds)
            dx_m[np.isnan(x) | np.isnan(dx_m)] = 0.
            dx_p[np.isnan(x) | np.isnan(dx_p)] = 0.

            with np.errstate(divide='ignore'): # log(0) for dx=nan=0
                logprior = logprior[..., None] + np.log(dx_p+dx_m)

            if name in impropers: # others should be accounted for in self.logprior() !
                logprior -= np.log(2)
            
        # Set up likelihood evaluations
        i_center = (logprior.shape[0]-1)//2
        assert i_center == n_steps
        assert np.all(np.array(logprior.shape) == 2*i_center+1)

        logL = np.empty(logprior.shape, dtype=float)
        logL[:] = -np.inf
        
        # Decompactify grid values
        for i, name in enumerate(names):
            if name in impropers:
                xi[i] = decompactify(xi[i], mci[name+'_orig'][0])

        # Progress display
        bar = tqdm(disable = not show_progress, desc='evidence integration')

        # (potentially parallel) likelihood evaluations
        neg_logL = self.MinTarget(self)
        if parallel._chunksize >= 0:
            neg_logL.likelihood_chunksize = -1 # no nested parallelization
        def eval_log_likelihood(ilist):
            # ilist : (N, len(xi)), dtype=int
            #     indices into xi
            xlist = np.array([[x[i] for x, i in zip(xi, ind)] for ind in ilist])
            ind_nans = np.any(np.isnan(xlist), axis=-1)

            for ind in ilist[ind_nans]:
                # Important for detecting that we did (attempt to) evaluate this point
                logL[tuple(ind)] = -self.max_penalty

            xlist = xlist[~ind_nans]
            ilist = ilist[~ind_nans]

            paramlist = [neg_logL.params_dict2array(
                            neg_logL.params_array2dict(len(names)*[np.nan])
                            | dict(zip(names, x)))
                         for x in xlist]

            imap = parallel._map(neg_logL, paramlist)
            for ind, nlogL in zip(ilist, imap):
                logL[tuple(ind)] = -nlogL
                bar.update()

        # Evaluate central grid points (those within the estimated credible interval)
        igrid = np.meshgrid(*len(xi)*[np.arange(-n_steps_per_cred, n_steps_per_cred+1)])
        ilist = np.stack([g.flatten() for g in igrid], axis=-1) + i_center
        eval_log_likelihood(ilist)

        # Iterative evaluation until we leave the maximum
        while True:
            logL_max = np.max(logL)
            candidates = -np.inf*np.ones(logL.shape, dtype=float)
            for ax in range(len(logL.shape)):
                ind_c = len(logL.shape)*[slice(None)]
                ind_L = len(logL.shape)*[slice(None)]

                ind_c[ax] = slice(1, None)
                ind_L[ax] = slice(None, -1)
                candidates[tuple(ind_c)] = np.maximum(candidates[tuple(ind_c)], logL[tuple(ind_L)])

                ind_c[ax] = slice(None, -1)
                ind_L[ax] = slice(1, None)
                candidates[tuple(ind_c)] = np.maximum(candidates[tuple(ind_c)], logL[tuple(ind_L)])

            candidates[logL > -np.inf] = -np.inf

            ilist = np.stack(np.nonzero(candidates > logL_max-DlogL), axis=-1)
            if len(ilist) == 0:
                break

            eval_log_likelihood(ilist)

        bar.close()

        # Integrate likelihood to find evidence
        with np.errstate(under='ignore'):
            ev = special.logsumexp(logL + logprior)

        out = [ev]
        if return_mci:
            out.append(mci)
        if return_evaluations:
            out.append((xi, logL, logprior))

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)
