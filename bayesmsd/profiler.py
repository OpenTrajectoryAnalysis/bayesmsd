"""
Implementation of the `Profiler` class

This is used to "explore the posterior" after running a `Fit <bayesmsd.Fit>`,
specifically to find the credible intervals.

Example
-------
>>> fit = <some bayesmsd.Fit subclass, c.f. bayesmsd.lib>
... profiler = bayesmsd.Profiler(fit)
... mci = profiler.find_MCI() # mci = max posterior estimate & credible
...                           # interval boundaries for each parameter

See also
--------
bayesmsd, Profiler, Fit <bayesmsd.fit.Fit>
"""
from tqdm.auto import tqdm

import numpy as np
from scipy import stats

class Profiler():
    """
    Exploration of the posterior after finding the point estimate

    This class provides a top layer on top of `Fit`, enabling more
    comprehensive exploration of the posterior after finding the (MAP) point
    estimate. Generally, it operates in two modes:

    - conditional posterior: wiggle each individual parameter, keeping all
      others fixed to the point estimate values, thus calculating conditional
      posterior values. In parameter space, this is a (multi-dimensional) cross
      along the coordinate axes.
    - profile posterior: instead of simply evaluating the posterior at each
      wiggle, keep the parameter fixed at the new value and optimize all
      others. Thus, in parameter space we are following the ridges of the
      posterior.

    From a Bayesian point of view, beyond conditional and profile posterior,
    the actually interesting quantity is of course the marginal posterior. This
    is best obtained by sampling the posterior by MCMC (see module description
    for an example implementation of such a sampler). Still, conditional or
    profile posterior often can give a useful overview over the shape of the
    posterior. The conditional posterior is also great for setting MCMC step
    sizes. Profile posteriors are of course significantly more expensive than
    conditionals.

    At the end of the day, this class moves along either profile or conditional
    posterior until it drops below a given cutoff, and then gives the lower
    bound, best value, and upper bound for each parameter. Inspired by
    frequentist confidence intervals, we determine the cutoff from a
    "confidence level" 1-α.

    We follow a two-step procedure to find the lower and upper bounds: first,
    we move out from the point estimate with a fixed step size (additively or
    multiplicatively) until the posterior drops below the cutoff. Within the
    thus defined bracket, we then find the actual bound to the desired accuracy
    by bisection. If the first step fails (i.e. the posterior does not drop
    below the cutoff far from the point estimate) the corresponding parameter
    direction is unidentifiable and the bound will be set to infinity
    
    Note that this class will also take care of the initial point estimate, if
    necessary.

    The main point of entry for the user is the `find_MCI` method, that
    performs the sweeps described above. Further, `run_fit` might be useful if
    you just need the point estimate.

    Parameters
    ----------
    fit : Fit
        the `Fit` object to use
    profiling : bool
        whether to run in profiling (``True``) or conditional (``False``) mode.
    conf : float
        the "confidence level" to use for the likelihood ratio cutoff
    conf_tol : float
        acceptable tolerance in the confidence level
    bracket_strategy : 'auto', dict, or list of dict
        specifies the strategy to use during the first step (pushing the
        parameter out to find a proper bracket). For full control, specify a
        list of dicts (one for each parameter) with the structure
        ``dict(multiplicative=(bool), step=(float),
        nonidentifiable_cutoffs=(low, high))``, where ``multiplicative``
        indicates whether propagation is ``param <-- param*step`` or ``param
        <-- param + step``. Correspondingly, ``step`` should be ``>1`` for
        ``multiplicative == True``, and ``step > 0`` for ``multiplicative ==
        False``. Finally, ``nonidentifiability_cutoffs`` specifies how far to
        search before the direction is considered unidentifiable. This is
        measured in "total stepsize" taken in either direction, so e.g. for a
        multiplicative strategy the default is ``nonidentifiability_cutoffs =
        (10, 10)``, meaning we search by a factor of 10 around the point
        estimate in either direction.
        
        Instead of specifying a list of dicts, you can also just give a single
        dict that will be applied to all parameters. Finally, by default this
        argument is set to ``'auto'``, which means that the strategy is
        determined from the bounds in `!fit`. If the lower bound is positive,
        the strategy is multiplicative, else additive.
    bracket_step : float
        global default for the ``'step'`` parameter if ``bracket_strategy ==
        'auto'``. Additive strategies use ``step = point_estimate *
        (bracket_step - 1)``.
    max_fit_runs : int
        an upper bound for how often to re-run the fit (when profiling).
    max_restarts : int
        sometimes the initial fit to find the point estimate might not converge
        properly, such that a better point estimate is found during the
        profiling runs. If that happens, the whole procedure is restarted from
        the new point estimate. This variable provides an upper bound on the
        number of these restarts. See also `!restart_on_better_point_estimate`
        below.
    verbosity : {0, 1, 2, 3}
        controls amount of messages during profiling. 0: nothing; 1: warnings
        only; 2: informational; 3: debugging

    Attributes
    ----------
    fit : Fit
        see Parameters
    min_target_from_fit : callable
        the minimization target of ``self.fit``
    ress : list
        storage of all evaluated parameter points. Each entry corresponds to a
        parameter dimension and contains a list of dicts that were obtained
        while sweeping that parameter. The individual entries are dicts like
        the one returned by `Fit.run`, with fields ``'params'`` and ``'logL'``.
    point_estimate : dict
        the MAP point estimate, a dict like the other points in `!ress`.
    conf, conf_tol : float
        see Parameters
    LR_target : float
        the target decline in the posterior value we're searching for
    LR_interval : [float, float]
        acceptable interval corresponding to `!conf_tol`
    iparam : int
        the index of the parameter currently being sweeped. This is a class
        attribute mainly for readability of the code.
    profiling : bool
        see Parameters
    bracket_strategy, bracket_step : list, float
        see Parameters
    max_fit_runs : int
        see Parameters
    run_count : int
        counts the runs executed so far
    max_restarts_per_parameters : int
        same as `!max_restarts` parameter.
    verbosity : int
        see Parameters
    restart_on_better_point_estimate : bool
        whether to restart upon finding a better point estimate. Might make
        sense to disable if your posterior is rugged
    bar : tqdm progress bar
        the progress bar showing successive fit evaluations

    See also
    --------
    run_fit, find_MCI
    """
    def __init__(self, fit,
                 profiling=True,
                 conf=0.95, conf_tol=0.001,
                 bracket_strategy='auto', # 'auto', dict(multiplicative=(bool),
                                          #              step=(float>1),
                                          #              nonidentifiable_cutoffs=[(low, high)],
                                          #             ),
                                          #         or list of such (one for each parameter)
                 bracket_step=1.2, # global default, but would be overridden by specifying bracket_strategy
                 max_fit_runs=100,
                 max_restarts=10,
                 verbosity=1, # 0: print nothing, 1: print warnings, 2: print everything, 3: debugging
                ):
        self.max_fit_runs = max_fit_runs
        self.run_count = 0
        self.max_restarts_per_parameter = max_restarts
        self.verbosity = verbosity

        self.fit = fit
        self.min_target_from_fit = fit.get_min_target()

        self.ress = [[] for _ in range(len(self.fit.bounds))] # one for each iparam
        self.point_estimate = None
        
        self.conf = conf
        self.conf_tol = conf_tol

        self.iparam = None
        if fit.number_of_fit_parameters() > 1:
            self.profiling = profiling # also sets self.LR_interval and self.LR_target
        else:
            self.vprint(1, "Cannot profile with a single fit parameter; setting profiling = False")
            self.profiling = False

        self._bracket_strategy_input = bracket_strategy # see expand_bracket_strategy()
        self.bracket_step = bracket_step

        self.restart_on_better_point_estimate = True
        
        self.bar = None
        
    ### Internals ###
        
    def vprint(self, verbosity, *args, **kwargs):
        """
        Prints only if ``self.verbosity >= verbosity``.
        """
        if self.verbosity >= verbosity:
            print(f"[bayesmsd.Profiler @ {self.run_count:d}]", (verbosity-1)*'--', *args, **kwargs)

    @property
    def profiling(self):
        return self._profiling
    
    @profiling.setter
    def profiling(self, val):
        # Make sure to keep LR_* in step with the profiling setting
        self._profiling = val

        if self.profiling:
            dof = 1
        else:
            n_params = len(self.fit.bounds)
            n_fixed = len(self.fit.fix_values)
            dof = n_params - n_fixed
            
        self.LR_interval = [stats.chi2(dof).ppf(self.conf-self.conf_tol)/2,
                            stats.chi2(dof).ppf(self.conf+self.conf_tol)/2]
        self.LR_target = np.mean(self.LR_interval)
        
    
    def expand_bracket_strategy(self):
        """
        Preprocessor for the `!bracket_strategy` attribute
        """
        # We need a point estimate to set up the additive scheme, so this can't be in __init__()
        if self.point_estimate is None: # pragma: no cover
            raise RuntimeError("Cannot set up brackets without point estimate")
            
        if self._bracket_strategy_input == 'auto':
            assert self.bracket_step > 1
            
            self.bracket_strategy = []
            for iparam, bounds in enumerate(self.fit.bounds):
                multiplicative = bounds[0] > 0
                if multiplicative:
                    step = self.bracket_step
                    nonidentifiable_cutoffs = [10, 10]
                else:
                    pe = self.point_estimate['params'][iparam]
                    diff_bounds = np.diff(bounds)[0]

                    if np.isfinite(diff_bounds):
                        step = diff_bounds / 20 * self.bracket_step
                        nonidentifiable_cutoffs = [30, 30] # irrelevant, bounds will kick in before

                    elif np.abs(pe) < 1e-10: # point estimate == 0, so no reference scale
                        step = 1
                        nonidentifiable_cutoffs = [10, 10]

                    else:
                        step = np.abs(pe)*(self.bracket_step - 1)
                        nonidentifiable_cutoffs = 2*[2*np.abs(pe)]
                
                self.bracket_strategy.append({
                    'multiplicative'          : multiplicative,
                    'step'                    : step,
                    'nonidentifiable_cutoffs' : nonidentifiable_cutoffs,
                })
        elif isinstance(self._bracket_strategy_input, dict):
            assert self._bracket_strategy_input['step'] > (1 if self._bracket_strategy_input['multiplicative'] else 0)
            self.bracket_strategy = len(self.fit.bounds)*[self._bracket_strategy_input]
        else:
            assert isinstance(self._bracket_strategy_input, list)
            self.bracket_strategy = self._bracket_strategy_input
        
    class FoundBetterPointEstimate(Exception):
        pass
    
    def restart_if_better_pe_found(fun):
        """
        Decorator taking care of restarts

        We use the `Profiler.FoundBetterPointEstimate` exception to handle
        restarts. So a function decorated with this decorator can just raise
        this exception and will then be restarted properly.
        """
        def decorated_fun(self, *args, **kwargs):
            for restarts in range(self.max_restarts_per_parameter):
                try:
                    return fun(self, *args, **kwargs)
                except Profiler.FoundBetterPointEstimate:
                    self.vprint(1, f"Warning: Found a better point estimate ({self.best_estimate['logL']} > {self.point_estimate['logL']})")
                    self.vprint(1, f"Will restart from there ({self.max_restarts_per_parameter-restarts} remaining)")
                    fit_kw = {}

                    # Some housekeeping
                    self.run_count = 0
                    fit_kw['show_progress'] = self.bar is not None

                    # If we're not calculating a profile likelihood, it does not make
                    # sense to keep the old results, since the parameters are different
                    if not self.profiling:
                        fit_kw['init_from'] = self.best_estimate
                        self.ress = [[] for _ in range(len(self.fit.bounds))]
                        self.point_estimate = None

                    # Get a new point estimate, starting from the better one we found
                    # Note that run_fit() starts from best_estimate if
                    # 'init_from' is not specified
                    self.vprint(2, "Finding new point estimate ...")
                    self.run_fit(**fit_kw)

            # If this loop runs out of restarts, we're pretty screwed overall
            raise RuntimeError("Ran out of restarts after finding a better "
                              f"point estimate (max_restarts = {self.max_restarts_per_parameter})") # pragma: no cover

        return decorated_fun
    
    ### Point estimation ###
        
    @staticmethod
    def likelihood_significantly_greater(res1, res2):
        """
        Helper function

        The threshold is 1e-3. Note that this is an asymmetric operation, i.e.
        there is a regime where neither `!res1` significantly greater `!res2`
        nor the other way round.

        Parameters
        ----------
        res1, res2 : dict
            like the output of `Fit.run`, and the entries of ``self.ress``.

        Returns
        -------
        bool
        """
        return res1['logL'] > res2['logL'] + 1e-3 # smaller differences are irrelevant for likelihoods
            
    @property
    def best_estimate(self):
        """
        The best current estimate

        This should usually be the point estimate, but we might find a better
        one along the way.
        """
        if self.point_estimate is None:
            return None
        else:
            best = self.point_estimate
            for ress in self.ress:
                try:
                    candidate = ress[np.argmax([res['logL'] for res in ress])]
                except ValueError: # argmax([])
                    continue
                if self.likelihood_significantly_greater(candidate, best):
                    best = candidate
            return best
    
    def check_point_estimate_against(self, res):
        """
        Check whether `!res` is better than current point estimate

        Parameters
        ----------
        res : dict
            the evaluated parameter point to check. A dict like the ones in
            ``self.ress``.

        Raises
        ------
        Profiler.FoundBetterPointEstimate
        """
        if (
                self.point_estimate is not None
            and self.likelihood_significantly_greater(res, self.point_estimate)
            and self.restart_on_better_point_estimate
            ):
            raise Profiler.FoundBetterPointEstimate
    
    def run_fit(self, is_new_point_estimate=True,
                **fit_kw,
               ):
        """
        Execute one fit run

        This is used to find the initial point estimate, as well as subsequent
        evaluations of the profile posterior.

        Parameters
        ----------
        is_new_point_estimate : bool
            whether we are looking for a new point estimate or just evaluating
            a profile point. This just affects where the result is stored
            internally
        fit_kw : keyword arguments
            additional parameters for `Fit.run`

        See also
        --------
        find_MCI
        """
        self.run_count += 1
        if self.run_count > self.max_fit_runs:
            raise RuntimeError(f"Ran out of likelihood evaluations (max_fit_runs = {self.max_fit_runs})")
            
        if 'init_from' not in fit_kw:
            fit_kw['init_from'] = self.best_estimate
        
        if self.point_estimate is None and fit_kw['init_from'] is None: # very first fit, so do simplex --> (gradient)
            res = self.fit.run(optimization_steps = ('simplex',),
                               **fit_kw,
                              )
            try: # try to refine
                fit_kw['init_from'] = res
                fit_kw['show_progress'] = False
                res = self.fit.run(optimization_steps = ('gradient',),
                                   **fit_kw,
                                  )
            except Exception as err: # okay, this didn't work, whatever # pragma: no cover
                self.vprint(2, "Gradient refinement failed. Point estimate might be imprecise. Not a fatal error, resuming operation")
                self.vprint(3, f"^ was {type(err).__name__}: {str(err)}")

        else: # we're starting from somewhere known, so start out trying to
              # move by gradient, use simplex if that doesn't work
            try:
                res = self.fit.run(optimization_steps = ('gradient',),
                                   verbosity=0,
                                   **fit_kw,
                                  )
            except Exception as err:
                self.vprint(2, "Gradient fit failed, using simplex")
                self.vprint(3, f"^ was {type(err).__name__}: {str(err)}")
                res = self.fit.run(optimization_steps = ('simplex',),
                                   **fit_kw,
                                  )
            # At this point, we used to check that the new result is indeed
            # better than the initial point, which was intended as a sanity
            # check. It ended up being problematic: when we are profiling, we
            # initialize to the closest available previous point, which of
            # course will not fulfill the constraint on the parameter of
            # interest. It's associated likelihood (which we would compare
            # against) thus can be better than any possible value that fulfills
            # the constraint. Long story short: no sanity check here.
        
        if is_new_point_estimate:
            self.point_estimate = res
        else:
            self.ress[self.iparam].append(res)
            self.check_point_estimate_against(res)

        if self.bar is not None:
            self.bar.update() # pragma: no cover
        
    ### Sweeping one parameter ###
    
    def find_closest_res(self, val, direction=None):
        """
        Find the closest previously evaluated point

        Parameters
        ----------
        val : float
            new value
        direction : {None, -1, 1}
            search only for existing values that are greater (1) or smaller
            (-1) than the specified one.

        Returns
        -------
        dict
            appropriate point from ``self.ress``

        See also
        --------
        profile_likelihood
        """
        ress = self.ress[self.iparam] + [self.point_estimate]
        
        values = np.array([res['params'][self.iparam] for res in ress])
        if val in values:
            i = np.argmax([res['logL'] for res in ress if res['params'][self.iparam] == val])
            return ress[i]
        
        if self.bracket_strategy[self.iparam]['multiplicative']:
            distances = np.abs([np.log(val/res['params'][self.iparam]) for res in ress])
        else:
            distances = np.abs([val-res['params'][self.iparam] for res in ress])
            
        if direction is not None:
            distances[np.sign(values - val) != direction] = np.inf
            if not np.any(np.isfinite(distances)):
                raise RuntimeError("Did not find any values in specified direction")
            
        min_dist = np.min(distances)
        
        i_candidates = np.nonzero(distances < min_dist+1e-10)[0] # We use bisection, so usually there will be two candidates
        ii_cand = np.argmax([ress[i]['logL'] for i in i_candidates])
        
        return ress[i_candidates[ii_cand]]
    
    def profile_likelihood(self, value, init_from='closest'):
        """
        Evaluate profile (or conditional) likelihood / posterior

        Parameters
        ----------
        value : float
            value of the current parameter of interest (``self.iparam``)
        init_from : dict or 'closest'
            from where to start the optimization (only relevant if
            ``self.profiling``). Set to 'closest' to use
            ``self.find_closest_res``.

        Returns
        -------
        dict
            like in ``self.ress`` (and also stored there)
        """
        if self.profiling:
            if init_from == 'closest':
                init_from = self.find_closest_res(value)

            self.run_fit(init_from = init_from,
                         fix_values = [(self.iparam, value)],
                         is_new_point_estimate = False,
                        )
        else:
            new_params = self.point_estimate['params'].copy()
            new_params[self.iparam] = value
            minus_logL = self.min_target_from_fit(new_params)
                
            if self.bar is not None:
                self.bar.update() # pragma: no cover
            
            self.ress[self.iparam].append({'logL' : -minus_logL, 'params' : new_params})
            self.check_point_estimate_against(self.ress[self.iparam][-1])
            
        return self.ress[self.iparam][-1]['logL']
    
    def iterate_bracket_point(self, x0, pL, direction,
                              step = None,
                             ):
        """
        First step of the procedure ("bracketing")

        In this first step we simply push out from the point estimate to try
        and establish a bracket containing the exact boundary point. See
        `!Profiler.bracket_strategy`. This function implements that push, for
        one parameter in one direction.

        Parameters
        ----------
        x0, pL : float
            the parameter value and associated posterior value to start from.
            Should be from the point estimate
        direction : {-1, 1}
            which direction to go in
        step : float or None
            use to override ``self.bracket_strategy[self.iparam]['step']``,
            which is used by default.

        Returns
        -------
        x, pL : float
            the found bracket end point and associated posterior. If the chosen
            direction turns out to be unidentifiable, ``x`` is set to the
            corresponding bound (might be ``inf``), and ``pL = np.inf``.

        See also
        --------
        initial_bracket_points
        """
        self.expand_bracket_strategy()
        if step is None:
            step = self.bracket_strategy[self.iparam]['step']
            
        bracket_param = 1 if self.bracket_strategy[self.iparam]['multiplicative'] else 0
        p = x0
        pL_thres = self.point_estimate['logL'] - self.LR_target
        hit_bound = False
        while pL > pL_thres and not hit_bound:
            self.vprint(3, "bracketing: {:.3f} > {:.3f} @ {}".format(pL, pL_thres, p))
            
            # Check identifiability cutoff
            ib = int((1+direction)/2)
            if bracket_param > self.bracket_strategy[self.iparam]['nonidentifiable_cutoffs'][ib]:
                self.vprint(2, "{} edge of confidence interval is non-identifiable".format('left' if direction == -1 else 'right'))
                p = self.fit.bounds[self.iparam][ib]
                pL = np.inf
                break

            # Update position
            if self.bracket_strategy[self.iparam]['multiplicative']:
                bracket_param *= step
                p = x0 * bracket_param**direction
            else:
                bracket_param += step
                p = x0 + direction*bracket_param

            # Enforce bounds
            if direction*(p - self.fit.bounds[self.iparam][ib]) >= 0:
                hit_bound = True
                p = self.fit.bounds[self.iparam][ib]

            # Update profile likelihood
            pL = self.profile_likelihood(p)
        else:
            if pL <= pL_thres:
                self.vprint(3, "bracketing: {:.3f} < {:.3f} @ {}".format(pL, pL_thres, p))
            else:
                assert hit_bound # Sanity check
                self.vprint(3, "bracket reached {} bound: {:.3f} > {:.3f} @ {}".format("lower" if direction < 0 else "upper", pL, pL_thres, p))
                pL = np.inf
            
        return p, pL
        
    def initial_bracket_points(self):
        """
        Execute bracketing in both directions

        See also
        --------
        find_bracket_point, find_MCI
        """
        self.expand_bracket_strategy()
        a, a_pL = self.iterate_bracket_point(self.point_estimate['params'][self.iparam], self.point_estimate['logL'], direction=-1)
        b, b_pL = self.iterate_bracket_point(self.point_estimate['params'][self.iparam], self.point_estimate['logL'], direction= 1)
        return (a, a_pL), (b, b_pL)
    
    def solve_bisection(self, bracket, bracket_pL):
        """
        Find exact root by recursive bisection

        Parameters
        ----------
        bracket : [float, float]
            the bracket of parameter values containing the root
        bracket_pL : [float, float]
            the posterior values associated with the bracket points

        Returns
        -------
        float
            the root within the bracket, to within the given precision (c.f
            ``self.conf_tolerance``)

        Notes
        -----
        All points evaluated along the way are stored in ``self.ress``
        """
        bracket_width = np.diff(bracket)[0]
        if bracket_width < 1e-10:
            # Sometimes the (profile) likelihood is discontinuous right at the
            # (prospective) CI bound, in which case we will not converge to the
            # LR_interval. For the CI this is no problem, it just extends to
            # the location of the jump. To be conservative, we include the jump
            # in the CI.
            self.vprint(3, "bracket collapsed; likelihoods are ({:.3f}, {:.3f})".format(*bracket_pL))
            return bracket[np.argmin(bracket_pL)]

        c = np.mean(bracket)
        c_pL = self.profile_likelihood(c)
        
        self.vprint(3, f"current bracket: {c} +- {bracket_width/2:.5g} @ {bracket_pL[0]:.3f}, {c_pL:.3f}, {bracket_pL[1]:.3f}")

        a_fun, b_fun = self.point_estimate['logL'] - bracket_pL - self.LR_target
        c_fun = self.point_estimate['logL'] - c_pL - self.LR_target
        i_update = 0 if a_fun*c_fun > 0 else 1
        assert [a_fun, b_fun][1-i_update]*c_fun <= 0
        
        bracket[i_update] = c
        bracket_pL[i_update] = c_pL
        
        if np.all([self.LR_interval[0] < LR < self.LR_interval[1] for LR in self.point_estimate['logL'] - bracket_pL]):
            return np.mean(bracket)
        else:
            return self.solve_bisection(bracket, bracket_pL)
    
    @restart_if_better_pe_found
    def find_single_MCI(self, iparam):
        """
        Run the whole profiling process for one parameter

        Parameters
        ----------
        iparam : int
            index of the parameter to sweep

        Returns
        -------
        m : float
            the point estimate value for this parameter
        ci : np.array([float, float])
            the bounds where the posterior has dropped below the maximum value
            the specified amount (c.f. ``self.conf``)

        See also
        --------
        find_MCI
        """
        self.iparam = iparam
        if self.point_estimate is None:
            raise RuntimeError("Need to have a point estimate before calculating confidence intervals") # pragma: no cover
            
        (a, a_pL), (b, b_pL) = self.initial_bracket_points()
        m = self.point_estimate['params'][self.iparam]
        m_pL = self.point_estimate['logL']

        self.vprint(3, "Found initial bracket boundaries, now solving by bisection")
        
        roots = np.array([np.nan, np.nan])
        if a_pL == np.inf:
            roots[0] = a
        if b_pL == np.inf:
            roots[1] = b
        
        for i, (bracket, bracket_pL) in enumerate(zip([(a, m), (m, b)], [(a_pL, m_pL), (m_pL, b_pL)])):
            if np.isnan(roots[i]):
                roots[i] = self.solve_bisection(np.asarray(bracket), np.asarray(bracket_pL))
            if i == 0:
                self.vprint(2, f"Found left edge @ {roots[i]}, now searching right one")
        
        self.vprint(2, f"found CI = {roots} (point estimate = {m}, iparam = {self.iparam})\n")
        return m, roots
    
    def find_MCI(self, iparam='all',
                 show_progress=False,
                ):
        """
        Perform the sweep for multiple parameters

        Parameters
        ----------
        iparam : 'all' or np.ndarray, dtype=int
            the parameters to sweep. If 'all', will sweep all independent
            parameters, i.e. those not fixed via the `Fit.fix_values`
            mechanism.

        Returns
        -------
        mcis : (n, 3) np.ndarray, dtype=float
            for each parameter, in order: point estimate, lower, and upper
            bound.

        See also
        --------
        run_fit
        """
        if show_progress and self.bar is None:
            self.bar = tqdm() # pragma: no cover
            
        if self.point_estimate is None:
            self.vprint(2, "Finding initial point estimate ...")
            self.run_fit(show_progress=show_progress)
        
        self.vprint(2, "initial point estimate: params = {}, logL = {}\n".format(self.point_estimate['params'],
                                                                                 self.point_estimate['logL'],
                                                                                ))
        
        n_params = len(self.point_estimate['params'])
        mcis = np.empty((n_params, 3), dtype=float)
        mcis[:] = np.nan
        
        # check, which parameters to run
        if iparam == 'all':
            fixed = [i for i, _ in self.fit.fix_values]
            iparams = np.array([i for i in range(n_params) if i not in fixed])
        else:
            iparams = np.asarray(iparam)
            if len(iparams.shape) == 0:
                iparams = np.array([iparams])
               
        # keep track of which point estimate was used for which parameters
        used_point_estimates = n_params*[None]
        while True: # limited by max_restarts_per_parameter
            for iparam in iparams:
                # Only run if we did not already sweep from here
                if used_point_estimates[iparam] is not self.point_estimate:
                    self.run_count = 0
                    self.vprint(2, f"starting iparam = {iparam}")
                    m, ci = self.find_single_MCI(iparam)
                    mcis[iparam, :] = m, *ci
                    used_point_estimates[iparam] = self.point_estimate
                
            if ( all([used_point_estimates[iparam] is self.point_estimate for iparam in iparams])
                 or not self.restart_on_better_point_estimate ):
                break
        
        if show_progress and self.bar is not None: # pragma: no cover
            self.bar.close()
            self.bar = None
            
        self.vprint(2, "Done\n")
        
        if len(iparams) == 1:
            return mcis[iparams[0]]
        else:
            return mcis
