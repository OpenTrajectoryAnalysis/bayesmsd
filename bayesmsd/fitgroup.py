from copy import deepcopy

import numpy as np

from noctiluca import parallel

from .fit import Fit

class Patch: # used for masking Fit.data and others
    def __call__(self, *args, **kwargs):
        return None

class FitGroup(Fit):
    """
    Running interdependent fits

    The `!FitGroup` allows running fits to different data sets simultaneously,
    while fixing parameters across these fits. A common use case is a scenario
    where we acquired SPT data of the same biological system with different
    technical settings: choice of microscope, choice of lag time, localization
    error & motion blur, etc. In this case we would like to run a fit that fits the
    same "true" MSD to the data, while maintaining independence for e.g. the
    localization errors associated with each data set.

    Usage is (hopefully) relatively intuitive:

    - define a few "simple" fits and collect them in a dict
    - give that dict to `!FitGroup`
    - use the usual ``.fix_to = ...`` mechanism to fix parameters in the fit
      group.  All parameters from the individual fits can be used, with their
      names prefixed by the fit name (the key for that fit in the dict)
    - `!FitGroup.run()` as you would `!Fit.run()`

    Parameters
    ----------
    fits_dict : dict of Fit
        the fits to run; we refer to the dict keys as "fit names"

    Attributes
    ----------
    parameters : dict of Parameter
        collection of all the individual fit parameters, with their names
        prefixed by the fit name (so ``'α (dim 0)'`` of the fit ``'fit1'``
        would become ``'fit1 α (dim 0)'``)
    constraints : list of callables
        same as `!Fit.constraints`. Note that you can use constraints within
        the individual fits, as well as on the group level. It usually makes
        sense to stay "as local as possible", though this is not a requirement.
    max_penalty : float
        same as `!Fit.max_penalty`
    verbosity : int
        same as `!Fit.verbosity`
    improper_priors : list of str
        aggregated list of `!Fit.improper_priors`, with names adjusted to the
        joint fit (i.e. prefixed with the fit name)
    likelihood_chunksize : int
        controls chunking of parallelization (if running in
        `!noctiluca.Parallelize` context): ``< 0`` prevents any
        parallelization; ``0`` submits the whole likelihood calculation into
        one process; ``> 0`` chunks the likelihood calculation by fits; i.e. a
        chunk size of 1 corresponds to each fit in `!fit_dict` running in its
        own process.
    """
    # TODO: this probably won't work with spline fits for now (because they
    # rely on fixed max trajectory length for time compaction). This should be
    # easy to fix by allowing the "max time to consider" as init argument;
    # also, adjusting the `logT` attribute for trajectories' Δt.
    def __init__(self, fits_dict):
        self.fits_dict = fits_dict

        # mask self.data.restoreSelection() and self.data_selection (which are used in Fit.run())
        self.data = Patch()
        self.data.restoreSelection = Patch()
        self.data_selection = Patch()

        # grab parameters from individual fits
        self.parameters = {}
        for fitname, fit in self.fits_dict.items():
            for paramname in fit.independent_parameters():
                joint_param_name = self.make_joint_param_name(fitname, paramname)
                self.parameters[joint_param_name] = deepcopy(fit.parameters[paramname])
        
        # other stuff to replicate `Fit` functionality
        self.constraints = [] # can add constraints at the group level
        self.max_penalty = min(fit.max_penalty for fit in self.fits_dict.values())

        self.verbosity = 1
        self.likelihood_chunksize = -1
        self.maxfev = 1e10

        self.improper_priors = [self.make_joint_param_name(fitname, paramname)
                                for fitname, fit in self.fits_dict.items()
                                for paramname in fit.improper_priors
                               ]

    ### FitGroup specific methods ###

    def joint_param_prefix(self, fitname):
        return fitname+' '

    def make_joint_param_name(self, fitname, paramname):
        return self.joint_param_prefix(fitname)+paramname

    def make_fit_param_name(self, fitname, paramname):
        prefix = self.joint_param_prefix(fitname)
        if paramname.startswith(prefix):
            return paramname[len(prefix):]
        else: # pragma: no cover
            raise RuntimeError(f"Parameter '{paramname}' does not seem to belong to Fit '{fitname}'")

    ### Kill some `Fit` functionality (no well-defined MSD here) ###

    def params2msdm(self, params): # pragma: no cover
        raise NotImplementedError
        
    def MSD(self, params, dt): # pragma: no cover
        raise NotImplementedError

    ### Overwrite others that can be aggregated from individual fits ###
    
    def initial_params(self):
        """
        Aggregate initial parameters from individual fits

        Returns
        -------
        dict
            initial parameters from individual fits, with properly prefixed
            names
        """
        group_params = {}
        for fitname, fit in self.fits_dict.items():
            fit_params = fit.initial_params()
            for name, val in fit_params.items():
                group_params[self.make_joint_param_name(fitname, name)] = val

        return group_params

    def logprior(self, params):
        """
        Aggregate log-priors from individual fits

        Parameters
        ----------
        params : dict
            parameters for which to evaluate the prior

        Returns
        -------
        float
            aggregated log-prior

        Notes
        -----
        Which parameters are handed in with `!params` determines what is
        considered a free parameter; so ensure to not hand in parameters that
        factually are fixed.

        The ability of a `!FitGroup` to calculate its prior is not really
        necessary and not used in the code; this function is thus implemented
        mostly for completeness/rigor (it is not necessary for
        `!Fit.evidence()` to work; what we need there is just that the
        individual fits take care of their respective priors respectively).
        """
        pi = 0
        for fitname, fit in self.fits_dict.items():
            fit_params = {}
            for paramname, val in params.items():
                try:
                    fit_params[self.make_fit_param_name(fitname, paramname)] = val
                except RuntimeError:
                    continue

            pi += fit.logprior(fit_params)
        return pi

    ### Things to keep as is in `Fit` ###

    # - vprint()
    # - _penalty()
    # - initial_offset(): return 0
    # - expand_fix_values()
    # - independent_parameters()

    ### Key functionality of FitGroup: override MinTarget ###

    # Parameter fixing and priors
    # ===========================
    # The priors in the individual fits should be adjusted properly for fixed
    # parameters; otherwise we might "overcount" the prior for some parameter
    # (i.e. add it multiple times). Consider the following "levels" at which we
    # can fix parameters:
    # 
    # - within each fit, use Parameter.fix_to = ...
    # - within the fit group, use Parameter.fix_to = ...
    # - use the fix_values argument to MinTarget.__init__()
    # 
    # The first two options are "structural" fixes and should always be taken
    # into account (i.e. the fixed parameters should not have a prior
    # calculated). The third option might be used e.g. by the Profiler and
    # should thus listen to 'adjust_prior_for_fixed_values': if that flag is
    # unset, when calculating the prior we should pretend that these parameters
    # are free (such that profile likelihoods are comparable to each other).
    # 
    # So: when assembling the fit-level MinTargets, we take care to account
    # properly for all of this:
    # 
    # - fit-level .fix_to is properly dealt with by
    #   fit.MinTarget(adjust_prior=True)
    # - group-level .fix_to we remove by hand
    # - group-level kwargs fixes we add back in by hand, iff adjust_prior ==
    #   False
            
    class MinTarget(Fit.MinTarget):
        """
        Cost function for the `!FitGroup`; overrides `!Fit.MinTarget`

        Mainly take care of proper parameter and prior management; actual
        likelihood evaluations are delegated to the individual fits.

        Parameters are same as for `Fit.MinTarget`.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # gives:
            # - self.fit
            # - self.likelihood_chunksize
            # - self.params_marginalized
            # - self.params_free
            # - self.params_to_constant
            # - self.params_to_other
            # - self.params_to_callable
            self.paramnames_prior = [] # priors are taken care of by the component fits (see below)

            try:
                adjust_prior_for_fixed_values = kwargs['adjust_prior_for_fixed_values']
            except KeyError:
                adjust_prior_for_fixed_values = True

            try:
                kwarg_fix_values = kwargs['fix_values']
            except KeyError:
                kwarg_fix_values = None
            if kwarg_fix_values is None:
                kwarg_fix_values = {}
            kwarg_fixed_params = [name for name, val in kwarg_fix_values.items() if val is not None]

            # Assemble MinTarget for each fit
            self.mintargets = {}
            for fitname, fit in self.fit.fits_dict.items():
                target = fit.MinTarget(fit)

                prior_names = target.paramnames_prior
                prior_names_joint = [self.fit.make_joint_param_name(fitname, paramname) for paramname in prior_names]
                calculate_prior_for = [name for name, joint_name in zip(prior_names, prior_names_joint)
                                       if (joint_name in self.params_free
                                           or (joint_name in kwarg_fixed_params
                                               and adjust_prior_for_fixed_values == False
                                           ))]
                target.paramnames_prior = calculate_prior_for
                
                self.mintargets[fitname] = target

        @staticmethod
        def _eval_target(target_and_params):
            # for parallelization
            target, params = target_and_params
            return target(params)

        def eval_atomic(self, params_array):
            params_dict = self.params_array2dict(params_array)

            penalty = self.fit._penalty(params_dict)
            if penalty < 0: # pragma: no cover
                return self.fit.max_penalty
            else:
                todo = []
                for fitname, target in self.mintargets.items():
                    params_dict_fit = {}
                    for paramname, paramval in params_dict.items():
                        try:
                            paramname = self.fit.make_fit_param_name(fitname, paramname)
                        except RuntimeError:
                            continue

                        params_dict_fit[paramname] = paramval

                    todo.append((target, target.params_dict2array(params_dict_fit)))

                imap = parallel._map(self._eval_target, todo,
                                     chunksize=self.likelihood_chunksize,
                                     )
                target_values = np.array([penalty] + list(imap))

                total = np.sum(target_values)
                if np.any(np.append(target_values, total) > self.fit.max_penalty): # pragma: no cover
                    return self.fit.max_penalty # prevent "max penalty hopping"
                else:
                    return total - self.offset
