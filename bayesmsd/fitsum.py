from copy import deepcopy

import numpy as np

from noctiluca import parallel

from . import deco
from .fit import Fit
from .parameters import Parameter, Linearize
from .lib import _MAX_LOG

class FitSum(Fit):
    """
    Fitting sums of elementary MSDs

    The main use case of this construct is to fit sums of powerlaws, but maybe
    it can be useful for other settings as well.

    Parameters
    ----------
    fits_dict : dict of Fit
        the fits to sum; we refer to the dict keys as "fit names"

    Attributes
    ----------
    parameters : dict of Parameter
        collection of all the individual fit parameters, with their names
        prefixed by the fit name (so ``'α (dim 0)'`` of the fit ``'fit1'``
        would become ``'fit1 α (dim 0)'``)
    constraints : list of callables
        same as `!Fit.constraints`
    verbosity : int
        same as `!Fit.verbosity`
    improper_priors : list of str
        aggregated list of `!Fit.improper_priors`, with names adjusted to the
        joint fit (i.e. prefixed with the fit name)
    likelihood_chunksize : int
        same as `!Fit.likelihood_chunksize`

    Notes
    -----
    Localization error can be added at the `!FitSum` level and it usually makes
    sense to stick with this. So fix localization error for elementary fits to
    0. Motion blur should be set for each fit individually (since it's linear
    in the MSD we can let the elementary Fits figure it out for themselves)

    `!Fit.data` should be identical (``is``) for each elementary Fit. This is
    checked upon initialization.
    """
    def __init__(self, fits_dict):
        self.fits_dict = fits_dict
        refkey = next(iter(self.fits_dict))

        data = self.fits_dict[refkey].data
        if not all([fit.data is data for fit in self.fits_dict.values()]):
            raise ValueError("Fits have different data sets; aborting")

        ss_order = self.fits_dict[refkey].ss_order
        if not all([fit.ss_order == ss_order for fit in self.fits_dict.values()]):
            raise ValueError("Fits have different ss_order; aborting")

        for fitname, fit in self.fits_dict.items():
            m1_keys = {key for key in fit.parameters if key.startswith('m1 (dim')}
            for key in m1_keys:
                if fit.parameters[key].fix_to != 0:
                    raise ValueError(f"Fit '{fitname}' has {key} != 0. Use FitSum-level m1 for trends.")

        super().__init__(data)
        self.ss_order = ss_order
        self.constraints = []

        # Add localization error
        for dim in range(self.d):
            self.parameters[f'log(σ²) (dim {dim})'] = Parameter((-np.inf, _MAX_LOG),
                                                                  linearization=Linearize.Exponential(),
                                                                  )

        # grab parameters from elementary fits
        for fitname, fit in self.fits_dict.items():
            for paramname in fit.independent_parameters():
                joint_param_name = self.make_joint_param_name(fitname, paramname)
                self.parameters[joint_param_name] = deepcopy(fit.parameters[paramname])

        # same for improper priors
        self.improper_priors = [self.make_joint_param_name(fitname, paramname)
                                for fitname, fit in self.fits_dict.items()
                                for paramname in fit.improper_priors
                               ]

    ### Copied from FitGroup ###

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

    ### `Fit` functionality ###

    def params2msdm(self, params):
        elementary_msd = [[] for dim in range(self.d)]
        for fitname, fit in self.fits_dict.items():
            myparams = {self.make_fit_param_name(fitname, name) : val
                        for name, val in params.items()
                        if name.startswith(self.joint_param_prefix(fitname))
                        }
            myparams = fit.fill_dependent_params(myparams)
            elem = fit.params2msdm(myparams)
            for dim in range(self.d):
                elementary_msd[dim].append(elem[dim][0]) # m1 == 0 is ensured on init

        msdm = []
        for dim in range(self.d):
            with np.errstate(under='ignore'):
                noise2 = np.exp(params[f"log(σ²) (dim {dim})"])

            @deco.MSDfun
            @deco.imaging(noise2=noise2, f=0) # motion blur should be handled by elementary fits
            def msd(dt):
                return np.sum([elem(dt) for elem in elementary_msd[dim]], axis=0)

            msdm.append((msd, params[f'm1 (dim {dim})']))
        return msdm

    ### Aggregate stuff from elementary fits ###
    
    def initial_params(self):
        """
        Aggregate initial parameters from elementary fits

        Returns
        -------
        dict
            initial parameters from elementary fits, with properly prefixed
            names
        """
        # This might not be the best thing we can do here, but hopefully fine-ish
        group_params = {}
        log_s2 = -np.inf
        for fitname, fit in self.fits_dict.items():
            fit_params = fit.initial_params()
            for name, val in fit_params.items():
                group_params[self.make_joint_param_name(fitname, name)] = val
                if 'log(σ²)' in name:
                    log_s2 = max(log_s2, val)

        for dim in range(self.d):
            group_params[f'm1 (dim {dim})'] = 0
            group_params[f'log(σ²) (dim {dim})'] = log_s2

        return group_params

    def logprior(self, params):
        """
        Aggregate log-priors from elementary fits

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
