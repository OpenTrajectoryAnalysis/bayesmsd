"""
Bayesian MSD fitting

Any *valid* MSD function MSD(Δt) defines a stationary Gaussian process with the
appropriate correlation structure. Therefore, instead of fitting the graph of
"empirically" calculated MSDs, we can perform Bayesian inference of parametric
MSD curves, using the likelihood function defined through these Gaussian
processes.

We discriminate two cases, depending on what exactly is stationary:

 + the trajectories themselves might be sampled from a stationary process (e.g.
   distance of two points on a polymer). In terms of the MSD, the decisive
   criterion is ``MSD(inf) < inf``. In this case, it is straightforward to
   prove the following relation between the MSD μ and autocovariance γ of the
   process:

    .. code-block:: text

        μ(k) = 2*( γ(0) - γ(k) )

   Thus, the full autocovariance function can be obtained from the MSD and the
   steady state covariance ``γ(0)``. For decaying correlations (``γ(k) --> 0 as
   k --> ∞``) we furthermore see that ``2*γ(0) = μ(∞)`` is the asymptotic value
   of the MSD. Finally, this allows us to calculate the covariance matrix of
   the process as

    .. code-block:: text

        C_ij := <x_i*x_j>
              = γ(|i-j|)
              = γ(0) - 1/2*μ(|i-j|)

   We call this case a steady state of order 0.
 + a special case of order 0 is the scenario where we have a stationary
   process, but when evaluating e.g. likelihoods we wish to condition on the
   first data point of the trajectory. This renders the resulting likelihoods
   comparable to those of an order 1 stationary process and can thus be useful
   sometimes; we refer to this scenario as ``ss_order = 0.5``.
 + in many cases (e.g. sampling a diffusing particle's position) the
   trajectories themselves will not be stationary, but the increment process
   is. In this case the Gaussian process of interest is the one generating the
   increments of the trajectory, whose autocorrelation is the second derivative
   of the MSD:

    .. code-block:: text

        γ(k) = 1/2 * (d/dk)^2 μ(k)

   where derivatives should be understood in a weak ("distributional") sense.
   More straightforwardly, the correlation matrix of the increments is given by

    .. code-block:: text

       C_ij := <(x_{i+1} - x_i)(x_{j+1}-x_j)>
             = 1/2 * ( μ(t_{i+1} - t_j) + μ(t_i - t_{j+1})
                      -μ(t_{i+1} - t_{j+1}) - μ(t_i - t_j) )

   where by definition we let ``μ(-k) = μ(k)``. In this case, we talk about a
   steady state of order 1.

In any case, the covariance matrix ``C`` (potentially together with a mean /
drift term for steady state order 0 / 1 respectively) defines a Gaussian
process, which lets us assign a likelihood to the generating MSD. Via this
construction, we can perform rigorous Bayesian analysis, cast in the familiar
language of MSDs.

This package provides a base class for performing such inferences / fits,
namely `Fit`. We also provide a few example implementations of specific fitting
schemes in the `lib` submodule. Finally, the `Profiler` allows to explore the
posterior once a point estimate has been found, by tracing out either
conditional posterior or profile posterior curves in each parameter direction.
Note that you can also just sample the posterior by MCMC.

See also
--------
Fit, Profiler
"""
from . import deco
from . import gp
from . import parameters
from .fit        import Fit
from .fitgroup   import FitGroup
from .fitsum     import FitSum
from .profiler   import Profiler
from . import lib
