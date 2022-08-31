[![Documentation Status](https://readthedocs.org/projects/bayesmsd/badge/?version=latest)](https://bayesmsd.readthedocs.io/en/latest/?badge=latest)

BayesMSD: properly fitting MSDs
===============================

While inspection of MSD curves is one of the most ubiquitous ways of analyzing
particle tracking data, it is also well known that extracting model parameters
from MSD curves is a statistical minefield[^1]. This problem can be addressed
quite nicely in the language of Gaussian processes, allowing statistically
rigorous MSD fits. This provides, for example, error bars on estimated model
parameters, which are quite noticeably missing in current literature.

Documentation is hosted at [ReadTheDocs](https://bayesmsd.readthedocs.org/en/latest)

[^1]: Vestergaard, Blainey, Flyvbjerg, __Optimal estimation of diffusion coefficients from single-particle trajectories__, _Physical Review E_, 2014; [DOI](https://doi.org/10.1103/PhysRevE.89.022726)
