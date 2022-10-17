[![Documentation Status](https://readthedocs.org/projects/bayesmsd/badge/?version=latest)](https://bayesmsd.readthedocs.io/en/latest/?badge=latest)

BayesMSD: properly fitting MSDs
===============================

While inspection of MSD curves is one of the most ubiquitous ways of analyzing
particle tracking data, it is also well known that extracting model parameters
from MSD curves is a statistical minefield[^1]. This problem can be addressed
quite nicely in the language of Gaussian processes, allowing statistically
rigorous MSD fits. This provides, for example, error bars on estimated model
parameters, which are quite noticeably missing from the current literature.

For a [Quickstart intro](https://bayesmsd.readthedocs.io/en/latest/examples/00_intro.html), more extensive [Tutorials & Examples](https://bayesmsd.readthedocs.io/en/latest/examples.html) and the full [API reference](https://bayesmsd.readthedocs.io/en/latest/bayesmsd.html) refer to the [documentation hosted at ReadTheDocs](https://bayesmsd.readthedocs.org/en/latest).

To install `bayesmsd` you can use the latest stable version from [PyPI](https://pypi.org/project/bayesmsd)
```sh
$ pip install --upgrade bayesmsd
```
or the very latest updates right from GitHub:
```sh
$ pip install git+https://github.com/OpenTrajectoryAnalysis/bayesmsd
```

[^1]: Vestergaard, Blainey, Flyvbjerg, __Optimal estimation of diffusion coefficients from single-particle trajectories__, _Physical Review E_, 2014; [DOI](https://doi.org/10.1103/PhysRevE.89.022726)

Developers
----------
Note the `Makefile`, which can be used to build the documentation (using
Sphinx); run unit tests and check code coverage; and build an updated package
for release with GNU `make`.

When editing the example notebooks,
[remember](https://nbsphinx.readthedocs.io/en/sizzle-theme/usage.html#Using-Notebooks-with-Git)
to remove output and empty cells before committing to the git repo.
[nbstripout](https://github.com/kynan/nbstripout) allows to do this
automatically upon commit.
