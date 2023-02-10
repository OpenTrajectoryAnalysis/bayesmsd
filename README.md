[![Documentation Status](https://readthedocs.org/projects/bayesmsd/badge/?version=latest)](https://bayesmsd.readthedocs.io/en/latest/?badge=latest)

BayesMSD: properly fitting MSDs
===============================

While inspection of MSD curves is one of the most ubiquitous ways of analyzing
particle tracking data, it is also well known that extracting model parameters
from MSD curves is a statistical minefield[^1]. This problem can be addressed
quite nicely in the language of Gaussian processes, allowing statistically
rigorous MSD fits. This provides, for example, error bars on estimated model
parameters, which are quite noticeably missing from the current literature.

For a [Quickstart
intro](https://bayesmsd.readthedocs.io/en/latest/examples/00_intro.html), more
extensive [Tutorials &
Examples](https://bayesmsd.readthedocs.io/en/latest/examples.html) and the full
[API reference](https://bayesmsd.readthedocs.io/en/latest/bayesmsd.html) refer
to the [documentation hosted at
ReadTheDocs](https://bayesmsd.readthedocs.org/en/latest).

To install `bayesmsd` you can use the latest stable version from [PyPI](https://pypi.org/project/bayesmsd)
```sh
$ pip install --upgrade bayesmsd
```
or the very latest updates right from GitHub:
```sh
$ pip install git+https://github.com/OpenTrajectoryAnalysis/bayesmsd
```

When cloning the repo and installing in editable mode, make sure to use `make
setup` to setup the parts of the local environment that are not tracked in git
(see [Developers](#developers)):
```sh
$ git clone https://github.com/OpenTrajectoryAnalysis/bayesmsd
$ cd bayesmsd && make setup
$ pip install -e .
```

[^1]: Vestergaard, Blainey, Flyvbjerg, __Optimal estimation of diffusion coefficients from single-particle trajectories__, _Physical Review E_, 2014; [DOI](https://doi.org/10.1103/PhysRevE.89.022726)

Developers
----------
We use GNU `make` to automate recurrent tasks. Targets include:
 - `make setup` : set up the local environment after cloning. Requires
   [nbstripout](https://github.com/kynan/nbstripout) which is used to [remove
   output and empty cells from the example
   notebooks](https://nbsphinx.readthedocs.io/en/sizzle-theme/usage.html#Using-Notebooks-with-Git).
 - `make recompile` : (re-)compile cython code
 - `make build` : build wheels for distribution on PyPI
 - `make tests` : run unittests
 - `make docs` : build Sphinx documentation
