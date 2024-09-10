DOCDIR = doc
EXAMPLEDIR = examples
SPHINXDIR = $(DOCDIR)/sphinx
SPHINXSOURCE = $(SPHINXDIR)/source
SPHINXBUILD = $(SPHINXDIR)/source/_build
TESTDIR = tests
TESTFILE = test_bayesmsd.py
COVERAGEREPFLAGS = --omit=*/noctiluca/*,*/rouse/*
COVERAGEREPDIR = $(TESTDIR)/coverage
DISTDIR = dist
MODULE = bayesmsd
CYTHONSRCDIR = bayesmsd/src
CYTHONBINDIR = bayesmsd/bin
CYTHONYELLOWDIR = doc/cython_yellow
TESTINSTALLDIR = $(TESTDIR)/installation
TESTINSTALLVENV = testinstall
TESTINSTALLPYTHON = PYTHONPATH= ./$(TESTINSTALLDIR)/$(TESTINSTALLVENV)/bin/python3

.PHONY : setup recompile yellow build pre-docs docs tests testinstall all clean mydocs mytests myall myclean

setup :
	mkdir -p $(COVERAGEREPDIR) $(DISTDIR) $(CYTHONBINDIR) $(CYTHONYELLOWDIR)
	nbstripout --install
	vim "+g/-m nbstripout/norm A --drop-empty-cells" +wq .git/config

all : docs tests build

bayesmsd/src/*.c : bayesmsd/src/*.pyx
	CYTHONIZE=1 python3 setup.py build_ext --inplace

recompile : bayesmsd/src/*.c

yellow : bayesmsd/src/*.pyx
	cython -3 -a $(CYTHONSRCDIR)/*.pyx
	@mv $(CYTHONSRCDIR)/*.html $(CYTHONYELLOWDIR)

build : recompile
	-@cd $(DISTDIR) && rm *
	python3 -m build # source dist & linux wheel
	@cd $(DISTDIR) && auditwheel repair *-linux_*.whl
	@cd $(DISTDIR) && mv wheelhouse/* . && rmdir wheelhouse
	@cd $(DISTDIR) && rm *-linux_*.whl
	PYTHON_ONLY=1 python3 -m build --wheel # py3-none-any wheel

pre-docs :
	sphinx-apidoc -f -o $(SPHINXSOURCE) $(MODULE)
	@rm $(SPHINXSOURCE)/modules.rst
	@cd $(SPHINXSOURCE) && vim -nS post-apidoc.vim
	cd $(SPHINXDIR) && $(MAKE) clean
	-@rm -rf $(SPHINXSOURCE)/$(EXAMPLEDIR)
	@cp -rf $(EXAMPLEDIR) $(SPHINXSOURCE)/
	@cd $(SPHINXSOURCE) && vim -nS write_examples_rst.vim

docs : pre-docs
	cd $(SPHINXDIR) && $(MAKE) html

docs-latex : pre-docs
	cd $(SPHINXDIR) && $(MAKE) latex

tests : recompile
	cd $(TESTDIR) && coverage run $(TESTFILE)
	@mv $(TESTDIR)/.coverage .
	coverage html -d $(COVERAGEREPDIR) $(COVERAGEREPFLAGS)
	coverage report --skip-covered $(COVERAGEREPFLAGS)

testinstall :
	mkdir -p $(TESTINSTALLDIR)
	rm -r $(TESTINSTALLDIR)/*
	cd $(TESTINSTALLDIR) && python3 -m venv --clear $(TESTINSTALLVENV)
	$(TESTINSTALLPYTHON) -m pip install --upgrade pip setuptools build
	$(TESTINSTALLPYTHON) -m build -o $(TESTINSTALLDIR)/dist
	$(TESTINSTALLPYTHON) -m pip install $(TESTINSTALLDIR)/dist/$(MODULE)-*.whl
	$(TESTINSTALLPYTHON) -c "import bayesmsd"

clean :
	-rm -r $(SPHINXBUILD)/*
	-rm -r $(COVERAGEREPDIR)/*
	-rm .coverage

# Personal convenience targets
# Edit DUMPPATH to point to a directory that you can easily access
# For example, when working remotely, sync output via Dropbox to inspect locally
DUMPPATH = "/home/simongh/Dropbox (MIT)/htmldump"
mydocs : docs
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mydocs-latex : docs-latex
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mytests : tests
	cp -r $(COVERAGEREPDIR)/* $(DUMPPATH)/coverage

myall : mydocs mytests

myclean : clean
	-rm -r $(DUMPPATH)/sphinx/*
	-rm -r $(DUMPPATH)/coverage/*
