import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import numpy as np
np.seterr(all='raise') # pay attention to details

import unittest
from unittest.mock import patch

import rouse
import noctiluca as nl
from context import bayesmsd

from multiprocessing import Pool

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kddO]kV?__all__j>>
"""
__all__ = [
    'TestDiffusive',
    'TestRouseLoci',
    'TestProfiler',
    'TestRandomStuff',
    'TestNewImplementation',
]

# We test mostly the library implementations, since the base class `Fit` is
# abstract
# 
# We also test on so few data that all the results are useless and we don't
# attempt to check for correctness of the fit. This code is supposed to be
# technical tests and not benchmarks, so it should run fast (-ish).
# 
# These tests are organized not by fit class, but by synthetic motion type

# Extend unittest.TestCase's capabilities to deal with numpy arrays
class myTestCase(unittest.TestCase):
    def assert_array_equal(self, array1, array2):
        try:
            np.testing.assert_array_equal(array1, array2)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

    def assert_array_almost_equal(self, array1, array2, decimal=10):
        try:
            np.testing.assert_array_almost_equal(array1, array2, decimal=decimal)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

    def assert_mci_almost_equal(self, mci1, mci2, places=2):
        self.assertEqual(mci1.keys(), mci2.keys())
        for key in mci1:
            m1, ci1 = mci1[key]
            m2, ci2 = mci2[key]

            self.assertAlmostEqual(m1, m2, places=places)
            self.assert_array_almost_equal(ci1, ci2, decimal=places)

class TestDiffusive(myTestCase):
    def setUp(self):
        def traj():
            return nl.Trajectory(np.cumsum(np.random.normal(size=(10, 3)), axis=0))

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    def testSpline(self):
        fit = bayesmsd.lib.SplineFit(self.data, ss_order=1, n=4)
        res = fit.run(verbosity=0, maxfev=500)

        for name in ["x1", "x2"]:
            self.assertGreater(res['params'][name], 0)
            self.assertLess(res['params'][name], 1)

        # provoke penalization
        res2 = fit.run(init_from={'params' : {"x1" : 1e-8,
                                              "x2" : 0.5,
                                              "y0" : 0, "y1" : 0,
                                              "y2" : 0, "y3" : 0,
                                              }})
        # provoke "infinite" penalization
        res2 = fit.run(init_from={'params' : {"x1" : 1e-20,
                                              "x2" : 0.5,
                                              "y0" : 0, "y1" : 0,
                                              "y2" : 0, "y3" : 0,
                                              }})

        # check compactify / decompactify cycle
        dt = np.array([1, 5, 23, 100, 579, np.inf])
        self.assert_array_almost_equal(np.log(dt), fit.decompactify_log(fit.compactify(dt)))
        fit = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=3)
        self.assert_array_almost_equal(np.log(dt), fit.decompactify_log(fit.compactify(dt)))

    def testNPX(self):
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=0)
        res = fit.run(verbosity=0)

        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=1)
        res = fit.run()
        
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=1, motion_blur_f=0.5)
        res = fit.run()

class TestRouseLoci(myTestCase):
    def setUp(self):
        model = rouse.Model(10)
        tracked = [2, 7]
        def traj():
            conf = model.conf_ss()
            traj = []
            for _ in range(10):
                conf = model.evolve(conf)
                traj.append(conf[tracked[0] - tracked[1]])

            return nl.Trajectory(traj)

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    @patch('builtins.print')
    def testSpline(self, mock_print):
        fit = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=4)
        res = fit.run()

        for name in ["x1", "x2"]:
            self.assertGreater(res['params'][name], 0)
            self.assertLess(res['params'][name], 2)

        self.assertSetEqual(set(fit.independent_fit_parameters()),
                            {'x1', 'x2', 'y0', 'y1', 'y2', 'y3'})

        # Test refining a spline fit
        fit2 = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=6,
                                    previous_spline_fit_and_result = (fit, res),
                                    )

        self.assertSetEqual(set(fit2.independent_fit_parameters()),
                            {      'x1', 'x2', 'x3', 'x4',
                             'y0', 'y1', 'y2', 'y3', 'y4', 'y5',
                            })

        with self.assertRaises(RuntimeError):
            res2 = fit2.run(optimization_steps=('gradient',), maxfev=10)

    def testRouse(self):
        fit = bayesmsd.lib.TwoLocusRouseFit(self.data, motion_blur_f=1)

        # no localization error
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 

        res = fit.run(full_output=True, optimization_steps=(dict(method='Nelder-Mead', options={'fatol' : 0.1, 'xatol' : 0.01}),))[-1][0]

        self.assertEqual(res['params']['log(σ²) (dim 0)'], -np.inf)
        self.assertSetEqual(set(fit.independent_fit_parameters()),
                            {'log(Γ) (dim 0)', 'log(J) (dim 0)'})

    @patch('builtins.print')
    def testNPX(self, mock_print):
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=1)
        res = fit.run()

        new_fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=2,
                                    previous_NPXFit_and_result = (fit, res),
                                    )
        new_res = new_fit.run()
        self.assertGreater(new_res['logL'], res['logL'])

        new2_fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=0,
                                     previous_NPXFit_and_result = (new_fit, new_res),
                                     )
        try:
            new2_res = new2_fit.run()
        except RuntimeError as err: # pragma: no cover
            pass # if fit does not converge, which might happen.
                 # ideally we would figure out how to make sure it converges
                 # or at least check that the right error message has been printed
        # the below comparison is meaningless, because we compare different ss_order likelihoods
        # self.assertLess(new2_res['logL'], new_res['logL'])

        new3_fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=1,
                                     previous_NPXFit_and_result = (new2_fit, new2_res),
                                     )
        try:
            new3_res = new3_fit.run()
            self.assertLess(new3_res['logL'], new_res['logL'])
            self.assertLess(new3_res['logL'], res['logL'])
        except RuntimeError: # pragma: no cover
            pass

        with self.assertRaises(ValueError):
            fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=0)

        with self.assertRaises(ValueError):
            data = self.data.apply(lambda traj: nl.Trajectory(traj[:][:, 0]))
            fit = bayesmsd.lib.NPXFit(data, ss_order=0, n=5,
                                    previous_NPXFit_and_result = (new2_fit, new2_res),
                                    )

class TestProfiler(myTestCase):
    # set up diffusive data set
    def setUp(self):
        def traj():
            return nl.Trajectory(np.cumsum(np.random.normal(size=(10, 3)), axis=0))

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)
        self.fit = bayesmsd.lib.SplineFit(self.data, ss_order=1, n=2)

    # self.fit powerlaw, aka 2-point spline
    @patch('builtins.print')
    def testGeneric(self, mockprint=None):
        with Pool(5) as mypool:
            with nl.Parallelize(mypool.imap, mypool.imap_unordered):
                # conditional posterior
                profiler = bayesmsd.Profiler(self.fit, profiling=False)
                profiler.run_fit()
                res = profiler.best_estimate # just checking some path within best_estimate

                mci_c = profiler.find_MCI()
                self.assertLess(np.mean([np.abs(ci - m) for m, ci in mci_c.values()]), 1)

                # profile posterior
                profiler = bayesmsd.Profiler(self.fit, profiling=True)
                mci_p = profiler.find_MCI()
                self.assertLess(np.mean([np.abs(ci - m) for m, ci in mci_p.values()]), 1)

                # check best_estimate in the case where it's not the point estimate
                res = profiler.best_estimate
                self.assertIs(res, profiler.point_estimate)
                res['params']["y1"] -= 1
                params_array = profiler.min_target_from_fit.params_dict2array(res['params'])
                res['logL'] = -profiler.min_target_from_fit(params_array)
                self.assertIsNot(profiler.best_estimate, res)

                # Artificially create a situation with a bad point estimate
                # IRL this can happen in rugged landscapes
                # profiling
                profiler = bayesmsd.Profiler(self.fit, profiling=True, verbosity=3)
                profiler.point_estimate = res
                mci2_p = profiler.find_MCI()

                self.assert_mci_almost_equal(mci_p, mci2_p, places=2)

                # check replacement of gradient fit by simplex
                # this will still fail, because the simplex also can't work
                # with maxfev=1
                with self.assertRaises(RuntimeError):
                    profiler.run_fit(init_from=profiler.point_estimate, maxfev=1)

                # conditional
                profiler = bayesmsd.Profiler(self.fit, profiling=False, verbosity=3)
                profiler.point_estimate = res
                mci2_c = profiler.find_MCI()

                self.assert_mci_almost_equal(mci_p, mci2_p, places=2)

    def testMaxFitRuns(self):
        profiler = bayesmsd.Profiler(self.fit, max_fit_runs=2)
        with self.assertRaises(RuntimeError):
            profiler.find_MCI()

    def testNonidentifiability(self):
        names = ['y0', 'y1']
        for name in names:
            self.fit.parameters[name].max_linearization_moves=(0, 0)

        profiler = bayesmsd.Profiler(self.fit)
        mci = profiler.find_MCI()

        for name in names:
            self.assert_array_equal(mci[name][1], [-np.inf, np.inf])
    
    def test_singleparam_no_profiling(self):
        self.fit.parameters['y0'].fix_to = 0
        profiler = bayesmsd.Profiler(self.fit, profiling=True, verbosity=0)
        self.assertFalse(profiler.profiling)

    def testClosestRes(self):
        profiler = bayesmsd.Profiler(self.fit)
        profiler.run_fit()
        res = profiler.point_estimate
        profiler.cur_param = "y0"

        closest = profiler.find_closest_res(res['params']["y0"])
        self.assertIs(closest, res)

        with self.assertRaises(RuntimeError):
            closest = profiler.find_closest_res(res['params']["y0"] + 1, direction=1)

#     def testBracketStrategy(self):
#         self.fit.bounds = [(-1, np.inf), (0, 2)]
#         profiler = bayesmsd.Profiler(self.fit, profiling=False)
# 
#         pe_params = np.array([0, 0.1])
#         profiler.point_estimate = {'params' : pe_params,
#                                    'logL' : profiler.min_target_from_fit(pe_params),
#                                    }
#         profiler.expand_bracket_strategy()
# 
#         for strat in profiler.bracket_strategy:
#             self.assertFalse(strat['multiplicative'])
# 
#         profiler.iparam = 1
#         profiler.min_target_from_fit = lambda *args, **kwargs : np.inf
#         p, pL = profiler.iterate_bracket_point(0.01, profiler.point_estimate['logL'], -1)
#         self.assertEqual(p, 0)

    def testLikelihoodShapes(self):
        profiler = bayesmsd.Profiler(self.fit, profiling=False)

        # Hit bounds while bracketing (leftwards)
        # Hit jump while bisecting    (rightwards)
        profiler.fit.parameters['y0'].bounds = (0, 1)
        profiler.fit.parameters['y1'].bounds = (0, 1)
        profiler.point_estimate = {'params' : {'y0' : 0.5, 'y1' : 0.5},
                                   'logL' : 0.,
                                  }

        # <some ugly hacking to get a user-defined likelihood function>
        min_target = self.fit.MinTarget(self.fit)
        def _min_target(self, params_array):
            params = min_target.params_array2dict(params_array)
            return 0 if params['y0'] < 0.73 else 1e10

# vvvvvvvvvvvvvvvvvvvvv
        MinTarget__call__ = type(min_target).__call__
        type(min_target).__call__ = _min_target
# ^^^^^^^^^^^^^^^^^^^^^

        profiler.min_target_from_fit = min_target

        m, roots = profiler.find_single_MCI("y0")
        self.assertEqual(roots[0], 0)
        self.assertAlmostEqual(roots[1], 0.73)

# vvvvvvvvvvvvvvvvvvvvv
        type(min_target).__call__ = MinTarget__call__
# ^^^^^^^^^^^^^^^^^^^^^

class TestRandomStuff(myTestCase):
    def test_MSD(self):
        data = nl.TaggedSet([nl.Trajectory([[1, 2, 3], [4, 5, 6]])], hasTags=False)
        fit = bayesmsd.lib.NPXFit(data, ss_order=1, n=0)
        params = {
            'log(σ²) (dim 0)' : -np.inf,
             'log(Γ) (dim 0)' : 0.387,
                  'α (dim 0)' : 0.89,
            'log(σ²) (dim 1)' : -np.inf,
             'log(Γ) (dim 1)' : 0.387,
                  'α (dim 1)' : 0.89,
            'log(σ²) (dim 2)' : -np.inf,
             'log(Γ) (dim 2)' : 0.387,
                  'α (dim 2)' : 0.89,
        }

        msd = fit.MSD(params)
        docstring = msd.__doc__
        self.assertIn(f"(dt,", docstring)
        self.assertIn(f"{params['α (dim 0)']}", docstring)
        self.assertIn(f"{np.exp(params['log(Γ) (dim 0)']):.4f}"[:-1], docstring) # prevent rounding, just truncate

        dt = np.arange(1, 10)
        logG  = params['log(Γ) (dim 0)']
        alpha = params[     'α (dim 0)']
        self.assert_array_almost_equal(msd(dt), data[0].d*np.exp(alpha*np.log(dt) + logG))
        self.assert_array_almost_equal(msd(dt), fit.MSD(params, dt))

    def test_generate(self):
        data = nl.TaggedSet([nl.Trajectory([[1, 2, 3], [4, 5, 6]])], hasTags=False)
        fit = bayesmsd.lib.NPXFit(data, ss_order=1, n=0)
        params = {
            'log(σ²) (dim 0)' : -np.inf,
             'log(Γ) (dim 0)' : 0.387,
                  'α (dim 0)' : 0.89,
            'log(σ²) (dim 1)' : -np.inf,
             'log(Γ) (dim 1)' : 0.387,
                  'α (dim 1)' : 0.89,
            'log(σ²) (dim 2)' : -np.inf,
             'log(Γ) (dim 2)' : 0.387,
                  'α (dim 2)' : 0.89,
        }

        data_sample = bayesmsd.gp.generate((fit, dict(params=params)), 10, n=2)
        self.assertEqual(len(data_sample), 2)
        self.assertEqual(len(data_sample[0]), 10)
        self.assertEqual(data_sample[0].d, 3)
        self.assertTrue(np.all(np.array([traj[0] for traj in data_sample]) == 0))

        fit = bayesmsd.lib.NPXFit(data, ss_order=0, n=1)
        params = {'log(σ²)' : -np.inf,
                  'log(Γ)' : 0.387, 'α' : 0.89,
                  'x0' : 0.5, 'y1' : 1,
                  }
        for name in list(params.keys()):
            params.update({f"{name} (dim {dim})" : params[name] for dim in range(fit.d)})
            del params[name]
        data_sample = bayesmsd.gp.generate((fit, dict(params=params)), 10, n=2)

        fit = bayesmsd.lib.TwoLocusRouseFit(data)
        params = {'log(σ²)' : -np.inf,
                  'log(Γ)' : 0.,
                  'log(J)' : 1.,
                  }
        for name in list(params.keys()):
            params.update({f"{name} (dim {dim})" : params[name] for dim in range(fit.d)})
            del params[name]
        data_sample = bayesmsd.gp.generate((fit.MSD(params), 0, 1), 10, n=2)
        self.assertEqual(len(data_sample), 2)
        self.assertEqual(len(data_sample[0]), 10)
        self.assertEqual(data_sample[0].d, 1)

        data_sample = bayesmsd.gp.generate((fit.MSD(params), 1, 1), 10, n=2)

class TestNewImplementation(myTestCase):
    def test_MSDfun(self):
        @bayesmsd.deco.MSDfun
        def non_imaging_fun(arg=5):
            pass # pragma: no cover
        
        self.assertIn('_kwargstring', non_imaging_fun.__dict__)

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])#, defaultTest='TestProfiler.testGeneric')
