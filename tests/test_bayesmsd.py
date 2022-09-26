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
    'TestRouseSingleLocus',
    'TestProfiler',
    'TestRandomStuff',
]

# We test mostly the library implementations, since the base class `Fit` is
# abstract
# 
# We also test on so few data that all the results are useless and we don't
# attempt to check for correctness of the fit. This code is supposed to be
# technical tests and not benchmarks, so it should run fast.
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

class TestDiffusive(myTestCase):
    def setUp(self):
        def traj():
            return nl.Trajectory(np.cumsum(np.random.normal(size=(10, 3)), axis=0))

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    def testSpline(self):
        fit = bayesmsd.lib.SplineFit(self.data, ss_order=1, n=4)
        res = fit.run(verbosity=0, maxfev=500)

        for i in [0, 1]:
            self.assertGreater(res['params'][i], 0)
            self.assertLess(res['params'][i], 1)

        # provoke penalization
        res2 = fit.run(init_from={'params' : np.array([1e-8, 0.5, 0, 0, 0, 0]), 'logL' : 0})
        # provoke "infinite" penalization
        res2 = fit.run(init_from={'params' : np.array([1e-20, 0.5, 0, 0, 0, 0]), 'logL' : 0})

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

        for i in [0, 1]:
            self.assertGreater(res['params'][i], 0)
            self.assertLess(res['params'][i], 2)

        self.assertEqual(fit.number_of_fit_parameters(), 6)

        # Test refining a spline fit
        fit2 = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=6,
                                    previous_spline_fit_and_result = (fit, res),
                                    )

        self.assertEqual(fit2.number_of_fit_parameters(), 10)

        with self.assertRaises(RuntimeError):
            res2 = fit2.run(optimization_steps=('gradient',), maxfev=10)


    def testRouse(self):
        fit = bayesmsd.lib.TwoLocusRouseFit(self.data, motion_blur_f=1)
        fit.fix_values += [(0, -np.inf), (3, -np.inf), (6, -np.inf)] # no localization error
        res = fit.run(full_output=True, optimization_steps=(dict(method='Nelder-Mead', options={'fatol' : 0.1, 'xatol' : 0.01}),))[-1][0]

        self.assertEqual(res['params'][0], -np.inf)
        self.assertEqual(fit.number_of_fit_parameters(), 2)

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

class TestRouseSingleLocus(myTestCase):
    def setUp(self):
        model = rouse.Model(11)
        tracked = 5
        def traj():
            conf = model.conf_ss()
            traj = []
            for _ in range(10):
                conf = model.evolve(conf)
                traj.append(conf[tracked])

            return nl.Trajectory(traj)

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    def testFit(self):
        fit = bayesmsd.lib.OneLocusRouseFit(self.data)
        fit.fix_values += [(0, -np.inf), (2, -np.inf), (4, -np.inf)] # no localization error
        res = fit.run(full_output=True, optimization_steps=(dict(method='Nelder-Mead', options={'fatol' : 0.1, 'xatol' : 0.01}),))[-1][0]

        self.assertEqual(res['params'][0], -np.inf)
        self.assertTrue(np.isfinite(res['params'][1]))

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
                profiler = bayesmsd.Profiler(self.fit, bracket_step=1.1, profiling=False)
                profiler.run_fit()
                res = profiler.best_estimate # just checking some path within best_estimate

                mci_c = profiler.find_MCI()
                self.assertLess(np.mean(np.abs(mci_c[:, [1, 2]] - mci_c[:, [0]])), 1)

                # profile posterior
                profiler = bayesmsd.Profiler(self.fit,
                                           bracket_strategy={
                                               'multiplicative' : True,
                                               'step' : 1.1,
                                               'nonidentifiable_cutoffs' : [10, 10],
                                           },
                                           profiling=True,
                                          )
                mci_p = profiler.find_MCI()
                self.assertLess(np.mean(np.abs(mci_p[:, [1, 2]] - mci_p[:, [0]])), 1)

                # check best_estimate in the case where it's not the point estimate
                res = profiler.best_estimate
                self.assertIs(res, profiler.point_estimate)
                res['params'][0] -= 1
                res['logL'] = -profiler.fit.get_min_target()(res['params'])
                self.assertIsNot(profiler.best_estimate, res)

                # Artificially create a situation with a bad point estimate
                # IRL this can happen in rugged landscapes
                # profiling
                profiler = bayesmsd.Profiler(self.fit, profiling=True, verbosity=3)
                profiler.point_estimate = res
                mci2_p = profiler.find_MCI()

                self.assertLess(np.max(np.abs(mci_p-mci2_p)), 0.01)

                # check replacement of gradient fit by simplex
                # this will still fail, because the simplex also can't work with maxfev=1
                with self.assertRaises(RuntimeError):
                    profiler.run_fit(init_from=profiler.point_estimate, maxfev=1)

                # conditional
                profiler = bayesmsd.Profiler(self.fit, profiling=False, verbosity=3)
                profiler.point_estimate = res
                mci2_c = profiler.find_MCI()

                self.assertLess(np.max(np.abs(mci_c-mci2_c)), 0.01)

    def testMaxFitRuns(self):
        profiler = bayesmsd.Profiler(self.fit, max_fit_runs=2)
        with self.assertRaises(RuntimeError):
            profiler.find_MCI()

    def testNonidentifiability(self):
        # also check bracket_strategy = list
        profiler = bayesmsd.Profiler(self.fit, bracket_strategy=2*[{
                                            'multiplicative' : False,
                                            'step' : 1e-5,
                                            'nonidentifiable_cutoffs' : [1e-5, 1e-5],
                                            }])
        mci = profiler.find_MCI()
        self.assert_array_equal(mci[:, [1, 2]], [[-np.inf, np.inf], [-np.inf, np.inf]])
    
    def testSingleiparam(self):
        self.fit.bounds[1] = (1e-10, np.inf) # make 'auto' brackets multiplicative
        profiler = bayesmsd.Profiler(self.fit)
        mci = profiler.find_MCI(iparam=1)
        self.assertTupleEqual(mci.shape, (3,))

    def test_singleparam_no_profiling(self):
        self.fit.bounds = [(-np.inf, np.inf)]
        profiler = bayesmsd.Profiler(self.fit, profiling=True, verbosity=0)
        self.assertFalse(profiler.profiling)

    def testClosestRes(self):
        profiler = bayesmsd.Profiler(self.fit)
        profiler.run_fit()
        res = profiler.point_estimate
        profiler.expand_bracket_strategy()

        profiler.iparam = 0
        closest = profiler.find_closest_res(res['params'][0])
        self.assertIs(closest, res)

        with self.assertRaises(RuntimeError):
            closest = profiler.find_closest_res(res['params'][0] + 1, direction=1)

    def testBracketStrategy(self):
        self.fit.bounds = [(-1, np.inf), (0, 2)]
        profiler = bayesmsd.Profiler(self.fit, profiling=False)

        pe_params = np.array([0, 0.1])
        profiler.point_estimate = {'params' : pe_params,
                                   'logL' : profiler.min_target_from_fit(pe_params),
                                   }
        profiler.expand_bracket_strategy()

        for strat in profiler.bracket_strategy:
            self.assertFalse(strat['multiplicative'])

        profiler.iparam = 1
        profiler.min_target_from_fit = lambda *args, **kwargs : np.inf
        p, pL = profiler.iterate_bracket_point(0.01, profiler.point_estimate['logL'], -1)
        self.assertEqual(p, 0)

    def testLikelihoodShapes(self):
        profiler = bayesmsd.Profiler(self.fit, profiling=False)
        profiler.iparam=0

        profiler.fit.bounds = [(0, 1), (0, 1)]
        profiler.point_estimate = {'params' : np.array([0.5, 0.5]),
                                   'logL' : 0.,
                                  }

        # Hit bounds while bracketing (leftwards)
        # Hit jump while bisecting    (rightwards)
        profiler.min_target_from_fit = lambda params: 0. if params[0] < 0.73 else 1e10
        m, roots = profiler.find_single_MCI(0)
        self.assertEqual(roots[0], 0)
        self.assertAlmostEqual(roots[1], 0.73)

class TestRandomStuff(myTestCase):
    def test_MSD(self):
        data = nl.TaggedSet([nl.Trajectory([[1, 2, 3], [4, 5, 6]])], hasTags=False)
        fit = bayesmsd.lib.NPXFit(data, ss_order=1, n=0)
        params = np.array(data[0].d*[-np.inf, 0.387, 0.89])

        msd = fit.MSD(dict(params=params))
        docstring = msd.__doc__
        self.assertIn(f"(dt,", docstring)
        self.assertIn(f"{params[2]}", docstring)
        self.assertIn(f"{np.exp(params[1]):.4f}"[:-1], docstring) # prevent rounding, just truncate

        dt = np.arange(1, 10)
        self.assert_array_almost_equal(msd(dt), data[0].d*np.exp(params[2]*np.log(dt) + params[1]))
        self.assert_array_almost_equal(msd(dt), fit.MSD(dict(params=params), dt))

    def test_generate(self):
        data = nl.TaggedSet([nl.Trajectory([[1, 2, 3], [4, 5, 6]])], hasTags=False)
        fit = bayesmsd.lib.NPXFit(data, ss_order=1, n=0)
        params = np.array(data[0].d*[-np.inf, 0.387, 0.89])

        data_sample = bayesmsd.gp.generate((fit, dict(params=params)), 10, n=2)
        self.assertEqual(len(data_sample), 2)
        self.assertEqual(len(data_sample[0]), 10)
        self.assertEqual(data_sample[0].d, 3)
        self.assertTrue(np.all(np.array([traj[0] for traj in data_sample]) == 0))

        fit = bayesmsd.lib.NPXFit(data, ss_order=0, n=1)
        params = np.array(data[0].d*[-np.inf, 0.387, 0.89, 0.5, 1])
        data_sample = bayesmsd.gp.generate((fit, dict(params=params)), 10, n=2)

        fit = bayesmsd.lib.TwoLocusRouseFit(data)
        params = np.array(data[0].d*[-np.inf, 0., 1.])
        data_sample = bayesmsd.gp.generate((fit.MSD(dict(params=params)), 0, 1), 10, n=2)
        self.assertEqual(len(data_sample), 2)
        self.assertEqual(len(data_sample[0]), 10)
        self.assertEqual(data_sample[0].d, 1)

        data_sample = bayesmsd.gp.generate((fit.MSD(dict(params=params)), 1, 1), 10, n=2)

class TestNewImplementation(myTestCase):
    def test_MSDfun(self):
        @bayesmsd.deco.MSDfun
        def non_imaging_fun(arg=5):
            pass # pragma: no cover
        
        self.assertIn('_kwargstring', non_imaging_fun.__dict__)

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
