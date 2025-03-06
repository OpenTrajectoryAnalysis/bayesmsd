import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
np.seterr(all='raise') # pay attention to details
from scipy import special

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
    'TestParameters',
    'TestDiffusive',
    'TestRouseLoci',
    'TestProfiler',
    'TestRandomStuff',
    'TestNewImplementation',
    'TestFitSum',
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

class TestParameters(myTestCase):
    def _test_linearization(self, lin, pes, x, n):
        for pe in pes:
            self.assert_array_almost_equal(lin.from_linear(pe,
                                            lin.to_linear(pe, x)),
                                           x)
            self.assert_array_almost_equal(lin.to_linear(pe,
                                            lin.from_linear(pe, n)),
                                           n)

    def test_parameter_and_linearization(self):
        L = bayesmsd.parameters.Linearize

        param = bayesmsd.parameters.Parameter()
        self.assert_array_equal(param.bounds, [-np.inf, np.inf])
        self.assertIs(param, param.linearization.param)
        self.assertIsInstance(param.linearization, L.Exponential)
        self._test_linearization(param.linearization,
            np.array([-10, -3.7, 0.01, 5.4, 70.9]),
            np.array([-101, -37.5, -1.3, 0., 7.8, 18.9]),
            np.array([-10, -5, 0, 3, 7]),
        )

        param = bayesmsd.parameters.Parameter((0, 1))
        self.assertIs(param, param.linearization.param)
        self.assertIsInstance(param.linearization, L.Bounded)
        self._test_linearization(param.linearization,
            np.array([0, 0.1, 0.37, 0.54, 0.709, 1]),
            np.array([-1.01, -0.375, 0., 0.13, 0.78, 1., 18.9]),
            np.array([-10, -5, 0, 3, 7]),
        )

        param = bayesmsd.parameters.Parameter((1, np.inf))
        self.assertIs(param, param.linearization.param)
        self.assertIsInstance(param.linearization, L.Multiplicative)
        self._test_linearization(param.linearization,
            np.array([1, 2, 5, 7.8, 100.8]),
            np.array([1.3, 7.8, 18.9, 37.5, 101]),
            np.array([-10, -5, 0, 3, 7]),
        )

        param = bayesmsd.parameters.Parameter((1, np.inf), linearization=L.Bounded)
        self.assertIs(param, param.linearization.param)

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
                                              }},
                       xatol=np.nan,
                       )
        # provoke "infinite" penalization
        res2 = fit.run(init_from={'params' : {"x1" : 1e-20,
                                              "x2" : 0.5,
                                              "y0" : 0, "y1" : 0,
                                              "y2" : 0, "y3" : 0,
                                              }},
                       xatol=np.nan,
                       )

        # check compactify / decompactify cycle
        dt = np.array([1, 5, 23, 100, 579, np.inf])
        self.assert_array_almost_equal(np.log(dt), fit.decompactify_log(fit.compactify(dt)))
        fit = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=3)
        self.assert_array_almost_equal(np.log(dt), fit.decompactify_log(fit.compactify(dt)))

    def testNPX(self):
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=0)
        fit.parameters['log(σ²) (dim 0)'].fix_to = -np.inf
        fit.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'
        fit.parameters['log(σ²) (dim 2)'].fix_to = lambda p : p['log(σ²) (dim 1)']
        res = fit.run(verbosity=0)

        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=0, parametrization='(log(αΓ), α)')
        fit.parameters['log(σ²) (dim 0)'].fix_to = -np.inf
        fit.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'
        fit.parameters['log(σ²) (dim 2)'].fix_to = lambda p : p['log(σ²) (dim 1)']
        res2 = fit.run(verbosity=0)

        self.assertAlmostEqual(res['logL'], res2['logL'], delta=0.01)

        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=1)
        fit.parameters['log(σ²) (dim 0)'].fix_to = -np.inf
        fit.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'
        fit.parameters['log(σ²) (dim 2)'].fix_to = lambda p : p['log(σ²) (dim 1)']
        res = fit.run()
        
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=1, motion_blur_f=0.5)
        fit.parameters['log(σ²) (dim 0)'].fix_to = -np.inf
        fit.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'
        fit.parameters['log(σ²) (dim 2)'].fix_to = lambda p : p['log(σ²) (dim 1)']
        res = fit.run()

    def testNP(self):
        fit = bayesmsd.lib.NPFit(self.data)
        fit.parameters['m1 (dim 0)'].fix_to = None # also fit the trend
        res = fit.run()

    def testNP_marginalized(self):
        fit = bayesmsd.lib.NPFit(self.data)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        fit.parameters['log(αΓ) (dim 0)'].fix_to = '<marginalize>'
        res = fit.run()

    def test_python_vs_cython_logLs(self):
        from bayesmsd.src.gp_py import logL as GP_logL_py
        
        trace = self.data[0][:][:, 0]
        msd = np.arange(15).astype(float)
        msd[-1] = 30 # for ss_order = 0

        logL_0_py = GP_logL_py(trace, 0, msd)
        logL_05_py = GP_logL_py(trace, 0.5, msd)
        logL_1_py = GP_logL_py(trace, 1, msd)
        logL_0_cy = bayesmsd.gp.GP.logL(trace, 0, msd)
        logL_05_cy = bayesmsd.gp.GP.logL(trace, 0.5, msd)
        logL_1_cy = bayesmsd.gp.GP.logL(trace, 1, msd)

        self.assertAlmostEqual(logL_0_py, logL_0_cy)
        self.assertAlmostEqual(logL_05_py, logL_05_cy)
        self.assertAlmostEqual(logL_1_py, logL_1_cy)

        p0 = GP_logL_py(np.array([5]), 0, msd)
        self.assertTrue(np.isfinite(p0))

        # msd2C_ss0 does not give the actual covariance matrix C in the code;
        # so check by hand
        C = bayesmsd.gp.msd2C_ss0(msd, np.arange(10))
        _ = np.linalg.cholesky(C) # check positive definite

    def test_penalty(self):
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1)
        params = {
            'm1 (dim 0)'      : 0,
            'm1 (dim 1)'      : 0,
            'm1 (dim 2)'      : 0,
            'log(σ²) (dim 0)' : 0,
            'log(σ²) (dim 1)' : 0,
            'log(σ²) (dim 2)' : 0,
            'log(Γ) (dim 0)'  : 0,
            'log(Γ) (dim 1)'  : 0,
            'log(Γ) (dim 2)'  : 0,
            'α (dim 0)'       : 1,
            'α (dim 1)'       : 1,
            'α (dim 2)'       : 1,
        }

        params['α (dim 0)'] = 3 # out of bounds
        self.assertLess(fit._penalty(params), 0)

        # test the mechanism for edge cases that _penalty() doesn't catch
        fit.verbosity = 0 # suppress error message
        fit._penalty = lambda x: 0
        logL = fit.logL(params)
        self.assertEqual(logL, -fit.max_penalty)

    def test_expand_fix_values(self):
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=1)

        fit.parameters['log(Γ) (dim 2)'].fix_to = 'log(Γ) (dim 0)'
        fit.parameters['log(Γ) (dim 1)'].fix_to = 'log(Γ) (dim 2)'
        fit.parameters['log(σ²) (dim 0)'].fix_to = 0
        fit.parameters['α (dim 1)'].fix_to = '<marginalize>'
        fit.parameters['α (dim 2)'].fix_to = lambda p : 0.5

        marginalized, free, to_const, to_other, to_call = fit.expand_fix_values()
        self.assertListEqual(marginalized, ['α (dim 1)'])
        self.assertIn(('log(σ²) (dim 0)', 0), to_const)
        self.assertIn(('log(Γ) (dim 2)', 'log(Γ) (dim 0)'), to_other)
        self.assertIn(('log(Γ) (dim 1)', 'log(Γ) (dim 2)'), to_other)
        self.assertEqual(to_call[0][0], 'α (dim 2)')

        # Attempt circular fixing
        fit.verbosity = 0 # mute error message
        fit.parameters['log(Γ) (dim 2)'].fix_to = 'log(Γ) (dim 1)'
        with self.assertRaises(RuntimeError):
            _ = fit.expand_fix_values()

        fit.parameters['log(Γ) (dim 1)'].fix_to = 'log(Γ) (dim 0)'
        fit.parameters['log(Γ) (dim 2)'].fix_to = 'α (dim 2)'
        with self.assertRaises(RuntimeError):
            _ = fit.expand_fix_values()

class TestRouseLoci(myTestCase):
    def setUp(self):
        model = rouse.Model(10, d=2)
        tracked = [2, 7]
        def traj():
            conf = model.conf_ss()
            traj = []
            for _ in range(10):
                conf = model.evolve(conf)
                traj.append(conf[tracked[0]] - conf[tracked[1]])

            return nl.Trajectory(traj)

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    @patch('builtins.print')
    def testSpline(self, mock_print):
        fit = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=4)
        res = fit.run()

        for name in ["x1", "x2"]:
            self.assertGreater(res['params'][name], 0)
            self.assertLess(res['params'][name], 2)

        self.assertSetEqual(set(fit.independent_parameters()),
                            {'x1', 'x2', 'y0', 'y1', 'y2', 'y3'})

        # Test refining a spline fit
        fit2 = bayesmsd.lib.SplineFit(self.data, ss_order=0, n=6,
                                    previous_spline_fit_and_result = (fit, res),
                                    )

        self.assertSetEqual(set(fit2.independent_parameters()),
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
        self.assertSetEqual(set(fit.independent_parameters()),
                            {'log(Γ) (dim 0)', 'log(J) (dim 0)'})

        fit = bayesmsd.lib.TwoLocusRouseFit(self.data, motion_blur_f=1, parametrization='(log(τ), log(J))')
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf
        res2 = fit.run()
        self.assertAlmostEqual(res['logL'], res2['logL'], delta=0.01)

        fit = bayesmsd.lib.TwoLocusRouseFit(self.data, motion_blur_f=1, parametrization='(log(Γ), log(τ))')
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf
        res2 = fit.run()
        self.assertAlmostEqual(res['logL'], res2['logL'], delta=0.01)

    def testHeuristic(self):
        fit = bayesmsd.lib.TwoLocusHeuristicFit(self.data)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf
        res = fit.run()

        params = res['params']
        params['n (dim 0)'] = np.inf
        msd = fit.MSD(params, dt=np.arange(10))
        self.assertTrue(np.all(np.isfinite(msd)))

        fit = bayesmsd.lib.TwoLocusHeuristicFit(self.data, parametrization='(log(Γ), log(J))')
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf
        res2 = fit.run()
        self.assertAlmostEqual(res['logL'], res2['logL'], delta=0.01)

        fit = bayesmsd.lib.TwoLocusHeuristicFit(self.data, parametrization='(log(Γ), log(τ))')
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf
        res3 = fit.run()
        self.assertAlmostEqual(res['logL'], res2['logL'], delta=0.01)

    def testSS05trimming(self):
        fit = bayesmsd.lib.TwoLocusRouseFit(self.data)
        fit.ss_order = 0.5
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        res = fit.run()

        res['params']['log(J) (dim 0)'] = 100
        target = fit.MinTarget(fit)
        logL = target(target.params_dict2array(res['params']))
        self.assertTrue(np.isfinite(logL))

        res['params']['log(J) (dim 0)'] = np.nan
        with self.assertRaises(ValueError):
            target(target.params_dict2array(res['params']))

    def testEvidence(self):
        fit = bayesmsd.lib.TwoLocusRouseFit([self.data[0].dims([0])])
        fit.parameters[f"log(σ²) (dim 0)"].fix_to = -np.inf 
        ev, (xi, logL, logprior) = fit.evidence(return_evaluations=True)
        self.assertTrue(np.isfinite(ev))
        with np.errstate(under='ignore'):
            self.assertEqual(ev, special.logsumexp(logL+logprior))

        # If a fit has no free parameters (which might happen, e.g. with
        # marginalization), evidence() should just return the likelihood
        fit.parameters[f"log(Γ) (dim 0)"].fix_to = '<marginalize>'
        fit.parameters[f"log(J) (dim 0)"].fix_to = 0
        self.assertEqual(len(fit.independent_parameters()), 0)
        res = fit.run()
        self.assertTrue(np.isfinite(res['logL']))
        ev = fit.evidence()
        self.assertTrue(np.isfinite(ev))


    def testEvidence_3D(self):
        fit = bayesmsd.lib.TwoLocusRouseFit([self.data[0]])
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        ev = fit.evidence(init_from_params={'log(Γ) (dim 0)' : 0., 'log(J) (dim 0)' : 0.})

    @patch('builtins.print')
    def testNPX(self, mock_print):
        fit = bayesmsd.lib.NPXFit(self.data, ss_order=0.5, n=1)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        res = fit.run()

        fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=1)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        res = fit.run()

        new_fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=2,
                                    previous_NPXFit_and_result = (fit, res),
                                    )
        for dim in range(new_fit.d):
            new_fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        new_res = new_fit.run()
        self.assertGreater(new_res['logL'], res['logL'])

        new2_fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=0,
                                     previous_NPXFit_and_result = (new_fit, new_res),
                                     )
        for dim in range(new2_fit.d):
            new2_fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
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
        for dim in range(new3_fit.d):
            new3_fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        try:
            new3_res = new3_fit.run()
            self.assertLess(new3_res['logL'], new_res['logL'] + 2)
        except RuntimeError: # pragma: no cover
            pass

        new4_fit = bayesmsd.lib.NPXFit(self.data, ss_order=1, n=1,
                                     previous_NPXFit_and_result = (new2_fit, new2_res),
                                     )
        for dim in range(new4_fit.d):
            new4_fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        try:
            new4_res = new4_fit.run()
            self.assertGreater(new4_res['logL'], new2_res['logL'] - 2)
        except RuntimeError: # pragma: no cover
            pass

        with self.assertRaises(ValueError):
            fit = bayesmsd.lib.NPXFit(self.data, ss_order=0, n=0)

        with self.assertRaises(ValueError):
            data = self.data.apply(lambda traj: nl.Trajectory(traj[:][:, 0]))
            fit = bayesmsd.lib.NPXFit(data, ss_order=0, n=5,
                                    previous_NPXFit_and_result = (new2_fit, new2_res),
                                    )

    def testDiscreteRouse(self):
        fit = bayesmsd.lib.DiscreteRouseFit(self.data)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        res = fit.run()

        fit = bayesmsd.lib.DiscreteRouseFit(self.data, motion_blur_f=1.)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        res = fit.run()

        fit = bayesmsd.lib.DiscreteRouseFit(self.data, motion_blur_f=1., use_approx=True)
        for dim in range(fit.d):
            fit.parameters[f"log(σ²) (dim {dim})"].fix_to = -np.inf 
        res = fit.run()

class TestFitGroup(myTestCase):
    def setUp(self):
        def gen_msd(G=5.7, a=0.62, noise2=0, f=0, lagtime=1):
            @bayesmsd.deco.MSDfun
            @bayesmsd.deco.imaging(noise2=noise2, f=f/lagtime, alpha0=a)
            def msd(dt):
                return G*(dt*lagtime)**a
            return msd

        dt_noise_f = {
                'a' : (10, 50, 10),
                'b' : (1, 5, 0.1),
                }

        data = nl.TaggedSet()
        for tag, (dt, noise, f) in dt_noise_f.items():
            d = bayesmsd.gp.generate((gen_msd(lagtime=dt, noise2=noise, f=f), 1, 2), T=10, n=5)
            for traj in d:
                traj.meta['Δt'] = dt
            d.addTags(tag)
            data |= d

        fits_dict = {}
        for tag, (_, _, f) in dt_noise_f.items():
            data.makeSelection(tag)
            fit = bayesmsd.lib.NPFit(data, motion_blur_f=f, parametrization='(log(αΓ), α)')
            fit.parameters['log(σ²) (dim 1)'].fix_to = 'log(σ²) (dim 0)'
            fits_dict[tag] = fit

        data.makeSelection()

        fitgroup = bayesmsd.FitGroup(fits_dict)
        fitgroup.parameters['b log(αΓ) (dim 0)'].fix_to = 'a log(αΓ) (dim 0)'
        fitgroup.parameters[      'b α (dim 0)'].fix_to =       'a α (dim 0)'

        self.fitgroup = fitgroup

    def test_run(self):
        self.fitgroup.likelihood_chunksize = 1
        with nl.Parallelize(n=2):
            res = self.fitgroup.run()
        self.assertEqual(res['params']['b log(αΓ) (dim 0)'], res['params']['a log(αΓ) (dim 0)'])

        self.fitgroup.likelihood_chunksize = 0
        res = self.fitgroup.run()

    def test_evidence(self):
        self.fitgroup.verbosity = 0
        with nl.Parallelize(2):
            ev = self.fitgroup.evidence(likelihood_chunksize=100)
        self.assertTrue(np.isfinite(ev))

# This takes quite long to execute; the idea was to check that FitGroup runs with marginalized parameters
#     def test_marginalization(self):
#         self.fitgroup.parameters['a log(αΓ) (dim 0)'].fix_to = 0
#         self.fitgroup.parameters['a α (dim 0)'].fix_to = '<marginalize>'
#         self.assertTrue(len(self.fitgroup.independent_parameters()), 0)
#         ev = self.fitgroup.evidence() # should just evaluate once
#         self.assertTrue(np.isfinite(ev))
    
    def test_logprior(self):
        params = {
                'a log(αΓ) (dim 0)' : -10,
                'a α (dim 0)' : 0.7,
                }
        pi1 = self.fitgroup.logprior(params)

        params['b α (dim 0)'] = 1.3
        pi2 = self.fitgroup.logprior(params)

        self.assertGreater(pi1, pi2)

class TestProfiler(myTestCase):
    # set up diffusive data set
    def setUp(self):
        def traj():
            return nl.Trajectory(np.cumsum(np.random.normal(size=(10, 1)), axis=0))

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)
        self.fit = bayesmsd.lib.SplineFit(self.data, ss_order=1, n=2)
        self.fit.likelihood_chunksize = 0

    # self.fit powerlaw, aka 2-point spline
    @patch('builtins.print')
    def testGeneric(self, mockprint=None):
        with nl.Parallelize(): # for funsies, won't make things faster
            # conditional posterior
            profiler = bayesmsd.Profiler(self.fit, profiling=False)
            with self.assertRaises(RuntimeError):
                profiler.find_single_MCI("y0")
            profiler.run_fit()
            res = profiler.best_estimate # just checking some path within best_estimate

        # ensure more than one bracketing step
        profiler.fit.parameters['y0'].linearization.step=0.01
        mci_c = profiler.find_MCI()
        profiler.fit.parameters['y0'].linearization.step=1.
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
            pe = profiler.point_estimate
            pe['params']['y1'] -= 1 # ensure that gradient fit does not just converge
                                    # after first evaluation
            profiler.run_fit(init_from=pe, maxfev=1)

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
            self.assert_array_equal(mci[name][1], [-bayesmsd.lib._MAX_LOG, bayesmsd.lib._MAX_LOG])
    
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

        MinTarget__call__ = type(min_target).__call__
        type(min_target).__call__ = _min_target

        profiler.min_target_from_fit = min_target

        m, roots = profiler.find_MCI("y0")["y0"]
        self.assertEqual(roots[0], 0)
        self.assertAlmostEqual(roots[1], 0.73)

        type(min_target).__call__ = MinTarget__call__

class TestRandomStuff(myTestCase):
    def test_full_params(self):
        fit = bayesmsd.lib.NPFit([[0]])
        fit.parameters['log(σ²) (dim 0)'].fix_to = 6.94

        params = {
                'α (dim 0)' : 0.5,
                'log(Γ) (dim 0)' : 1.,
                }
        params = fit.fill_dependent_params(params)
        self.assertEqual(params['log(σ²) (dim 0)'], 6.94)

    def test_xatol_convergence(self):
        fit = bayesmsd.lib.NPFit([[0]]) # dummy
        fit.parameters['log(σ²) (dim 0)'].fix_to = -np.inf
        fit.parameters[      'α (dim 0)'].fix_to = 1.

        G_eval = np.exp(-np.arange(100))
        mt = fit.MinTarget(fit)
        mt.xatol = 1e-5
        mt.eval_atomic = lambda params: np.log(params[0]) # won't converge
        with self.assertRaises(mt.xatolConverged):
            _ = [mt(np.array([G])) for G in G_eval]

    def test_MSD(self):
        data = nl.TaggedSet([nl.Trajectory([[1, 2, 3], [4, 5, 6]])], hasTags=False)
        fit = bayesmsd.lib.NPXFit(data, ss_order=1, n=0)
        params = {
            'log(σ²) (dim 0)' : -np.inf,
             'log(Γ) (dim 0)' : 0.387,
                  'α (dim 0)' : 0.89,
                 'm1 (dim 0)' : 0.54,
            'log(σ²) (dim 1)' : -np.inf,
             'log(Γ) (dim 1)' : 0.387,
                  'α (dim 1)' : 0.89,
                 'm1 (dim 1)' : 0.54,
            'log(σ²) (dim 2)' : -np.inf,
             'log(Γ) (dim 2)' : 0.387,
                  'α (dim 2)' : 0.89,
                 'm1 (dim 2)' : 0.54,
        }

        msd = fit.MSD(params)
        docstring = msd.__doc__
        self.assertIn(f"(dt,", docstring)
        self.assertIn(f"{params['α (dim 0)']}", docstring)
        self.assertIn(f"{np.exp(params['log(Γ) (dim 0)']):.4f}"[:-1], docstring) # prevent rounding, just truncate

        dt = np.arange(1, 10)
        logG  = params['log(Γ) (dim 0)']
        alpha = params[     'α (dim 0)']
        m     = params[    'm1 (dim 0)']
        self.assert_array_almost_equal(msd(dt), data[0].d*np.exp(alpha*np.log(dt) + logG))
        self.assert_array_almost_equal(msd(dt), fit.MSD(params, dt))

    def test_generate(self):
        data = nl.TaggedSet([nl.Trajectory([[1, 2, 3], [4, 5, 6]])], hasTags=False)
        fit = bayesmsd.lib.NPXFit(data, ss_order=1, n=0)
        params = {
            'log(σ²) (dim 0)' : -np.inf,
             'log(Γ) (dim 0)' : 0.387,
                  'α (dim 0)' : 0.89,
                 'm1 (dim 0)' : 0.54,
            'log(σ²) (dim 1)' : -np.inf,
             'log(Γ) (dim 1)' : 0.387,
                  'α (dim 1)' : 0.89,
                 'm1 (dim 1)' : 0.54,
            'log(σ²) (dim 2)' : -np.inf,
             'log(Γ) (dim 2)' : 0.387,
                  'α (dim 2)' : 0.89,
                 'm1 (dim 2)' : 0.54,
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
                  'm1' : 0,
                  }
        for name in list(params.keys()):
            params.update({f"{name} (dim {dim})" : params[name] for dim in range(fit.d)})
            del params[name]
        data_sample = bayesmsd.gp.generate((fit, dict(params=params)), 10, n=2)

        fit = bayesmsd.lib.TwoLocusRouseFit(data)
        params = {'log(σ²)' : -np.inf,
                  'log(Γ)' : 0.,
                  'log(J)' : 1.,
                  'm1' : 0,
                  }
        for name in list(params.keys()):
            params.update({f"{name} (dim {dim})" : params[name] for dim in range(fit.d)})
            del params[name]
        data_sample = bayesmsd.gp.generate((fit.MSD(params), 0, 1), 10, n=2)
        self.assertEqual(len(data_sample), 2)
        self.assertEqual(len(data_sample[0]), 10)
        self.assertEqual(data_sample[0].d, 1)

        data_sample = bayesmsd.gp.generate((fit.MSD(params), 1, 1), 10, n=2)

    def test_generate_ds_like(self):
        data = nl.TaggedSet([(nl.Trajectory([1, 2, np.nan, 4]), {'foo'}),
                             (nl.Trajectory([np.nan, 2, 3, 4]), {'bar', 'baz'})])

        @bayesmsd.deco.MSDfun
        def msd(dt):
            return dt

        data_new = bayesmsd.gp.generate_dataset_like(data, (msd, 1, 1))

        self.assertEqual(data._tags[0], data_new._tags[0])
        self.assertIsNot(data._tags[0], data_new._tags[0])
        data_new._tags[0].add('moo')
        self.assertNotEqual(data._tags[0], data_new._tags[0])

        self.assert_array_equal(np.isnan(data_new[1][:][:, 0]),
                                np.array([True, False, False, False]),
                                )

class TestNewImplementation(myTestCase):
    def test_MSDfun(self):
        @bayesmsd.deco.MSDfun
        def non_imaging_fun(arg=5):
            pass # pragma: no cover
        
        self.assertIn('_kwargstring', non_imaging_fun.__dict__)

class TestFitSum(myTestCase):
    def setUp(self):
        def traj():
            dat = np.cumsum(np.random.normal(size=(10, 3)), axis=0)
            dat += np.random.normal(size=(10, 3)) # add some noise
            return nl.Trajectory(dat)

        self.data = nl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    def testTwoNPFits(self):
        fit1 = bayesmsd.lib.NPFit(self.data)
        fit1.parameters['α (dim 0)'].fix_to = 0.5
        fit2 = bayesmsd.lib.NPFit(self.data)
        fit2.parameters['α (dim 0)'].fix_to = 1
        for dim in range(3):
            fit1.parameters[f'log(σ²) (dim {dim})'].fix_to = -np.inf
            fit2.parameters[f'log(σ²) (dim {dim})'].fix_to = -np.inf

        # Test a few error scenarios
        dummy_m1 = bayesmsd.lib.NPFit(self.data)
        dummy_m1.parameters['m1 (dim 0)'].fix_to = 5
        with self.assertRaises(ValueError):
            bayesmsd.FitSum({'1' : fit1, '2' : bayesmsd.lib.NPFit([[0]])}) # different data
        with self.assertRaises(ValueError):
            bayesmsd.FitSum({'1' : fit1, '2' : bayesmsd.lib.TwoLocusRouseFit(self.data)}) # different ss_order
        with self.assertRaises(ValueError):
            bayesmsd.FitSum({'1' : fit1, '2' : dummy_m1}) # m1 != 0

        fitsum = bayesmsd.FitSum({'α=0.5' : fit1, 'α=1' : fit2})
        for dim in range(1, 3):
            fitsum.parameters[f'log(σ²) (dim {dim})'].fix_to = 'log(σ²) (dim 0)'

        res = fitsum.run()
        self.assertTrue(np.isfinite(res['logL']))

if __name__ == '__main__': # pragma: no cover
    import cProfile
    with cProfile.Profile() as pr:
        unittest.main(module=__file__.split('/')[-1][:-3], exit=False)#, argv=sys.argv+['-v'])
        pr.dump_stats('/'.join(__file__.split('/')[:-1]+['profiling.stats']))
