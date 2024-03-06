"""
Make sure the models work with the emcee package.
"""

import emcee
import numpy as np

from . import fixtures


class UnilateralEmceeTestCase(
    fixtures.BinaryUnilateralModelMixin,
    fixtures.IgnoreWarningsTestCase,
):
    """Test the emcee package with the Unilateral model."""

    def setUp(self):
        super().setUp(graph_size="small")
        self.model.set_modality("PET", spec=0.86, sens=0.79)
        self.load_patient_data(filename="2021-usz-oropharynx.csv")


    def test_emcee(self):
        """Test the emcee package with the Unilateral model."""
        nwalkers, ndim = 50, len(self.model.get_params())
        nsteps = 100
        initial = self.rng.uniform(size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=self.model.likelihood,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
            parameter_names=list(self.model.get_params().keys()),
        )
        sampler.run_mcmc(initial, nsteps, progress=True)
        samples = sampler.get_chain(discard=int(0.9*nsteps), flat=True)
        self.assertGreater(sampler.acceptance_fraction.mean(), 0.2)
        self.assertLess(sampler.acceptance_fraction.mean(), 0.5)
        self.assertTrue(np.all(samples.mean(axis=0) >= 0.0))
        self.assertTrue(np.all(samples.mean(axis=0) <= 1.0))
