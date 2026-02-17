"""Test if setting params via the named subset works correctly."""

import unittest

from lymph.types import ExtraParamsError

from . import fixtures
from .fixtures import RNG


class UnilateralNamedParamsTestCase(
    fixtures.BinaryUnilateralModelMixin,
    unittest.TestCase,
):
    """Tests for named params on unilateral models."""

    def test_set_named_params_default_behavior(self) -> None:
        """Ensure `set_named_params` works as `set_params` when no `named_params` set."""
        params = self.model.get_params(as_dict=True)
        new_params = {param: RNG.uniform() for param in params.keys()}
        self.model.set_named_params(**new_params)
        self.assertEqual(new_params, self.model.get_params(as_dict=True))

    def test_named_params_setter(self) -> None:
        """Check that setting `named_params` works correctly."""
        with self.assertRaises(ValueError):
            self.model.named_params = ["invalid identifier!"]

        with self.assertRaises(ValueError):
            self.model.named_params = 123

        params = self.model.get_params(as_dict=True).keys()
        params_subset = [param for param in params if RNG.uniform() > 0.5]
        self.model.named_params = params_subset

        for stored, subset in zip(
            self.model.named_params,
            params_subset,
            strict=True,
        ):
            self.assertEqual(stored, subset)

    def test_set_named_params_named_easy_subset(self) -> None:
        """Ensure `set_named_params` works correctly with an easy subset.

        An "easy subset" is a literal subset of the params.
        """
        params = self.model.get_params(as_dict=True)
        new_params = {param: RNG.uniform() for param in params.keys()}
        params_subset = {k: RNG.uniform() for k in params if RNG.uniform() > 0.5}

        self.model.set_params(**new_params)
        self.model.named_params = params_subset.keys()
        self.model.set_named_params(**params_subset)

        for param, new_val in new_params.items():
            stored_params = self.model.get_params(as_dict=True)
            if param in params_subset:
                self.assertEqual(params_subset[param], stored_params[param])
            else:
                self.assertEqual(new_val, stored_params[param])

        self.assertEqual(set(params_subset.keys()), set(self.model.named_params))

    def test_set_named_params_raises(self) -> None:
        """Ensure `set_named_params` raises when provided with invalid keys."""
        self.model.named_params = ["spread"]
        with self.assertRaises(ExtraParamsError):
            self.model.set_named_params(invalid=RNG.uniform())

    def test_set_named_params_allows_global_alias_not_named(self) -> None:
        """Allow global keys like `spread` even if not in `named_params`."""
        params = self.model.get_params(as_dict=True)
        new_params = {param: RNG.uniform() for param in params.keys()}
        first_lnl = list(self.model.graph.lnls.keys())[0]
        first_lnl_param = f"Tto{first_lnl}_spread"

        self.model.set_params(**new_params)
        self.model.named_params = [first_lnl_param]
        spread_val = RNG.uniform()
        self.model.set_named_params(spread=spread_val)

        stored_params = self.model.get_params(as_dict=True)
        for param, stored_param in stored_params.items():
            if "spread" in param:
                self.assertEqual(stored_param, spread_val)

    def test_set_named_params_hard_subset(self) -> None:
        """Ensure `set_named_params` works correctly with a hard subset.

        A "hard subset" is a subset that includes "global params". I.e., `spread` would
        not be a literal subset, because those are named something like `TtoII_spread`. But
        the `set_params()` method does accept it and will set all spread params with the
        provided value. It should be possible to set the `named_params` to such names and
        then set them with the `set_named_params()` method.
        """
        params = self.model.get_params(as_dict=True)
        new_params = {param: RNG.uniform() for param in params.keys()}
        first_lnl = list(self.model.graph.lnls.keys())[0]
        first_lnl_param = f"Tto{first_lnl}_spread"
        params_subset = {k: RNG.uniform() for k in ["spread", first_lnl_param]}

        self.model.set_params(**new_params)
        self.model.named_params = params_subset.keys()
        self.model.set_named_params(**params_subset)

        stored_params = self.model.get_params(as_dict=True)
        for param, new_val, stored_param in zip(
            params.keys(),
            new_params.values(),
            stored_params.values(),
            strict=True,
        ):
            if param == first_lnl_param:
                self.assertEqual(params_subset[first_lnl_param], stored_param)
            elif "spread" in param:
                self.assertEqual(params_subset["spread"], stored_param)
            else:
                self.assertEqual(new_val, stored_param)

    def test_get_named_params_hard_subset(self) -> None:
        """Check that getting globals like `spread` works correctly."""
        params = self.model.get_params(as_dict=True)
        new_params = {param: RNG.uniform() for param in params.keys()}
        first_lnl = list(self.model.graph.lnls.keys())[0]
        first_lnl_param = f"Tto{first_lnl}_spread"
        params_subset = {k: RNG.uniform() for k in ["spread", first_lnl_param]}

        self.model.set_params(**new_params)
        self.model.named_params = params_subset.keys()
        self.model.set_named_params(**params_subset)

        stored_params = self.model.get_named_params()
        self.assertEqual(params_subset, stored_params)


class BilateralNamedParamsTestCase(
    fixtures.BilateralModelMixin,
    unittest.TestCase,
):
    """Tests for named params on bilateral models."""

    def test_named_params_setter(self) -> None:
        """Check that setting `named_params` works correctly."""
        self.model.named_params = ["ipsi_spread"]
        self.assertEqual(self.model.named_params, ["ipsi_spread"])

    def test_set_global_params_for_side(self) -> None:
        """Check that setting e.g. `"ipsi_spread"` works as global param to ipsi side."""
        params = self.model.get_params(as_dict=True)
        new_params = {param: RNG.uniform() for param in params.keys()}

        self.model.named_params = ["ipsi_spread"]
        self.model.set_params(**new_params)
        ipsi_spread_val = RNG.uniform()
        self.model.set_named_params(ipsi_spread=ipsi_spread_val)

        ipsi_stored_params = self.model.ipsi.get_params(as_dict=True)

        for param, stored_param in ipsi_stored_params.items():
            if "spread" in param:
                self.assertEqual(stored_param, ipsi_spread_val)
            else:
                self.assertIn(stored_param, new_params.values())
