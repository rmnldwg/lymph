"""
Fxitures for tests.
"""
import logging
import unittest
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import scipy as sp

from lymph import diagnose_times
from lymph.modalities import Clinical, Modality, Pathological
from lymph.models import Bilateral, Midline, Unilateral
from lymph.types import DataWarning, PatternType

MODALITIES = {
    "CT": Clinical(spec=0.81, sens=0.86),
    "FNA": Pathological(spec=0.95, sens=0.81),
}
RNG = np.random.default_rng(42)


class IgnoreWarningsTestCase(unittest.TestCase):
    """Test case that ignores warnings."""

    def setUp(self) -> None:
        """Ignore warnings."""
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        warnings.simplefilter("ignore", category=DataWarning)
        super().setUp()


def get_graph(size: str = "large") -> dict[tuple[str, str], list[str]]:
    """Return either a ``"small"``, a ``"medium"`` or a ``"large"`` graph."""
    if size == "small":
        return {
            ("tumor", "T"): ["II", "III"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): [],
        }

    if size == "medium":
        return {
            ("tumor", "T"): ["I", "II", "III", "IV"],
            ("lnl", "I"): ["II"],
            ("lnl", "II"): ["III"],
            ("lnl", "III"): ["IV"],
            ("lnl", "IV"): [],
        }

    if size == "large":
        return {
            ("tumor", "T"): ["I", "II", "III", "IV", "V", "VII"],
            ("lnl", "I"): [],
            ("lnl", "II"): ["I", "III", "V"],
            ("lnl", "III"): ["IV", "V"],
            ("lnl", "IV"): [],
            ("lnl", "V"): [],
            ("lnl", "VII"): [],
        }

    raise ValueError(f"Unknown graph size: {size}")


def get_logger(
    level: str = logging.INFO,
    handler: logging.Handler = logging.StreamHandler(),
) -> logging.Logger:
    """Return the :py:mod:`lymph` package's logger with a handler."""
    logger = logging.getLogger("lymph")
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def _create_random_frozen_dist(
    max_time: int,
    rng: np.random.Generator = RNG,
) -> np.ndarray:
    """Create a random frozen diagnose time distribution."""
    unnormalized = rng.random(size=max_time + 1)
    return unnormalized / np.sum(unnormalized)

def _create_random_parametric_dist(
    max_time: int,
    rng: np.random.Generator = RNG,
) -> diagnose_times.Distribution:
    """Create a binomial diagnose time distribution with random params."""
    def _pmf(support: np.ndarray, p: float = rng.random()) -> np.ndarray:
        return sp.stats.binom.pmf(support, p=p, n=max_time + 1)

    return diagnose_times.Distribution(
        distribution=_pmf,
        max_time=max_time,
    )

def create_random_dist(
    type_: str,
    max_time: int,
    rng: np.random.Generator = RNG,
) -> np.ndarray | Callable:
    """Create a random frozen or parametric distribution."""
    if type_ == "frozen":
        return _create_random_frozen_dist(max_time=max_time, rng=rng)

    if type_ == "parametric":
        return _create_random_parametric_dist(max_time=max_time, rng=rng)

    raise ValueError(f"Unknown distribution type: {type_}")


def create_random_pattern(lnls: list[str]) -> PatternType:
    """Create a random involvement pattern."""
    return {
        lnl: RNG.choice([True, False, None])
        for lnl in lnls
    }


class BinaryUnilateralModelMixin:
    """Mixin class for simple binary model fixture creation."""

    def setUp(self, graph_size: str = "large"):
        """Initialize a simple binary model."""
        super().setUp()
        self.rng = np.random.default_rng(42)
        self.graph_dict = get_graph(size=graph_size)
        self.model = Unilateral.binary(graph_dict=self.graph_dict)
        self.logger = get_logger(level=logging.INFO)


    def create_random_params(self) -> dict[str, float]:
        """Create random parameters for the model."""
        params = {
            f"{name}_{type_}": self.rng.random()
            for name, edge in self.model.graph.edges.items()
            for type_ in edge.get_params(as_dict=True).keys()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            params.update({
                f"{t_stage}_{type_}": self.rng.random()
                for t_stage, dist in self.model.get_all_distributions().items()
                for type_ in dist.get_params(as_dict=True).keys()
            })
        return params


    def init_diag_time_dists(self, **dists) -> None:
        """Init the diagnose time distributions."""
        for t_stage, type_ in dists.items():
            self.model.set_distribution(
                t_stage,
                create_random_dist(type_, self.model.max_time, self.rng),
            )


    def load_patient_data(
        self,
        filename: str = "2021-clb-oropharynx.csv",
    ) -> None:
        """Load patient data from a CSV file."""
        filepath = Path(__file__).parent / "data" / filename
        self.raw_data = pd.read_csv(filepath, header=[0,1,2])
        self.model.load_patient_data(self.raw_data, side="ipsi")


class BilateralModelMixin:
    """Mixin for testing the bilateral model."""
    model_kwargs: dict[str, Any] | None = None

    def setUp(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}

        super().setUp()
        self.rng = np.random.default_rng(42)
        self.graph_dict = get_graph("large")
        self.model = Bilateral(graph_dict=self.graph_dict, **self.model_kwargs)
        self.init_diag_time_dists(early="frozen", late="parametric")
        self.model.set_params(**self.create_random_params())
        self.logger = get_logger(level=logging.INFO)


    def init_diag_time_dists(self, **dists) -> None:
        """Init the diagnose time distributions."""
        for t_stage, type_ in dists.items():
            self.model.set_distribution(
                t_stage,
                create_random_dist(type_, self.model.max_time, self.rng),
            )


    def create_random_params(self) -> dict[str, float]:
        """Create a random set of parameters."""
        params = self.model.get_params(as_dict=True)

        for name in params:
            params[name] = self.rng.random()

        return params


    def load_patient_data(
        self,
        filename: str = "2021-usz-oropharynx.csv",
    ) -> None:
        """Load patient data from a CSV file."""
        filepath = Path(__file__).parent / "data" / filename
        self.raw_data = pd.read_csv(filepath, header=[0,1,2])
        self.model.load_patient_data(self.raw_data)



class TrinaryFixtureMixin:
    """Mixin class for simple trinary model fixture creation."""

    def setUp(self):
        """Initialize a simple trinary model."""
        self.rng = np.random.default_rng(42)
        self.graph_dict = get_graph(size="large")
        self.model = Unilateral.trinary(graph_dict=self.graph_dict)
        self.logger = get_logger(level=logging.INFO)


    def create_random_params(self) -> dict[str, float]:
        """Create random parameters for the model."""
        params = {
            f"{name}_{type_}": self.rng.random()
            for name, edge in self.model.graph.edges.items()
            for type_ in edge.get_params(as_dict=True).keys()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            params.update({
                f"{t_stage}_{type_}": self.rng.random()
                for t_stage, dist in self.model.get_all_distributions().items()
                for type_ in dist.get_params(as_dict=True).keys()
            })

        return params


    def init_diag_time_dists(self, **dists) -> None:
        """Init the diagnose time distributions."""
        for t_stage, type_ in dists.items():
            self.model.set_distribution(
                t_stage,
                create_random_dist(type_, self.model.max_time, self.rng),
            )


    def get_modalities_subset(self, names: list[str]) -> dict[str, Modality]:
        """Create a dictionary of modalities."""
        modalities_in_data = {
            "CT": Clinical(spec=0.76, sens=0.81),
            "MRI": Clinical(spec=0.63, sens=0.81),
            "PET": Clinical(spec=0.86, sens=0.79),
            "FNA": Pathological(spec=0.98, sens=0.80),
            "diagnostic_consensus": Clinical(spec=0.86, sens=0.81),
            "pathology": Pathological(spec=1.0, sens=1.0),
            "pCT": Clinical(spec=0.86, sens=0.81),
        }
        return {name: modalities_in_data[name] for name in names}


    def load_patient_data(
        self,
        filename: str = "2021-clb-oropharynx.csv",
    ) -> None:
        """Load patient data from a CSV file."""
        filepath = Path(__file__).parent / "data" / filename
        self.raw_data = pd.read_csv(filepath, header=[0,1,2])
        self.model.load_patient_data(self.raw_data, side="ipsi")


class MidlineFixtureMixin:
    """Mixing for testing the Midline model."""

    def setUp(
        self,
        seed: int = 42,
        graph_size: Literal["small", "medium", "large"] = "small",
        use_mixing: bool = True,
        use_central: bool = False,
        use_midext_evo: bool = True,
        is_symmetric: dict[str, bool] | None = None,
    ) -> None:
        """Initialize rng and model."""
        self.rng = np.random.default_rng(seed)
        self.model = Midline(
            graph_dict=get_graph(graph_size),
            is_symmetric=is_symmetric or {"tumor_spread": False, "lnl_spread": True},
            use_mixing=use_mixing,
            use_central=use_central,
            use_midext_evo=use_midext_evo,
        )


    def init_diag_time_dists(self, **dists) -> None:
        """Init the diagnose time distributions."""
        for t_stage, type_ in dists.items():
            self.model.set_distribution(
                t_stage,
                create_random_dist(type_, self.model.max_time, self.rng),
            )
