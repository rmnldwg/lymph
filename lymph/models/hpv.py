"""Module for HPV/noHPV lymphatic tumor progression models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from functools import wraps
from typing import Any, Literal

import numpy as np
import pandas as pd

from lymph import diagnosis_times, modalities, models, types, utils

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


def select_hpv_model(method):
    """Decorate methods that simply delegate to the `hpv` or `nohpv` model."""

    @wraps(method)
    def wrapper(self, *args, hpv_status: bool | None = None, **kwargs):
        if hpv_status is None:
            raise ValueError("Must provide hpv_status.")

        _model = self.hpv if hpv_status else self.nohpv
        return method(_model, *args, **kwargs)

    return wrapper


class HPVUnilateral(
    diagnosis_times.Composite,
    modalities.Composite,
    types.Model,
):
    """Class that models metastatic progression in HPV and non HPV lymphatic systems.

    This is achieved by creating two instances of the
    :py:class:`~lymph.models.Unilateral` model, one for the HPV+ and one for the
    HPV- patients.

    .. seealso::

        :py:class:`~lymph.models.Unilateral`
            Two instances of this class are created as attributes. One for the HPV+ and
            one for the HPV- model.

    .. warning::

        This class is still a bit experimental and not super thoroughly tested. It may
        especially cause issues if one wanted to use e.g. a bilateral model composed of
        two :py:class:`~lymph.models.HPVUnilateral` models.
    """

    def __init__(
        self,
        graph_dict: types.GraphDictType,
        named_params: Sequence[str] | None = None,
        uni_kwargs: dict[str, Any] | None = None,
        hpv_kwargs: dict[str, Any] | None = None,
        nohpv_kwargs: dict[str, Any] | None = None,
        **_kwargs,
    ) -> None:
        """Initialize a ``unilateral`` HPV model.

        The ``graph_dict`` is a dictionary of tuples as keys and lists of strings as
        values. It is passed to both :py:class:`.models.Unilateral` instances,
        which in turn pass it to the :py:class:`.graph.Representation` class that
        stores the graph.
        """
        self._init_models(
            graph_dict=graph_dict,
            uni_kwargs=uni_kwargs,
            hpv_kwargs=hpv_kwargs,
            nohpv_kwargs=nohpv_kwargs,
        )

        if named_params is not None:
            self.named_params = named_params

        diagnosis_times.Composite.__init__(
            self,
            distribution_children={"hpv": self.hpv, "nohpv": self.nohpv},
            is_distribution_leaf=False,
        )
        modalities.Composite.__init__(
            self,
            modality_children={"hpv": self.hpv, "nohpv": self.nohpv},
            is_modality_leaf=False,
        )

    def _init_models(
        self,
        graph_dict: dict[tuple[str], list[str]],
        uni_kwargs: dict[str, Any] | None = None,
        hpv_kwargs: dict[str, Any] | None = None,
        nohpv_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the two unilateral models."""
        uni_kwargs = uni_kwargs or {}

        _hpv_kwargs = uni_kwargs.copy()
        _hpv_kwargs["graph_dict"] = graph_dict
        _hpv_kwargs.update(hpv_kwargs or {})

        _nohpv_kwargs = uni_kwargs.copy()
        _nohpv_kwargs["graph_dict"] = graph_dict
        _nohpv_kwargs.update(nohpv_kwargs or {})

        self.hpv = models.Unilateral(**_hpv_kwargs)
        self.nohpv = models.Unilateral(**_nohpv_kwargs)

        # set b_2 key name
        self.base_2_key = list(self.hpv.graph.tumors.keys())[0] + "toII"

    @classmethod
    def binary(cls, *args, **kwargs) -> HPVUnilateral:
        """Initialize a binary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @classmethod
    def trinary(cls, *args, **kwargs) -> HPVUnilateral:
        """Initialize a trinary bilateral model.

        This is a convenience method that sets the ``allowed_states`` of the
        ``uni_kwargs`` to ``[0, 1, 2]``. All other ``args`` and ``kwargs`` are
        passed to the :py:meth:`.__init__` method.
        """
        uni_kwargs = kwargs.pop("uni_kwargs", {})
        uni_kwargs["allowed_states"] = [0, 1, 2]
        return cls(*args, uni_kwargs=uni_kwargs, **kwargs)

    @property
    def is_trinary(self) -> bool:
        """Return whether the model is trinary."""
        if self.hpv.is_trinary != self.nohpv.is_trinary:
            raise ValueError("Both models must be of the same 'narity'.")

        return self.hpv.is_trinary

    @property
    def is_binary(self) -> bool:
        """Return whether the model is binary."""
        if self.hpv.is_binary != self.nohpv.is_binary:
            raise ValueError("Both sides must be of the same 'narity'.")

        return self.nohpv.is_binary

    def get_tumor_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from tumor to LNLs."""
        params = {
            "hpv": self.hpv.get_tumor_spread_params(as_flat=as_flat),
            "nohpv": utils.flatten(
                {
                    self.base_2_key: self.nohpv.get_tumor_spread_params(as_flat=False)[
                        self.base_2_key
                    ],
                },
            )
            if as_flat is True
            else {
                self.base_2_key: self.nohpv.get_tumor_spread_params(as_flat=False)[
                    self.base_2_key
                ],
            },
        }
        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_lnl_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread from LNLs to tumor.

        Similarly to the :py:meth:`.get_tumor_spread_params` method.
        However, since the spread from LNLs is symmetric in HPV and noHPV,
        the spread parameters are the same and only one set is returned.
        """
        params = self.hpv.get_lnl_spread_params(as_flat=as_flat)

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_spread_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model's spread edges.

        This is consistent with how the :py:meth:`.set_params`
        """
        params = self.get_tumor_spread_params(as_flat=False)
        params.update(self.get_lnl_spread_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> types.ParamsType:
        """Return the parameters of the model.

        It returns the combination of the call to the :py:meth:`.Unilateral.get_params`
        of the HPV- and noHPV model. For the use of the ``as_dict`` and
        ``as_flat`` arguments, see the documentation of the
        :py:meth:`.types.Model.get_params` method.

        Also see the :py:meth:`.get_spread_params` method to understand how the
        symmetry settings affect the return value.
        """
        params = self.get_spread_params(as_flat=as_flat)
        params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = utils.flatten(params)

        return params if as_dict else params.values()

    def set_tumor_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from tumor to LNLs."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=["HPV", "noHPV"],
        )

        hpv_kwargs = global_kwargs.copy()
        hpv_kwargs.update(kwargs.get("HPV", {}))
        nohpv_kwargs = global_kwargs.copy()
        nohpv_kwargs.update(kwargs.get("noHPV", {}))

        args = self.hpv.set_tumor_spread_params(*args, **hpv_kwargs)
        args = self.nohpv.set_tumor_spread_params(*args, **nohpv_kwargs)
        utils.synchronize_params(  # might be redundant check later
            get_from=self.hpv.graph.lnl_edges,
            set_to=self.nohpv.graph.lnl_edges,
        )

        return args

    def set_lnl_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread from LNLs to tumor."""
        kwargs, global_kwargs = utils.unflatten_and_split(
            kwargs,
            expected_keys=["HPV", "noHPV"],
        )

        hpv_kwargs = global_kwargs.copy()
        hpv_kwargs.update(kwargs.get("HPV", {}))
        nohpv_kwargs = global_kwargs.copy()
        nohpv_kwargs.update(kwargs.get("noHPV", {}))

        args = self.hpv.set_lnl_spread_params(*args, **hpv_kwargs)
        return self.nohpv.set_lnl_spread_params(*args, **nohpv_kwargs)

    def set_spread_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set the parameters of the model's spread edges."""
        args = self.set_tumor_spread_params(*args, **kwargs)
        return self.set_lnl_spread_params(*args, **kwargs)

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Set new parameters to the model."""
        args = self.set_spread_params(*args, **kwargs)
        return self.set_distribution_params(*args, **kwargs)

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        side: str = "ipsi",
        mapping: callable | dict[int, Any] = utils.early_late_mapping,
    ) -> None:
        """Load patient data into the model.

        Amounts to calling the :py:meth:`~lymph.models.Unilateral.load_patient_data`
        method of both the HPV+ and the HPV- model.
        """
        # TODO: What about patients with unknown HPV status?
        is_hpv_pos = patient_data["patient", "#", "hpv_status"] == True  # noqa: E712
        is_hpv_neg = patient_data["patient", "#", "hpv_status"] == False  # noqa: E712

        hpv_patient_data = patient_data.loc[is_hpv_pos]
        nohpv_patient_data = patient_data.loc[is_hpv_neg]

        self.hpv.load_patient_data(
            patient_data=hpv_patient_data,
            side=side,
            mapping=mapping,
        )
        self.nohpv.load_patient_data(
            patient_data=nohpv_patient_data,
            side=side,
            mapping=mapping,
        )

    def _hmm_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the HMM likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        hpv_likelihood = self.hpv._hmm_likelihood(log=log, t_stage=t_stage)
        nohpv_likelihood = self.nohpv._hmm_likelihood(log=log, t_stage=t_stage)

        if log:
            llh += hpv_likelihood + nohpv_likelihood
        else:
            llh *= hpv_likelihood * nohpv_likelihood
        return llh

    def _bn_likelihood(self, log: bool = True, t_stage: str | None = None) -> float:
        """Compute the BN likelihood of data, using the stored params."""
        llh = 0.0 if log else 1.0

        hpv_likelihood = self.hpv._bn_likelihood(log=log, t_stage=t_stage)
        nohpv_likelihood = self.nohpv._bn_likelihood(log=log, t_stage=t_stage)

        if log:
            llh += hpv_likelihood + nohpv_likelihood
        else:
            llh *= hpv_likelihood * nohpv_likelihood
        return llh

    def likelihood(
        self,
        given_params: types.ParamsType | None = None,
        log: bool = True,
        t_stage: str | None = None,
        mode: Literal["HMM", "BN"] = "HMM",
    ):
        """Compute the (log-)likelihood of the stored data given the model (and params).

        See the documentation of :py:meth:`.types.Model.likelihood` for more
        information on how to use the ``given_params`` parameter.

        Returns the log-likelihood if ``log`` is set to ``True``. The ``mode`` parameter
        determines whether the likelihood is computed for the hidden Markov model
        (``"HMM"``) or the Bayesian network (``"BN"``).

        .. note::

            The computation is much faster if no parameters are given, since then the
            transition matrix does not need to be recomputed.

        .. seealso::

            :py:meth:`.Unilateral.likelihood`
                The corresponding unilateral function.

        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            utils.safe_set_params(self, given_params)
        except ValueError:
            return -np.inf if log else 0.0

        if mode == "HMM":
            return self._hmm_likelihood(log, t_stage)
        if mode == "BN":
            return self._bn_likelihood(log, t_stage)

        raise ValueError("Invalid mode. Must be either 'HMM' or 'BN'.")

    # The methods below must be implemented to satisfy the Model interface.
    # However, realistically one would only use either the `hpv` or the `nohpv` model
    # and not a combination of them. Therefore, they will just call the corresponding
    # method of the HPV model.

    @select_hpv_model
    def state_dist(self, model: models.Unilateral, *args, **kwargs) -> np.ndarray:
        """Compute the distribution over possible states.

        See :py:meth:`.models.Unilateral.state_dist` for more information.
        """
        return model.state_dist(*args, **kwargs)

    @select_hpv_model
    def posterior_state_dist(
        self,
        model: models.Unilateral,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Compute the posterior distribution over hidden states given a diagnosis.

        See :py:meth:`.models.Unilateral.posterior_state_dist` for more information.
        """
        return model.posterior_state_dist(*args, **kwargs)

    @select_hpv_model
    def marginalize(self, model: models.Unilateral, *args, **kwargs) -> np.ndarray:
        """Marginalize ``given_state_dist`` over matching ``involvement`` patterns.

        See :py:meth:`.models.Unilateral.marginalize` for more information.
        """
        return model.marginalize(*args, **kwargs)

    @select_hpv_model
    def risk(self, model: models.Unilateral, *args, **kwargs) -> np.ndarray:
        """Compute risk of a certain ``involvement``, using the ``given_diagnosis``.

        See :py:meth:`.models.Unilateral.risk` for more information.
        """
        return model.risk(*args, **kwargs)
