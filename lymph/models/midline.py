from __future__ import annotations

import warnings
from typing import Any

from lymph.helper import DelegatorMixin
from lymph.models.bilateral import Bilateral


class Midline(DelegatorMixin):
    """Model bilateral lymphatic spread with varying tumor lateralization.

    It makes use of multiple instances of the :py:class:~lymph.models.Bilateral` class,
    using one for each different tumor lateralization.

    See Also:
        :py:class:~lymph.models.Bilateral`
    """
    def __init__(
        self,
        graph_dict: dict[tuple[str], list[str]],
        is_symmetric: dict[str, bool] | None = None,
        tumor_lateralizations: list[str] | None = None,
        unilateral_kwargs: dict[str, Any] | None = None,
        ipsilateral_kwargs: dict[str, Any] | None = None,
        contralateral_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize one bilateral model for each tumor lateralization of interest."""
        super().__init__()

        self.models = {}
        for lateralization in tumor_lateralizations:
            if lateralization not in ["lateral", "midline_extension", "central"]:
                warnings.warn(
                    f"Unrecognized tumor lateralization '{lateralization}'. "
                    "Only 'lateral', 'midline_extension', and 'central' are supported. "
                    "Skipping this lateralization."
                )
                continue

            self.models[lateralization] = Bilateral(
                graph_dict,
                is_symmetric=is_symmetric,
                unilateral_kwargs=unilateral_kwargs,
                ipsilateral_kwargs=ipsilateral_kwargs,
                contralateral_kwargs=contralateral_kwargs,
            )
