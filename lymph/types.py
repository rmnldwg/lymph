"""
Type aliases and protocols used in the lymph package.
"""
from typing import Protocol

from pandas._libs.missing import NAType


class HasSetParams(Protocol):
    """Protocol for classes that have a ``set_params`` method."""
    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        ...


class HasGetParams(Protocol):
    """Protocol for classes that have a ``get_params`` method."""
    def get_params(
        self,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> tuple[float] | dict[str, float]:
        ...


PatternType = dict[str, bool | NAType | None]
"""Type alias for an involvement pattern."""

DiagnoseType = dict[str, PatternType]
"""Type alias for a diagnose, which is a involvement pattern per diagnostic modality."""