"""Defines a collection of coordinates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np

import pulser.math as pm
from pulser.math.abstract_array import AbstractArrayLike

COORD_PRECISION = 6


@dataclass(eq=False, frozen=True)
class CoordsCollection:
    """Defines a unique collection of coordinates.

    The coordinates are always sorted under the same convention: ascending
    order along x, then along y, then along z (if applicable). Respecting
    this order, the traps are then numbered starting from 0.

    Args:
        _coords: The coordinates.
    """

    _coords: pm.AbstractArray | list

    @property
    def dimensionality(self) -> int:
        """The dimensionality of the coordinates (2 or 3)."""
        return self._sorted_coords.shape[1]

    @property
    def sorted_coords(self) -> pm.AbstractArray:
        """The sorted coordinates."""
        # Copies to prevent direct access to self._sorted_coords
        return self._sorted_coords.copy()

    @cached_property  # Acts as an attribute in a frozen dataclass
    def _sorted_coords(self) -> pm.AbstractArray:
        coords = pm.vstack(cast(Sequence[AbstractArrayLike], self._coords))
        rounded_coords = pm.round(coords, decimals=COORD_PRECISION)
        sorting = self._calc_sorting_order(rounded_coords)
        return rounded_coords[sorting]

    def _calc_sorting_order(
        self, rounded_coords: pm.AbstractArray
    ) -> np.ndarray:
        """Calculates the unique order that sorts the coordinates."""
        # Sorting the coordinates 1st left to right, 2nd bottom to top
        dims = rounded_coords.shape[1]
        sorter = [
            rounded_coords.as_array(detach=True)[:, i]
            for i in range(dims - 1, -1, -1)
        ]
        sorting = np.lexsort(tuple(sorter))
        return cast(np.ndarray, sorting)

    @property
    def _hash_object(self) -> hashlib._Hash:
        # Include dimensionality because the array is flattened with tobytes()
        hash_ = hashlib.sha256(bytes(self.dimensionality))
        hash_.update(self.sorted_coords.as_array(detach=True).tobytes())
        return hash_

    def _safe_hash(self) -> bytes:
        return self._hash_object.digest()
