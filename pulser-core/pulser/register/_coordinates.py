"""Defines a collection of coordinates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np

import pulser.math as pm

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
    def sorted_coords(self) -> np.ndarray:
        """The sorted coordinates."""
        # Copies to prevent direct access to self._sorted_coords
        return self._sorted_coords.as_array(detach=True).copy()

    @cached_property
    def _coords_arr(self) -> pm.AbstractArray:
        return pm.vstack(cast(Sequence, self._coords))

    @cached_property
    def _rounded_coords(self) -> pm.AbstractArray:
        return pm.round(self._coords_arr, decimals=COORD_PRECISION)

    @cached_property  # Acts as an attribute in a frozen dataclass
    def _sorted_coords(self) -> pm.AbstractArray:
        sorting = self._calc_sorting_order()
        return self._rounded_coords[sorting]

    def _calc_sorting_order(self) -> np.ndarray:
        """Calculates the unique order that sorts the coordinates."""
        # Sorting the coordinates 1st left to right, 2nd bottom to top
        dims = self._rounded_coords.shape[1]
        arr = self._rounded_coords.as_array(detach=True)
        sorter = [arr[:, i] for i in range(dims - 1, -1, -1)]
        sorting = np.lexsort(tuple(sorter))
        return cast(np.ndarray, sorting)

    @property
    def _hash_object(self) -> hashlib._Hash:
        # Include dimensionality because the array is flattened with tobytes()
        hash_ = hashlib.sha256(bytes(self.dimensionality))
        hash_.update(self.sorted_coords.tobytes())
        return hash_

    def _safe_hash(self) -> bytes:
        return self._hash_object.digest()
