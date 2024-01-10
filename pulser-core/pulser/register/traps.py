# Copyright 2023 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines a set of traps from their coordinates."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pulser.register._coordinates import COORD_PRECISION, CoordsCollection


@dataclass(init=False, eq=False, frozen=True)
class Traps(ABC, CoordsCollection):
    """Defines a unique set of traps.

    The traps are always sorted under the same convention: ascending order
    along x, then along y, then along z (if applicable). Respecting this order,
    the traps are then numbered starting from 0.

    Args:
        trap_coordinates: The coordinates of each trap.
    """

    slug: str | None

    def __init__(self, trap_coordinates: ArrayLike, slug: str | None = None):
        """Initializes a RegisterLayout."""
        array_type_error_msg = ValueError(
            "'trap_coordinates' must be an array or list of coordinates."
        )

        try:
            coords_arr = np.array(trap_coordinates, dtype=float)
        except ValueError as e:
            raise array_type_error_msg from e

        shape = coords_arr.shape
        if len(shape) != 2:
            raise array_type_error_msg

        if shape[1] not in (2, 3):
            raise ValueError(
                f"Each coordinate must be of size 2 or 3, not {shape[1]}."
            )

        if len(np.unique(trap_coordinates, axis=0)) != shape[0]:
            raise ValueError(
                "All trap coordinates of a register layout must be unique."
            )
        object.__setattr__(self, "_coords", trap_coordinates)
        object.__setattr__(self, "slug", slug)

    @property
    def traps_dict(self) -> dict:
        """Mapping between trap IDs and coordinates."""
        return dict(enumerate(self.sorted_coords))

    @cached_property  # Acts as an attribute in a frozen dataclass
    def _coords_to_traps(self) -> dict[tuple[float, ...], int]:
        return {tuple(coord): id for id, coord in self.traps_dict.items()}

    @property
    def number_of_traps(self) -> int:
        """The number of traps in the layout."""
        return len(self._sorted_coords)

    def get_traps_from_coordinates(self, *coordinates: ArrayLike) -> list[int]:
        """Finds the trap ID for a given set of trap coordinates.

        Args:
            coordinates: The coordinates to return the trap IDs.

        Returns:
            The list of trap IDs corresponding to the coordinates.
        """
        traps = []
        rounded_coords = np.round(
            np.array(coordinates), decimals=COORD_PRECISION
        )
        for coord, rounded in zip(coordinates, rounded_coords):
            key = tuple(rounded)
            if key not in self._coords_to_traps:
                raise ValueError(
                    f"The coordinate '{coord!s}' is not a part of the "
                    "RegisterLayout."
                )
            traps.append(self._coords_to_traps[key])
        return traps

    @property
    @abstractmethod
    def _hash_object(self) -> hashlib._Hash:
        return super()._hash_object

    def static_hash(self) -> str:
        """Returns the idempotent hash.

        Python's standard hash is not idempotent as it changes between
        sessions. This hash can be used when an idempotent hash is
        required.

        Returns:
            str: An hexstring encoding the hash.

        Note:
            This hash will be returned as an hexstring without
            the '0x' prefix (unlike what is returned by 'hex()').
        """
        return self._safe_hash().hex()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Traps):
            return False
        return self._safe_hash() == other._safe_hash()

    def __str__(self) -> str:
        return self.slug or self.__repr__()
