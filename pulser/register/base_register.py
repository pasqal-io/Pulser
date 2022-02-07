# Copyright 2021 Pulser Development Team
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
"""Defines the abstract register class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterable
from collections.abc import Sequence as abcSequence
from typing import Any, cast, Optional, Union, TypeVar, Type

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from pulser.json.utils import obj_to_dict

T = TypeVar("T", bound="BaseRegister")
QubitId = Union[int, str]


class BaseRegister(ABC):
    """The abstract class for a register."""

    @abstractmethod
    def __init__(self, qubits: Mapping[Any, ArrayLike]):
        """Initializes a custom Register."""
        if not isinstance(qubits, dict):
            raise TypeError(
                "The qubits have to be stored in a dictionary "
                "matching qubit ids to position coordinates."
            )
        if not qubits:
            raise ValueError(
                "Cannot create a Register with an empty qubit " "dictionary."
            )
        self._ids = list(qubits.keys())
        self._coords = [np.array(v, dtype=float) for v in qubits.values()]
        self._dim = 0

    @property
    def qubits(self) -> dict[QubitId, np.ndarray]:
        """Dictionary of the qubit names and their position coordinates."""
        return dict(zip(self._ids, self._coords))

    @classmethod
    def from_coordinates(
        cls: Type[T],
        coords: np.ndarray,
        center: bool = True,
        prefix: Optional[str] = None,
        labels: Optional[abcSequence[QubitId]] = None,
    ) -> T:
        """Creates the register from an array of coordinates.

        Args:
            coords (ndarray): The coordinates of each qubit to include in the
                register.

        Keyword args:
            center(defaut=True): Whether or not to center the entire array
                around the origin.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
            labels (ArrayLike): The list of qubit ids. If defined, each qubit
                id will be set to the corresponding value.

        Returns:
            Register: A register with qubits placed on the given coordinates.
        """
        if center:
            coords = coords - np.mean(coords, axis=0)  # Centers the array
        if prefix is not None:
            pre = str(prefix)
            qubits = {pre + str(i): pos for i, pos in enumerate(coords)}
            if labels is not None:
                raise NotImplementedError(
                    "It is impossible to specify a prefix and "
                    "a set of labels at the same time"
                )

        elif labels is not None:
            if len(coords) != len(labels):
                raise ValueError(
                    f"Label length ({len(labels)}) does not"
                    f"match number of coordinates ({len(coords)})"
                )
            qubits = dict(zip(cast(Iterable, labels), coords))
        else:
            qubits = dict(cast(Iterable, enumerate(coords)))
        return cls(qubits)

    ) -> None:
                )

    def _draw_checks(
        self,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
    ) -> None:
        """Checks common in all register drawings.

        Keyword Args:
            blockade_radius(float, default=None): The distance (in μm) between
                atoms below the Rydberg blockade effect occurs.
            draw_half_radius(bool, default=False): Whether or not to draw the
                half the blockade radius surrounding each atoms. If `True`,
                requires `blockade_radius` to be defined.
            draw_graph(bool, default=True): Whether or not to draw the
                interaction between atoms as edges in a graph. Will only draw
                if the `blockade_radius` is defined.
        """
        # Check spacing
        if blockade_radius is not None and blockade_radius <= 0.0:
            raise ValueError(
                "Blockade radius (`blockade_radius` ="
                f" {blockade_radius})"
                " must be greater than 0."
            )

        if draw_half_radius:
            if blockade_radius is None:
                raise ValueError("Define 'blockade_radius' to draw.")
            if len(self._ids) == 1:
                raise NotImplementedError(
                    "Needs more than one atom to draw the blockade radius."
                )

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        qs = dict(zip(self._ids, map(np.ndarray.tolist, self._coords)))
        return obj_to_dict(self, qs)

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return False

        return set(self._ids) == set(other._ids) and all(
            (
                np.array_equal(
                    self._coords[i],
                    other._coords[other._ids.index(id)],
                )
                for i, id in enumerate(self._ids)
            )
        )
