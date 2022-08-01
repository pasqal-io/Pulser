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
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as abcSequence
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import ArrayLike

from pulser.json.utils import obj_to_dict

if TYPE_CHECKING:  # pragma: no cover
    from pulser.register.register_layout import RegisterLayout

T = TypeVar("T", bound="BaseRegister")
QubitId = Union[int, str]


class _LayoutInfo(NamedTuple):
    """Auxiliary class to store the register layout information."""

    layout: RegisterLayout
    trap_ids: tuple[int, ...]


class BaseRegister(ABC):
    """The abstract class for a register."""

    @abstractmethod
    def __init__(self, qubits: Mapping[Any, ArrayLike], **kwargs: Any):
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
        self._ids: tuple[QubitId, ...] = tuple(qubits.keys())
        self._coords = [np.array(v, dtype=float) for v in qubits.values()]
        self._dim = self._coords[0].size
        self._layout_info: Optional[_LayoutInfo] = None
        self._init_kwargs(**kwargs)

    def _init_kwargs(self, **kwargs: Any) -> None:
        if kwargs:
            if kwargs.keys() != {"layout", "trap_ids"}:
                raise ValueError(
                    "If specifying 'kwargs', they must only be 'layout' and "
                    "'trap_ids'."
                )
            layout: RegisterLayout = kwargs["layout"]
            trap_ids: tuple[int, ...] = tuple(kwargs["trap_ids"])
            self._validate_layout(layout, trap_ids)
            self._layout_info = _LayoutInfo(layout, trap_ids)

    @property
    def qubits(self) -> dict[QubitId, np.ndarray]:
        """Dictionary of the qubit names and their position coordinates."""
        return dict(zip(self._ids, self._coords))

    @property
    def qubit_ids(self) -> tuple[QubitId, ...]:
        """The qubit IDs of this register."""
        return self._ids

    @property
    def layout(self) -> Optional[RegisterLayout]:
        """The layout used to define the register."""
        return self._layout_info.layout if self._layout_info else None

    def find_indices(self, id_list: abcSequence[QubitId]) -> list[int]:
        """Computes indices of qubits.

        This can especially be useful when building a Pulser Sequence
        with a parameter denoting qubits.

        Example:
            Let ``reg`` be a register with qubit Ids "a", "b" and "c":

            >>> reg.find_indices(["a", "b", "c", "a"])

            It returns ``[0, 1, 2, 0]``, following the qubit order of the
            register.

            Then, it is possible to use these indices when building a
            sequence, typically by assigning them to an array of variables
            that can be provided as an argument to ``target_index``
            and ``phase_shift_index``.

        Args:
            id_list: IDs of the qubits to find.

        Returns:
            Indices of the qubits to denote, only valid for the
            given mapping.
        """
        if not set(id_list) <= set(self.qubit_ids):
            raise ValueError(
                "The IDs list must be selected among the IDs of the register's"
                " qubits."
            )
        return [self.qubit_ids.index(id_) for id_ in id_list]

    @classmethod
    def from_coordinates(
        cls: Type[T],
        coords: np.ndarray,
        center: bool = True,
        prefix: Optional[str] = None,
        labels: Optional[abcSequence[QubitId]] = None,
        **kwargs: Any,
    ) -> T:
        """Creates the register from an array of coordinates.

        Args:
            coords: The coordinates of each qubit to include in the
                register.

        Args:
            center: Whether or not to center the entire array around the
                origin.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
            labels: The list of qubit ids. If defined, each qubit id will be
                set to the corresponding value.

        Returns:
            A register with qubits placed on the given coordinates.
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
        return cls(qubits, **kwargs)

    def _validate_layout(
        self, register_layout: RegisterLayout, trap_ids: tuple[int, ...]
    ) -> None:
        """Sets the RegisterLayout that originated this register."""
        trap_coords = register_layout.coords
        if register_layout.dimensionality != self._dim:
            raise ValueError(
                "The RegisterLayout dimensionality is not the same as this "
                "register's."
            )
        if len(set(trap_ids)) != len(trap_ids):
            raise ValueError("Every 'trap_id' must be a unique integer.")

        if len(trap_ids) != len(self._ids):
            raise ValueError(
                "The amount of 'trap_ids' must be equal to the number of atoms"
                " in the register."
            )

        for reg_coord, trap_id in zip(self._coords, trap_ids):
            if np.any(reg_coord != trap_coords[trap_id]):
                raise ValueError(
                    "The chosen traps from the RegisterLayout don't match this"
                    " register's coordinates."
                )

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """Serializes the object.

        During deserialization, it will be reconstructed using
        'from_coordinates', so that it uses lists instead of a dictionary
        (in JSON, lists elements keep their types, but dictionaries keys do
        not).
        """
        cls_dict = obj_to_dict(
            None,
            _build=False,
            _name=self.__class__.__name__,
            _module=self.__class__.__module__,
        )

        kwargs = (
            {} if self._layout_info is None else self._layout_info._asdict()
        )

        return obj_to_dict(
            self,
            cls_dict,
            [np.ndarray.tolist(qubit_coords) for qubit_coords in self._coords],
            False,
            None,
            self._ids,
            **kwargs,
            _submodule=self.__class__.__name__,
            _name="from_coordinates",
        )

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return False

        return list(self._ids) == list(other._ids) and all(
            (
                np.allclose(  # Accounts for rounding errors
                    self._coords[i],
                    other._coords[other._ids.index(id)],
                )
                for i, id in enumerate(self._ids)
            )
        )
