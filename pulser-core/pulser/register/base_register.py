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

import json
import warnings
from abc import ABC, abstractmethod
from collections import Counter
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

import pulser.math as pm
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.json.utils import obj_to_dict
from pulser.register._coordinates import CoordsCollection
from pulser.register.weight_maps import DetuningMap

if TYPE_CHECKING:
    from pulser.register.register_layout import RegisterLayout

T = TypeVar("T", bound="BaseRegister")
QubitId = str


class _LayoutInfo(NamedTuple):
    """Auxiliary class to store the register layout information."""

    layout: RegisterLayout
    trap_ids: tuple[int, ...]


class BaseRegister(ABC, CoordsCollection):
    """The abstract class for a register."""

    @abstractmethod
    def __init__(
        self,
        qubits: Mapping[str, ArrayLike] | Mapping[int, ArrayLike],
        **kwargs: Any,
    ):
        """Initializes a custom Register."""
        if not isinstance(qubits, dict):
            raise TypeError(
                "The qubits have to be stored in a dictionary "
                "matching qubit ids to position coordinates; "
                f"got {type(qubits)}."
            )
        if not qubits:
            raise ValueError(
                "Cannot create a Register with an empty qubit " "dictionary."
            )
        super().__init__(
            [pm.AbstractArray(v, dtype=float) for v in qubits.values()]
        )
        self._ids: tuple[QubitId, ...] = tuple(qubits.keys())
        if any(not isinstance(id, str) for id in self._ids):
            with warnings.catch_warnings():
                warnings.filterwarnings("once")
                warnings.warn(
                    "Usage of `int`s or any non-`str`types as `QubitId`s will "
                    "be deprecated. Define your `QubitId`s as `str`s, prefer "
                    "setting `prefix='q'` when using classmethods, as that "
                    "will become the new default once `int` qubit IDs become "
                    "invalid.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self._layout_info: Optional[_LayoutInfo] = None
        self._init_kwargs(**kwargs)

    def _init_kwargs(self, **kwargs: Any) -> None:
        if kwargs:
            if kwargs.keys() != {"layout", "trap_ids"}:
                raise ValueError(
                    "If specifying 'kwargs', they must only be 'layout' and "
                    f"'trap_ids'; got {list(kwargs)}."
                )
            layout: RegisterLayout = kwargs["layout"]
            trap_ids: tuple[int, ...] = tuple(kwargs["trap_ids"])
            self._validate_layout(layout, trap_ids)
            self._layout_info = _LayoutInfo(layout, trap_ids)

    @property
    def qubits(self) -> dict[QubitId, pm.AbstractArray]:
        """Dictionary of the qubit names and their position coordinates."""
        return dict(zip(self._ids, self._coords_arr))

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
        valid_ids = set(self.qubit_ids)
        if not set(id_list) <= valid_ids:
            invalid = [id_ for id_ in id_list if id_ not in valid_ids]
            raise ValueError(
                "The IDs list must be selected among the IDs of the register's"
                f" qubits; {invalid} not in {list(self.qubit_ids)}."
            )
        return [self.qubit_ids.index(id_) for id_ in id_list]

    @classmethod
    def from_coordinates(
        cls: Type[T],
        coords: ArrayLike | pm.TensorLike,
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
        coords_ = pm.vstack(cast(abcSequence, coords)).astype(float)
        if center:
            coords_ = coords_ - pm.mean(coords_, axis=0)  # Centers the array
        qubits: dict[str, pm.AbstractArray]
        if prefix is not None:
            pre = str(prefix)
            qubits = {pre + str(i): pos for i, pos in enumerate(coords_)}
            if labels is not None:
                raise NotImplementedError(
                    "It is impossible to specify a prefix and "
                    "a set of labels at the same time; "
                    f"got prefix={prefix!r} and labels={list(labels)}."
                )

        elif labels is not None:
            if len(coords_) != len(labels):
                raise ValueError(
                    f"Label length ({len(labels)}) does not "
                    f"match number of coordinates ({len(coords_)}); "
                    f"got coords {coords!r} and labels {list(labels)}."
                )
            qubits = dict(zip(cast(Iterable, labels), coords_))
        else:
            qubits = dict(cast(Iterable, enumerate(coords_)))
        return cls(qubits, **kwargs)

    def _validate_layout(
        self, register_layout: RegisterLayout, trap_ids: tuple[int, ...]
    ) -> None:
        """Sets the RegisterLayout that originated this register."""
        trap_coords = register_layout.coords
        if register_layout.dimensionality != self.dimensionality:
            raise ValueError(
                "The RegisterLayout dimensionality "
                f"({register_layout.dimensionality}D) is not the same as this "
                f"register's ({self.dimensionality}D); Layout's coordinates "
                f"are {register_layout.coords} and register's "
                f"{self.sorted_coords}."
            )
        if len(set(trap_ids)) != len(trap_ids):
            repeated = [t for t, freq in Counter(trap_ids).items() if freq > 1]
            raise ValueError(
                "Every 'trap_id' must be a unique integer; "
                f"found repeated ids {repeated} in {list(trap_ids)}."
            )

        if len(trap_ids) != len(self._ids):
            raise ValueError(
                f"The amount of 'trap_ids' {len(trap_ids)} is not equal to "
                f"the number of atoms {len(self._ids)} in the register. Got "
                f"trap ids {trap_ids} for atoms {self._ids}."
            )
        if not set(trap_ids).issubset(register_layout.traps_dict):
            raise ValueError(
                "All 'trap_ids' must correspond to the ID of a trap in the "
                f"layout; got {trap_ids}, must be among "
                f"{list(register_layout.traps_dict)}."
            )
        for reg_coord, trap_id in zip(
            self._coords_arr.as_array(detach=True), trap_ids
        ):
            if np.any(reg_coord != trap_coords[trap_id]):
                raise ValueError(
                    "The chosen traps from the RegisterLayout don't match this"
                    f" register's coordinates; trap {trap_id} is at "
                    f"{trap_coords[trap_id].tolist()} but the register "
                    f"has {reg_coord.tolist()}."
                )

    def define_detuning_map(
        self,
        detuning_weights: Mapping[QubitId, float],
        slug: str | None = None,
    ) -> DetuningMap:
        """Defines a DetuningMap for some qubits of the register.

        Args:
            detuning_weights: A mapping between the IDs of the targeted qubits
                and detuning weights (between 0 and 1).
            slug: An optional identifier for the detuning map.

        Returns:
            A DetuningMap associating detuning weights to the trap coordinates
            of the targeted qubits.
        """
        valid_ids = set(self.qubit_ids)
        if not set(detuning_weights.keys()) <= valid_ids:
            invalid = [id_ for id_ in detuning_weights if id_ not in valid_ids]
            raise ValueError(
                "The qubit ids linked to detuning weights have to be defined"
                f" in the register. Got {invalid}, which are not in "
                f"{list(self.qubit_ids)}."
            )
        return DetuningMap(
            pm.vstack(
                [self.qubits[qubit_id] for qubit_id in detuning_weights]
            ),
            list(detuning_weights.values()),
            slug,
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
            [qubit_coords.tolist() for qubit_coords in self._coords_arr],
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

        return self._ids == other._ids and np.allclose(
            self._coords_arr.as_array(detach=True),
            other._coords_arr.as_array(detach=True),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.qubits})"

    def coords_hex_hash(self) -> str:
        """Returns the idempotent hash of the coordinates.

        Returns:
            str: An hexstring encoding the hash.

        Note:
            This hash will be returned as an hexstring without
            the '0x' prefix (unlike what is returned by 'hex()').
        """
        return self._safe_hash().hex()

    @abstractmethod
    def _to_abstract_repr(self) -> list[dict[str, Union[QubitId, float]]]:
        pass

    def to_abstract_repr(self) -> str:
        """Serializes the register into an abstract JSON object."""
        abstr_reg: dict[str, Any] = dict(register=self._to_abstract_repr())
        if self.layout is not None:
            abstr_reg["layout"] = self.layout
        abstr_reg_str = json.dumps(abstr_reg, cls=AbstractReprEncoder)
        validate_abstract_repr(abstr_reg_str, "register")
        return abstr_reg_str
