# Copyright 2020 Pulser Development Team
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

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from sys import version_info
from typing import Any, Optional, cast
from warnings import warn

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pulser.channels.base_channel import Channel
from pulser.devices.interaction_coefficients import c6_dict
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.utils import obj_to_dict
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register_layout import COORD_PRECISION, RegisterLayout

if version_info[:2] >= (3, 8):  # pragma: no cover
    from typing import Literal, get_args
else:  # pragma: no cover
    try:
        from typing_extensions import Literal, get_args  # type: ignore
    except ImportError:
        raise ImportError(
            "Using pulser with Python version 3.7 requires the"
            " `typing_extensions` module. Install it by running"
            " `pip install typing-extensions`."
        )

DIMENSIONS = Literal[2, 3]


@dataclass(frozen=True, repr=False)  # type: ignore[misc]
class BaseDevice(ABC):
    r"""Base class of a neutral-atom device.

    Attributes:
        name: The name of the device.
        dimensions: Whether it supports 2D or 3D arrays.
        rybderg_level : The value of the principal quantum number :math:`n`
            when the Rydberg level used is of the form
            :math:`|nS_{1/2}, m_j = +1/2\rangle`.
        max_atom_num: Maximum number of atoms supported in an array.
        max_radial_distance: The furthest away an atom can be from the center
            of the array (in μm).
        min_atom_distance: The closest together two atoms can be (in μm).
        interaction_coeff_xy: :math:`C_3/\hbar` (in :math:`\mu m^3 / \mu s`),
            which sets the van der Waals interaction strength between atoms in
            different Rydberg states. Needed only if there is a Microwave
            channel in the device. If unsure, 3700.0 is a good default value.
        supports_slm_mask: Whether the device supports the SLM mask feature.
        max_layout_filling: The largest fraction of a layout that can be filled
            with atoms.
    """
    name: str
    dimensions: DIMENSIONS
    rydberg_level: int
    _channels: tuple[tuple[str, Channel], ...]
    min_atom_distance: float
    max_atom_num: Optional[int]
    max_radial_distance: Optional[int]
    interaction_coeff_xy: Optional[float] = None
    supports_slm_mask: bool = False
    max_layout_filling: float = 0.5
    reusable_channels: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        def type_check(
            param: str, type_: type, value_override: Any = None
        ) -> None:
            value = (
                getattr(self, param)
                if value_override is None
                else value_override
            )
            if not isinstance(value, type_):
                raise TypeError(
                    f"{param} must be of type '{type_.__name__}', "
                    f"not '{type(value).__name__}'."
                )

        type_check("name", str)
        if self.dimensions not in get_args(DIMENSIONS):
            raise ValueError(
                f"'dimensions' must be one of {get_args(DIMENSIONS)}, "
                f"not {self.dimensions}."
            )
        self._validate_rydberg_level(self.rydberg_level)
        for ch_id, ch_obj in self._channels:
            type_check("All channel IDs", str, value_override=ch_id)
            type_check("All channels", Channel, value_override=ch_obj)

        for param in (
            "min_atom_distance",
            "max_atom_num",
            "max_radial_distance",
        ):
            value = getattr(self, param)
            if param in self._optional_parameters:
                prelude = "When defined, "
                is_none = value is None
            elif value is None:
                raise TypeError(
                    f"'{param}' can't be None in a '{type(self).__name__}' "
                    "instance."
                )
            else:
                prelude = ""
                is_none = False

            if param == "min_atom_distance":
                comp = "greater than or equal to zero"
                valid = is_none or value >= 0
            else:
                if not is_none:
                    type_check(param, int)
                comp = "greater than zero"
                valid = is_none or value > 0
            msg = prelude + f"'{param}' must be {comp}, not {value}."
            if not valid:
                raise ValueError(msg)

        if any(
            ch.basis == "XY" for _, ch in self._channels
        ) and not isinstance(self.interaction_coeff_xy, float):
            raise TypeError(
                "When the device has a 'Microwave' channel, "
                "'interaction_coeff_xy' must be a 'float',"
                f" not '{type(self.interaction_coeff_xy)}'."
            )
        type_check("supports_slm_mask", bool)
        type_check("reusable_channels", bool)

        if not (0.0 < self.max_layout_filling <= 1.0):
            raise ValueError(
                "The maximum layout filling fraction must be "
                "greater than 0. and less than or equal to 1., "
                f"not {self.max_layout_filling}."
            )

        def to_tuple(obj: tuple | list) -> tuple:
            if isinstance(obj, (tuple, list)):
                obj = tuple(to_tuple(el) for el in obj)
            return obj

        # Turns mutable lists into immutable tuples
        object.__setattr__(self, "_channels", to_tuple(self._channels))

    @property
    @abstractmethod
    def _optional_parameters(self) -> tuple[str, ...]:
        pass

    @property
    def channels(self) -> dict[str, Channel]:
        """Dictionary of available channels on this device."""
        return dict(self._channels)

    @property
    def supported_bases(self) -> set[str]:
        """Available electronic transitions for control and measurement."""
        return {ch.basis for ch in self.channels.values()}

    @property
    def interaction_coeff(self) -> float:
        r""":math:`C_6/\hbar` coefficient of chosen Rydberg level."""
        return float(c6_dict[self.rydberg_level])

    def __repr__(self) -> str:
        return self.name

    def rydberg_blockade_radius(self, rabi_frequency: float) -> float:
        """Calculates the Rydberg blockade radius for a given Rabi frequency.

        Args:
            rabi_frequency: The Rabi frequency, in rad/µs.

        Returns:
            The rydberg blockade radius, in μm.
        """
        return (self.interaction_coeff / rabi_frequency) ** (1 / 6)

    def rabi_from_blockade(self, blockade_radius: float) -> float:
        """The maximum Rabi frequency value to enforce a given blockade radius.

        Args:
            blockade_radius: The Rydberg blockade radius, in µm.

        Returns:
            The maximum rabi frequency value, in rad/µs.
        """
        return self.interaction_coeff / blockade_radius**6

    def validate_register(self, register: BaseRegister) -> None:
        """Checks if 'register' is compatible with this device.

        Args:
            register: The Register to validate.
        """
        if not isinstance(register, BaseRegister):
            raise TypeError(
                "'register' must be a pulser.Register or "
                "a pulser.Register3D instance."
            )

        if register._dim > self.dimensions:
            raise ValueError(
                f"All qubit positions must be at most {self.dimensions}D "
                "vectors."
            )
        self._validate_coords(register.qubits, kind="atoms")

        if register.layout is not None:
            try:
                self.validate_layout(register.layout)
            except (ValueError, TypeError):
                raise ValueError(
                    "The 'register' is associated with an incompatible "
                    "register layout."
                )
            self.validate_layout_filling(register)

    def validate_layout(self, layout: RegisterLayout) -> None:
        """Checks if a register layout is compatible with this device.

        Args:
            layout: The RegisterLayout to validate.
        """
        if not isinstance(layout, RegisterLayout):
            raise TypeError("'layout' must be a RegisterLayout instance.")

        if layout.dimensionality > self.dimensions:
            raise ValueError(
                "The device supports register layouts of at most "
                f"{self.dimensions} dimensions."
            )

        self._validate_coords(layout.traps_dict, kind="traps")

    def validate_layout_filling(
        self, register: BaseRegister | MappableRegister
    ) -> None:
        """Checks if a register properly fills its layout.

        Args:
            register: The register to validate. Must be created from a register
                layout.
        """
        if register.layout is None:
            raise TypeError(
                "'validate_layout_filling' can only be called for"
                " registers with a register layout."
            )
        n_qubits = len(register.qubit_ids)
        max_qubits = int(
            register.layout.number_of_traps * self.max_layout_filling
        )
        if n_qubits > max_qubits:
            raise ValueError(
                "Given the number of traps in the layout and the "
                "device's maximum layout filling fraction, the given"
                f" register has too many qubits ({n_qubits}). "
                "On this device, this layout can hold at most "
                f"{max_qubits} qubits."
            )

    def _validate_atom_number(self, coords: list[np.ndarray]) -> None:
        max_atom_num = cast(int, self.max_atom_num)
        if len(coords) > max_atom_num:
            raise ValueError(
                f"The number of atoms ({len(coords)})"
                " must be less than or equal to the maximum"
                f" number of atoms supported by this device"
                f" ({max_atom_num})."
            )

    def _validate_atom_distance(
        self, ids: list[QubitId], coords: list[np.ndarray], kind: str
    ) -> None:
        def invalid_dists(dists: np.ndarray) -> np.ndarray:
            cond1 = dists - self.min_atom_distance < -(
                10 ** (-COORD_PRECISION)
            )
            # Ensures there are no identical traps when
            # min_atom_distance = 0
            cond2 = dists < 10 ** (-COORD_PRECISION)
            return cast(np.ndarray, np.logical_or(cond1, cond2))

        if len(coords) > 1:
            distances = pdist(coords)  # Pairwise distance between atoms
            if np.any(invalid_dists(distances)):
                sq_dists = squareform(distances)
                mask = np.triu(np.ones(len(coords), dtype=bool), k=1)
                bad_pairs = np.argwhere(
                    np.logical_and(invalid_dists(sq_dists), mask)
                )
                bad_qbt_pairs = [(ids[i], ids[j]) for i, j in bad_pairs]
                raise ValueError(
                    f"The minimal distance between {kind} in this device "
                    f"({self.min_atom_distance} µm) is not respected "
                    f"(up to a precision of 1e{-COORD_PRECISION} µm) "
                    f"for the pairs: {bad_qbt_pairs}"
                )

    def _validate_radial_distance(
        self, ids: list[QubitId], coords: list[np.ndarray], kind: str
    ) -> None:
        too_far = np.linalg.norm(coords, axis=1) > self.max_radial_distance
        if np.any(too_far):
            raise ValueError(
                f"All {kind} must be at most {self.max_radial_distance} μm "
                f"away from the center of the array, which is not the case "
                f"for: {[ids[int(i)] for i in np.where(too_far)[0]]}"
            )

    def _validate_rydberg_level(self, ryd_lvl: int) -> None:
        if not isinstance(ryd_lvl, int):
            raise TypeError("Rydberg level has to be an int.")
        if not 49 < ryd_lvl < 101:
            raise ValueError("Rydberg level should be between 50 and 100.")

    def _params(self) -> dict[str, Any]:
        # This is used instead of dataclasses.asdict() because asdict()
        # is recursive and we have Channel dataclasses in the args that
        # we don't want to convert to dict
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _validate_coords(
        self, coords_dict: dict[QubitId, np.ndarray], kind: str = "atoms"
    ) -> None:
        ids = list(coords_dict.keys())
        coords = list(coords_dict.values())
        if kind == "atoms" and not (
            "max_atom_num" in self._optional_parameters
            and self.max_atom_num is None
        ):
            self._validate_atom_number(coords)
        self._validate_atom_distance(ids, coords, kind)
        if not (
            "max_radial_distance" in self._optional_parameters
            and self.max_radial_distance is None
        ):
            self._validate_radial_distance(ids, coords, kind)

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _to_abstract_repr(self) -> dict[str, Any]:
        params = self._params()
        ch_list = []
        for ch_name, ch_obj in params.pop("_channels"):
            ch_list.append(ch_obj._to_abstract_repr(ch_name))

        return {"version": "1", "channels": ch_list, **params}

    def to_abstract_repr(self) -> str:
        """Serializes the Sequence into an abstract JSON object."""
        return json.dumps(self, cls=AbstractReprEncoder)


@dataclass(frozen=True, repr=False)
class Device(BaseDevice):
    r"""Specifications of a neutral-atom device.

    A Device instance is immutable and must have all of its parameters defined.
    For usage in emulations, it can be converted to a VirtualDevice through the
    `Device.to_virtual()` method.

    Attributes:
        name: The name of the device.
        dimensions: Whether it supports 2D or 3D arrays.
        rybderg_level : The value of the principal quantum number :math:`n`
            when the Rydberg level used is of the form
            :math:`|nS_{1/2}, m_j = +1/2\rangle`.
        max_atom_num: Maximum number of atoms supported in an array.
        max_radial_distance: The furthest away an atom can be from the center
            of the array (in μm).
        min_atom_distance: The closest together two atoms can be (in μm).
        interaction_coeff_xy: :math:`C_3/\hbar` (in :math:`\mu m^3 / \mu s`),
            which sets the van der Waals interaction strength between atoms in
            different Rydberg states.
        supports_slm_mask: Whether the device supports the SLM mask feature.
        max_layout_filling: The largest fraction of a layout that can be filled
            with atoms.
        pre_calibrated_layouts: RegisterLayout instances that are already
            available on the Device.
    """
    max_atom_num: int
    max_radial_distance: int
    pre_calibrated_layouts: tuple[RegisterLayout, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        for ch_id, ch_obj in self._channels:
            if ch_obj.is_virtual():
                _sep = "', '"
                raise ValueError(
                    "A 'Device' instance cannot contain virtual channels."
                    f" For channel '{ch_id}', please define: "
                    f"'{_sep.join(ch_obj._undefined_fields())}'"
                )
        for layout in self.pre_calibrated_layouts:
            self.validate_layout(layout)
        # Hack to override the docstring of an instance
        object.__setattr__(self, "__doc__", self._specs(for_docs=True))

    @property
    def _optional_parameters(self) -> tuple[str, ...]:
        return ()

    @property
    def calibrated_register_layouts(self) -> dict[str, RegisterLayout]:
        """Register layouts already calibrated on this device."""
        return {str(layout): layout for layout in self.pre_calibrated_layouts}

    def change_rydberg_level(self, ryd_lvl: int) -> None:
        """Changes the Rydberg level used in the Device.

        Args:
            ryd_lvl: the Rydberg level to use (between 50 and 100).

        Note:
            Deprecated in version 0.8.0. Convert the device to a VirtualDevice
            with 'Device.to_virtual()' and use
            'VirtualDevice.change_rydberg_level()' instead.
        """
        warn(
            "'Device.change_rydberg_level()' is deprecated and will be removed"
            " in version 0.9.0.\nConvert the device to a VirtualDevice with "
            "'Device.to_virtual()' and use "
            "'VirtualDevice.change_rydberg_level()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Ignoring type because it expects a VirtualDevice
        # Won't fix because this line will be removed
        VirtualDevice.change_rydberg_level(self, ryd_lvl)  # type: ignore

    def to_virtual(self) -> VirtualDevice:
        """Converts the Device into a VirtualDevice."""
        params = self._params()
        all_params_names = set(params)
        target_params_names = {f.name for f in fields(VirtualDevice)}
        for param in all_params_names - target_params_names:
            del params[param]
        return VirtualDevice(**params)

    def print_specs(self) -> None:
        """Prints the device specifications."""
        title = f"{self.name} Specifications"
        header = ["-" * len(title), title, "-" * len(title)]
        print("\n".join(header))
        print(self._specs())

    def _specs(self, for_docs: bool = False) -> str:
        lines = [
            "\nRegister parameters:",
            f" - Dimensions: {self.dimensions}D",
            f" - Rydberg level: {self.rydberg_level}",
            f" - Maximum number of atoms: {self.max_atom_num}",
            f" - Maximum distance from origin: {self.max_radial_distance} μm",
            (
                " - Minimum distance between neighbouring atoms: "
                f"{self.min_atom_distance} μm"
            ),
            f" - Maximum layout filling fraction: {self.max_layout_filling}",
            f" - SLM Mask: {'Yes' if self.supports_slm_mask else 'No'}",
            "\nChannels:",
        ]

        ch_lines = []
        for name, ch in self._channels:
            if for_docs:
                ch_lines += [
                    f" - ID: '{name}'",
                    f"\t- Type: {ch.name} (*{ch.basis}* basis)",
                    f"\t- Addressing: {ch.addressing}",
                    (
                        "\t"
                        + r"- Maximum :math:`\Omega`:"
                        + f" {ch.max_amp:.4g} rad/µs"
                    ),
                    (
                        "\t"
                        + r"- Maximum :math:`|\delta|`:"
                        + f" {ch.max_abs_detuning:.4g} rad/µs"
                    ),
                ]
                if ch.addressing == "Local":
                    ch_lines += [
                        "\t- Minimum time between retargets: "
                        f"{ch.min_retarget_interval} ns",
                        f"\t- Fixed retarget time: {ch.fixed_retarget_t} ns",
                        f"\t- Maximum simultaneous targets: {ch.max_targets}",
                    ]
                ch_lines += [
                    f"\t- Clock period: {ch.clock_period} ns",
                    f"\t- Minimum instruction duration: {ch.min_duration} ns",
                ]
            else:
                ch_lines.append(f" - '{name}': {ch!r}")

        return "\n".join(lines + ch_lines)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(
            self, _build=False, _module="pulser.devices", _name=self.name
        )

    def _to_abstract_repr(self) -> dict[str, Any]:
        d = super()._to_abstract_repr()
        d["is_virtual"] = False
        return d


@dataclass(frozen=True)
class VirtualDevice(BaseDevice):
    r"""Specifications of a virtual neutral-atom device.

    A VirtualDevice can only be used for emulation and allows some parameters
    to be left undefined. Furthermore, it optionally allows the same channel
    to be declared multiple times in the same Sequence (when
    `reusable_channels=True`) and allows the Rydberg level to be changed.

    Attributes:
        name: The name of the device.
        dimensions: Whether it supports 2D or 3D arrays.
        rybderg_level : The value of the principal quantum number :math:`n`
            when the Rydberg level used is of the form
            :math:`|nS_{1/2}, m_j = +1/2\rangle`.
        max_atom_num: Maximum number of atoms supported in an array.
        max_radial_distance: The furthest away an atom can be from the center
            of the array (in μm).
        min_atom_distance: The closest together two atoms can be (in μm).
        interaction_coeff_xy: :math:`C_3/\hbar` (in :math:`\mu m^3 / \mu s`),
            which sets the van der Waals interaction strength between atoms in
            different Rydberg states.
        supports_slm_mask: Whether the device supports the SLM mask feature.
        max_layout_filling: The largest fraction of a layout that can be filled
            with atoms.
        reusable_channels: Whether each channel can be declared multiple times
            on the same pulse sequence.
    """
    min_atom_distance: float = 0
    max_atom_num: Optional[int] = None
    max_radial_distance: Optional[int] = None
    supports_slm_mask: bool = True
    reusable_channels: bool = True

    @property
    def _optional_parameters(self) -> tuple[str, ...]:
        return ("max_atom_num", "max_radial_distance")

    def change_rydberg_level(self, ryd_lvl: int) -> None:
        """Changes the Rydberg level used in the Device.

        Args:
            ryd_lvl: the Rydberg level to use (between 50 and 100).
        """
        self._validate_rydberg_level(ryd_lvl)
        object.__setattr__(self, "rydberg_level", ryd_lvl)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, _module="pulser.devices", **self._params())

    def _to_abstract_repr(self) -> dict[str, Any]:
        d = super()._to_abstract_repr()
        d["is_virtual"] = True
        return d
