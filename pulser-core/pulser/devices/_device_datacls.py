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
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Any, Literal, cast, get_args

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pulser.channels.base_channel import Channel
from pulser.channels.dmm import DMM
from pulser.devices.interaction_coefficients import c6_dict
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.json.utils import get_dataclass_defaults, obj_to_dict
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register_layout import RegisterLayout
from pulser.register.traps import COORD_PRECISION

DIMENSIONS = Literal[2, 3]

ALWAYS_OPTIONAL_PARAMS = ("max_sequence_duration", "max_runs", "dmm_objects")
PARAMS_WITH_ABSTR_REPR = ("channel_objects", "channel_ids", "dmm_objects")


@dataclass(frozen=True, repr=False)
class BaseDevice(ABC):
    r"""Base class of a neutral-atom device.

    Attributes:
        name: The name of the device.
        dimensions: Whether it supports 2D or 3D arrays.
        channel_objects: The Channel subclass instances specifying each
            channel in the device.
        channel_ids: Custom IDs for each channel object. When defined,
            an ID must be given for each channel. If not defined, the IDs are
            generated internally based on the channels' names and addressing.
        dmm_objects: The DMM subclass instances specifying each channel in the
            device. They are referenced by their order in the list, with the ID
            "dmm_[index in dmm_objects]".
        rybderg_level: The value of the principal quantum number :math:`n`
            when the Rydberg level used is of the form
            :math:`|nS_{1/2}, m_j = +1/2\rangle`.
        max_atom_num: Maximum number of atoms supported in an array.
        max_radial_distance: The furthest away an atom can be from the center
            of the array (in μm).
        min_atom_distance: The closest together two atoms can be (in μm).
        interaction_coeff_xy: :math:`C_3/\hbar`
            (in :math:`rad \cdot \mu s^{-1} \cdot \mu m^3`),
            which sets the van der Waals interaction strength between atoms in
            different Rydberg states. Needed only if there is a Microwave
            channel in the device. If unsure, 3700.0 is a good default value.
        supports_slm_mask: Whether the device supports the SLM mask feature.
        max_layout_filling: The largest fraction of a layout that can be filled
            with atoms.
        max_sequence_duration: The maximum allowed duration for a sequence
            (in ns).
        max_runs: The maximum number of runs allowed on the device. Only used
            for backend execution.
    """
    name: str
    dimensions: DIMENSIONS
    rydberg_level: int
    min_atom_distance: float
    max_atom_num: int | None
    max_radial_distance: int | None
    interaction_coeff_xy: float | None = None
    supports_slm_mask: bool = False
    max_layout_filling: float = 0.5
    max_sequence_duration: int | None = None
    max_runs: int | None = None
    reusable_channels: bool = field(default=False, init=False)
    channel_ids: tuple[str, ...] | None = None
    channel_objects: tuple[Channel, ...] = field(default_factory=tuple)
    dmm_objects: tuple[DMM, ...] = field(default_factory=tuple)

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

        for param in (
            "min_atom_distance",
            "max_atom_num",
            "max_radial_distance",
            "max_sequence_duration",
            "max_runs",
        ):
            value = getattr(self, param)
            if (
                param in self._optional_parameters
                or param in ALWAYS_OPTIONAL_PARAMS
            ):
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

        type_check("supports_slm_mask", bool)
        type_check("reusable_channels", bool)

        if not (0.0 < self.max_layout_filling <= 1.0):
            raise ValueError(
                "The maximum layout filling fraction must be "
                "greater than 0. and less than or equal to 1., "
                f"not {self.max_layout_filling}."
            )

        for ch_obj in self.channel_objects:
            type_check("All channels", Channel, value_override=ch_obj)

        for dmm_obj in self.dmm_objects:
            type_check("All DMM channels", DMM, value_override=dmm_obj)

        if self.supports_slm_mask and not self.dmm_objects:
            raise ValueError(
                "One DMM object should be defined to support SLM mask."
            )

        if self.channel_ids is not None:
            if not (
                isinstance(self.channel_ids, (tuple, list))
                and all(isinstance(el, str) for el in self.channel_ids)
            ):
                raise TypeError(
                    "When defined, 'channel_ids' must be a tuple or a list "
                    "of strings."
                )
            if len(self.channel_ids) != len(set(self.channel_ids)):
                raise ValueError(
                    "When defined, 'channel_ids' can't have "
                    "repeated elements."
                )
            if len(self.channel_ids) != len(self.channel_objects):
                raise ValueError(
                    "When defined, the number of channel IDs must"
                    " match the number of channel objects."
                )
            if set(self.channel_ids) & set(self.dmm_channels.keys()):
                raise ValueError(
                    "When defined, the names of channel IDs must be different"
                    " than the names of DMM channels 'dmm_0', 'dmm_1', ... ."
                )

        else:
            # Make the channel IDs from the default IDs
            ids_counter: Counter = Counter()
            ids = []
            for ch_obj in self.channel_objects:
                id = ch_obj.default_id()
                ids_counter.update([id])
                if ids_counter[id] > 1:
                    # If there is more than one with the same ID
                    id += f"_{ids_counter[id]}"
                ids.append(id)
            object.__setattr__(self, "channel_ids", tuple(ids))

        if any(
            ch.basis == "XY" for ch in self.channel_objects
        ) and not isinstance(self.interaction_coeff_xy, float):
            raise TypeError(
                "When the device has a 'Microwave' channel, "
                "'interaction_coeff_xy' must be a 'float',"
                f" not '{type(self.interaction_coeff_xy)}'."
            )

        def to_tuple(obj: tuple | list) -> tuple:
            if isinstance(obj, (tuple, list)):
                obj = tuple(to_tuple(el) for el in obj)
            return obj

        # Turns mutable lists into immutable tuples
        for param in self._params():
            if "channel" in param or param == "dmm_objects":
                object.__setattr__(self, param, to_tuple(getattr(self, param)))

    @property
    @abstractmethod
    def _optional_parameters(self) -> tuple[str, ...]:
        pass

    @property
    def channels(self) -> dict[str, Channel]:
        """Dictionary of available channels on this device."""
        return dict(zip(cast(tuple, self.channel_ids), self.channel_objects))

    @property
    def dmm_channels(self) -> dict[str, DMM]:
        """Dictionary of available DMM channels on this device."""
        return {
            f"dmm_{i}": dmm_obj for (i, dmm_obj) in enumerate(self.dmm_objects)
        }

    @property
    def supported_bases(self) -> set[str]:
        """Available electronic transitions for control and measurement."""
        return {ch.basis for ch in self.channel_objects}

    @property
    def interaction_coeff(self) -> float:
        r"""The interaction coefficient for the chosen Rydberg level.

        Corresponds to :math:`C_6/\hbar` (in units of
        :math:`rad \cdot \mu s^{-1} \cdot \mu m^6`)
        for the interaction term of the Ising hamiltonian.
        """
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
        # mypy can't guarantee that float**float is a float, so we need to cast
        return cast(
            float, (self.interaction_coeff / rabi_frequency) ** (1 / 6)
        )

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

        if register.dimensionality > self.dimensions:
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

    def _params(self, init_only: bool = False) -> dict[str, Any]:
        # This is used instead of dataclasses.asdict() because asdict()
        # is recursive and we have Channel dataclasses in the args that
        # we don't want to convert to dict
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not init_only or f.init
        }

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
        defaults = get_dataclass_defaults(fields(self))
        params = self._params()
        for p in ALWAYS_OPTIONAL_PARAMS:
            if params[p] == defaults[p]:
                params.pop(p, None)
        # Delete parameters of PARAMS_WITH_ABSTR_REPR in params
        for p in PARAMS_WITH_ABSTR_REPR:
            params.pop(p, None)
        ch_list = []
        for ch_name, ch_obj in self.channels.items():
            ch_list.append(ch_obj._to_abstract_repr(ch_name))
        # Add version and channels to params
        params.update({"version": "1", "channels": ch_list})
        dmm_list = []
        for dmm_name, dmm_obj in self.dmm_channels.items():
            dmm_list.append(dmm_obj._to_abstract_repr(dmm_name))
        if dmm_list:
            params["dmm_objects"] = dmm_list
        return params

    def to_abstract_repr(self) -> str:
        """Serializes the device into an abstract JSON object."""
        abstr_dev_str = json.dumps(self, cls=AbstractReprEncoder)
        validate_abstract_repr(abstr_dev_str, "device")
        return abstr_dev_str


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
        interaction_coeff_xy: :math:`C_3/\hbar`
            (in :math:`rad \cdot \mu s^{-1} \cdot \mu m^3`),
            which sets the van der Waals interaction strength between atoms in
            different Rydberg states. Needed only if there is a Microwave
            channel in the device. If unsure, 3700.0 is a good default value.
        supports_slm_mask: Whether the device supports the SLM mask feature.
        max_layout_filling: The largest fraction of a layout that can be filled
            with atoms.
        max_sequence_duration: The maximum allowed duration for a sequence
            (in ns).
        max_runs: The maximum number of runs allowed on the device. Only used
            for backend execution.
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
        for ch_id, ch_obj in {**self.channels, **self.dmm_channels}.items():
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

    def is_calibrated_layout(self, register_layout: RegisterLayout) -> bool:
        """Checks whether a layout is within the calibrated layouts.

        Args:
            register_layout: The RegisterLayout to check.

        Returns:
            True if register_layout is found among calibrated_register_layouts,
            False otherwise.
        """
        return any(
            [
                register_layout == layout
                for layout in list(self.calibrated_register_layouts.values())
            ]
        )

    def register_is_from_calibrated_layout(
        self, register: BaseRegister | MappableRegister
    ) -> bool:
        """Checks whether a register was constructed from a calibrated layout.

        If the register is a BaseRegister, checks that it has a layout. If so,
        or if it is a MappableRegister, check that its layout is within the
        calibrated layouts.

        Args:
            register_layout: the Register or MappableRegister to check.

        Returns:
            True if register has a layout and it is found among
            calibrated_register_layouts, False otherwise.
        """
        if not isinstance(register, (BaseRegister, MappableRegister)):
            raise TypeError(
                "The register to check must be of type "
                "BaseRegister or MappableRegister."
            )
        if isinstance(register, BaseRegister) and register.layout is None:
            return False
        return self.is_calibrated_layout(cast(RegisterLayout, register.layout))

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
        ]

        if self.max_sequence_duration is not None:
            lines.append(
                " - Maximum sequence duration: "
                f"{self.max_sequence_duration} ns"
            )

        ch_lines = ["\nChannels:"]
        for name, ch in {**self.channels, **self.dmm_channels}.items():
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
                    )
                    if not isinstance(ch, DMM)
                    else (
                        "\t"
                        + r"- Bottom :math:`|\delta|`:"
                        + f" {ch.bottom_detuning:.4g} rad/µs"
                    ),
                    f"\t- Minimum average amplitude: {ch.min_avg_amp} rad/µs",
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
        interaction_coeff_xy: :math:`C_3/\hbar`
            (in :math:`rad \cdot \mu s^{-1} \cdot \mu m^3`),
            which sets the van der Waals interaction strength between atoms in
            different Rydberg states. Needed only if there is a Microwave
            channel in the device. If unsure, 3700.0 is a good default value.
        supports_slm_mask: Whether the device supports the SLM mask feature.
        max_layout_filling: The largest fraction of a layout that can be filled
            with atoms.
        max_sequence_duration: The maximum allowed duration for a sequence
            (in ns).
        max_runs: The maximum number of runs allowed on the device. Only used
            for backend execution.
        reusable_channels: Whether each channel can be declared multiple times
            on the same pulse sequence.
    """
    min_atom_distance: float = 0
    max_atom_num: int | None = None
    max_radial_distance: int | None = None
    supports_slm_mask: bool = True
    # Needed to support SLM mask by default
    dmm_objects: tuple[DMM, ...] = (DMM(),)
    reusable_channels: bool = True

    @property
    def _optional_parameters(self) -> tuple[str, ...]:
        return ("max_atom_num", "max_radial_distance")

    def change_rydberg_level(self, ryd_lvl: int) -> None:
        r"""Changes the Rydberg level used in the Device.

        Find the :math:`C_6/\hbar` coefficient matching the Rydberg level on
        `this page <https://github.com/pasqal-io/Pulser/blob/develop/
        pulser-core/pulser/devices/interaction_coefficients/C6_coeffs.json>`_

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
