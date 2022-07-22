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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pulser.channels import Channel
from pulser.devices.interaction_coefficients import c6_dict
from pulser.json.utils import obj_to_dict
from pulser.pulse import Pulse
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.register_layout import COORD_PRECISION, RegisterLayout


@dataclass(frozen=True, repr=False)
class Device:
    r"""Definition of a neutral-atom device.

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
    """

    name: str
    dimensions: int
    rydberg_level: int
    max_atom_num: int
    max_radial_distance: int
    min_atom_distance: int
    _channels: tuple[tuple[str, Channel], ...]
    # Ising interaction coeff
    interaction_coeff_xy: float = 3700.0
    pre_calibrated_layouts: tuple[RegisterLayout, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        # Hack to override the docstring of an instance
        object.__setattr__(self, "__doc__", self._specs(for_docs=True))
        for layout in self.pre_calibrated_layouts:
            self.validate_layout(layout)

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

    @property
    def calibrated_register_layouts(self) -> dict[str, RegisterLayout]:
        """Register layouts already calibrated on this device."""
        return {str(layout): layout for layout in self.pre_calibrated_layouts}

    def print_specs(self) -> None:
        """Prints the device specifications."""
        title = f"{self.name} Specifications"
        header = ["-" * len(title), title, "-" * len(title)]
        print("\n".join(header))
        print(self._specs())

    def __repr__(self) -> str:
        return self.name

    def change_rydberg_level(self, ryd_lvl: int) -> None:
        """Changes the Rydberg level used in the Device.

        Args:
            ryd_lvl: the Rydberg level to use (between 50 and 100).

        Note:
            Modifications to the `rydberg_level` attribute only affect the
            outcomes of local emulations.
        """
        if not isinstance(ryd_lvl, int):
            raise TypeError("Rydberg level has to be an int.")
        if not ((49 < ryd_lvl) & (101 > ryd_lvl)):
            raise ValueError("Rydberg level should be between 50 and 100.")

        object.__setattr__(self, "rydberg_level", ryd_lvl)

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

    def validate_pulse(self, pulse: Pulse, channel_id: str) -> None:
        """Checks if a pulse can be executed on a specific device channel.

        Args:
            pulse: The pulse to validate.
            channel_id: The channel ID used to index the chosen channel
                on this device.
        """
        if not isinstance(pulse, Pulse):
            raise TypeError(
                f"'pulse' must be of type Pulse, not of type {type(pulse)}."
            )

        ch = self.channels[channel_id]
        if np.any(pulse.amplitude.samples > ch.max_amp):
            raise ValueError(
                "The pulse's amplitude goes over the maximum "
                "value allowed for the chosen channel."
            )
        if np.any(
            np.round(np.abs(pulse.detuning.samples), decimals=6)
            > ch.max_abs_detuning
        ):
            raise ValueError(
                "The pulse's detuning values go out of the range "
                "allowed for the chosen channel."
            )

    def _specs(self, for_docs: bool = False) -> str:
        lines = [
            "\nRegister requirements:",
            f" - Dimensions: {self.dimensions}D",
            rf" - Rydberg level: {self.rydberg_level}",
            f" - Maximum number of atoms: {self.max_atom_num}",
            f" - Maximum distance from origin: {self.max_radial_distance} μm",
            (
                " - Minimum distance between neighbouring atoms: "
                f"{self.min_atom_distance} μm"
            ),
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
                    f"\t- Phase Jump Time: {ch.phase_jump_time} ns",
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

    def _validate_coords(
        self, coords_dict: dict[QubitId, np.ndarray], kind: str = "atoms"
    ) -> None:
        ids = list(coords_dict.keys())
        coords = list(coords_dict.values())
        max_number = self.max_atom_num * (2 if kind == "traps" else 1)
        if len(coords) > max_number:
            raise ValueError(
                f"The number of {kind} ({len(coords)})"
                " must be less than or equal to the maximum"
                f" number of {kind} supported by this device"
                f" ({max_number})."
            )

        if len(coords) > 1:
            distances = pdist(coords)  # Pairwise distance between atoms
            if np.any(
                distances - self.min_atom_distance
                < -(10 ** (-COORD_PRECISION))
            ):
                sq_dists = squareform(distances)
                mask = np.triu(np.ones(len(coords), dtype=bool), k=1)
                bad_pairs = np.argwhere(
                    np.logical_and(sq_dists < self.min_atom_distance, mask)
                )
                bad_qbt_pairs = [(ids[i], ids[j]) for i, j in bad_pairs]
                raise ValueError(
                    f"The minimal distance between {kind} in this device "
                    f"({self.min_atom_distance} µm) is not respected for the "
                    f"pairs: {bad_qbt_pairs}"
                )

        too_far = np.linalg.norm(coords, axis=1) > self.max_radial_distance
        if np.any(too_far):
            raise ValueError(
                f"All {kind} must be at most {self.max_radial_distance} μm "
                f"away from the center of the array, which is not the case "
                f"for: {[ids[int(i)] for i in np.where(too_far)[0]]}"
            )

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(
            self, _build=False, _module="pulser.devices", _name=self.name
        )
