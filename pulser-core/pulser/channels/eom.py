# Copyright 2022 Pulser Development Team
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
"""Configuration parameters for a channel's EOM."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Flag
from itertools import chain
from typing import Any, cast

import numpy as np

from pulser.json.utils import obj_to_dict

# Conversion factor from modulation bandwith to rise time
# For more info, see https://tinyurl.com/bdeumc8k
MODBW_TO_TR = 0.48


class RydbergBeam(Flag):
    """The beams that make up a Rydberg channel."""

    BLUE = 1
    RED = 2

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self.value)

    def _to_abstract_repr(self) -> str:
        return cast(str, self.name)


@dataclass(frozen=True)
class BaseEOM:
    """A base class for the EOM configuration.

    Attributes:
        mod_bandwidth: The EOM modulation bandwidth at -3dB (50% reduction),
            in MHz.
    """

    mod_bandwidth: float  # MHz

    def __post_init__(self) -> None:
        if self.mod_bandwidth <= 0.0:
            raise ValueError(
                "'mod_bandwidth' must be greater than zero, not"
                f" {self.mod_bandwidth}."
            )
        elif self.mod_bandwidth > MODBW_TO_TR * 1e3:
            raise NotImplementedError(
                f"'mod_bandwidth' must be lower than {MODBW_TO_TR*1e3} MHz"
            )

    @property
    def rise_time(self) -> int:
        """The rise time (in ns).

        Defined as the time taken to go from 10% to 90% output in response to
        a step change in the input.
        """
        return int(MODBW_TO_TR / self.mod_bandwidth * 1e3)

    def _to_dict(self) -> dict[str, Any]:
        params = {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }
        return obj_to_dict(self, **params)

    def _to_abstract_repr(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class RydbergEOM(BaseEOM):
    """The EOM configuration for a Rydberg channel.

    Attributes:
        mod_bandwidth: The EOM modulation bandwidth at -3dB (50% reduction),
            in MHz.
        limiting_beam: The beam with the smallest amplitude range.
        max_limiting_amp: The maximum amplitude the limiting beam can reach,
            in rad/µs.
        intermediate_detuning: The detuning between the two beams, in rad/µs.
        controlled_beams: The beams that can be switched on/off with an EOM.
    """

    limiting_beam: RydbergBeam
    max_limiting_amp: float  # rad/µs
    intermediate_detuning: float  # rad/µs
    controlled_beams: tuple[RydbergBeam, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        for param in ["max_limiting_amp", "intermediate_detuning"]:
            value = getattr(self, param)
            if value <= 0.0:
                raise ValueError(
                    f"'{param}' must be greater than zero, not {value}."
                )
        if not isinstance(self.controlled_beams, tuple):
            if not isinstance(self.controlled_beams, list):
                raise TypeError(
                    "The 'controlled_beams' must be provided as a tuple "
                    "or list."
                )
            # Convert list to tuple to keep RydbergEOM hashable
            object.__setattr__(
                self, "controlled_beams", tuple(self.controlled_beams)
            )
        if not self.controlled_beams:
            raise ValueError(
                "There must be at least one beam in 'controlled_beams'."
            )
        for beam in chain((self.limiting_beam,), self.controlled_beams):
            if not (isinstance(beam, RydbergBeam) and beam in RydbergBeam):
                raise TypeError(
                    "Every beam must be one of options of the `RydbergBeam`"
                    f" enumeration, not {self.limiting_beam}."
                )

    def detuning_off_options(
        self, rabi_frequency: float, detuning_on: float
    ) -> np.ndarray:
        """Calculates the possible detuning values when the amplitude is off.

        Args:
            rabi_frequency: The Rabi frequency when executing a pulse,
                in rad/µs.
            detuning_on: The detuning when executing a pulse, in rad/µs.

        Returns:
            The possible detuning values when in between pulses.
        """
        # detuning = offset + lightshift

        # offset takes into account the lightshift when both beams are on
        # which is not zero when the Rabi freq of both beams is not equal
        offset = detuning_on - self._lightshift(rabi_frequency, *RydbergBeam)
        if len(self.controlled_beams) == 1:
            # When only one beam is controlled, the lighshift during delays
            # corresponds to having only the other beam (which can't be
            # switched off) on.
            lightshifts = [
                self._lightshift(rabi_frequency, ~self.controlled_beams[0])
            ]

        else:
            # When both beams are controlled, we have three options for the
            # lightshift: (ON, OFF), (OFF, ON) and (OFF, OFF)
            lightshifts = [
                self._lightshift(rabi_frequency, beam)
                for beam in self.controlled_beams
            ]
            # Case where both beams are off ie (OFF, OFF) -> no lightshift
            lightshifts.append(0.0)

        # We sum the offset to all lightshifts to get the effective detuning
        return np.array(lightshifts) + offset

    def _lightshift(
        self, rabi_frequency: float, *beams_on: RydbergBeam
    ) -> float:
        # lightshift = (rabi_blue**2 - rabi_red**2) / 4 * int_detuning
        rabi_freqs = self._rabi_freq_per_beam(rabi_frequency)
        bias = {RydbergBeam.RED: -1, RydbergBeam.BLUE: 1}
        # beam off -> beam_rabi_freq = 0
        return sum(bias[beam] * rabi_freqs[beam] ** 2 for beam in beams_on) / (
            4 * self.intermediate_detuning
        )

    def _rabi_freq_per_beam(
        self, rabi_frequency: float
    ) -> dict[RydbergBeam, float]:
        # rabi_freq = (rabi_red * rabi_blue) / (2 * int_detuning)
        limit_rabi_freq = self.max_limiting_amp**2 / (
            2 * self.intermediate_detuning
        )
        # limit_rabi_freq is the maximum effective rabi frequency value
        # below which the rabi frequency of both beams can be matched
        if rabi_frequency <= limit_rabi_freq:
            # Both beams the same rabi_freq
            beam_amp = np.sqrt(2 * rabi_frequency * self.intermediate_detuning)
            return {beam: beam_amp for beam in RydbergBeam}

        # The limiting beam is at its maximum amplitude while the other
        # has the necessary amplitude to reach the desired effective rabi freq
        return {
            self.limiting_beam: self.max_limiting_amp,
            ~self.limiting_beam: 2
            * self.intermediate_detuning
            * rabi_frequency
            / self.max_limiting_amp,
        }
