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
"""Defines the detuning map modulator."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional

import numpy as np

import pulser.math as pm
from pulser.channels.base_channel import Channel
from pulser.json.utils import get_dataclass_defaults
from pulser.pulse import Pulse
from pulser.register.weight_maps import DetuningMap

OPTIONAL_ABSTR_DMM_FIELDS = ["total_bottom_detuning", "min_avg_abs_detuning"]


@dataclass(init=True, frozen=True)
class DMM(Channel):
    """Defines a Detuning Map Modulator (DMM) Channel.

    A Detuning Map Modulator can be used to define `Global` detuning Pulses
    (of zero amplitude and phase). These Pulses are locally modulated by the
    weights of a `DetuningMap`, thus providing a local control over the
    detuning. The detuning of the pulses added to a DMM has to be negative,
    between 0 and `bottom_detuning`, and the sum of the weights multiplied by
    that detuning has to be below `total_bottom_detuning`. Channel targeting
    the transition between the ground and rydberg states, thus encoding the
    'ground-rydberg' basis.

    Note:
        The protocol to add pulses to the DMM Channel is by default
        "no-delay".

    Args:
        bottom_detuning: Minimum possible detuning per detuning map spot
            (in rad/µs); must be below zero.
        total_bottom_detuning: Minimum possible detuning distributed over all
            detuning map spots (in rad/µs); must be below zero.
        min_avg_abs_detuning: The minimum acceptable value for the average
            detuning (in absolute value and in rad/µs) applied on any detuning
            map spot (when not 0). Defaults to 0.
        clock_period: The duration of a clock cycle (in ns). The duration of a
            pulse or delay instruction is enforced to be a multiple of the
            clock cycle.
        min_duration: The shortest duration an instruction can take.
        max_duration: The longest duration an instruction can take.
        mod_bandwidth: The modulation bandwidth (in MHz), following Pulser's
            non-standard definition (2x the -3dB bandwidth, or the frequency at
            75% amplitude attenuation).
    """

    bottom_detuning: float | None = None
    total_bottom_detuning: float | None = None
    min_avg_abs_detuning: float = 0.0
    addressing: Literal["Global"] = field(
        default="Global", init=False, repr=False
    )
    max_abs_detuning: Optional[float] = field(
        default=None, init=False, repr=False
    )
    max_amp: float = field(default=0, init=False, repr=False)
    min_retarget_interval: Optional[int] = field(
        default=None, init=False, repr=False
    )
    fixed_retarget_t: Optional[int] = field(
        default=None, init=False, repr=False
    )
    max_targets: Optional[int] = field(default=None, init=False, repr=False)
    propagation_dir: tuple[float, float, float] | None = field(
        default=None, init=False, repr=False
    )
    min_avg_amp: float = field(default=0, init=False, repr=False)
    custom_phase_jump_time: int | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bottom_detuning and self.bottom_detuning > 0:
            raise ValueError(
                "'bottom_detuning' must be negative (got "
                f"{self.bottom_detuning})."
            )
        if self.total_bottom_detuning:
            if self.total_bottom_detuning > 0:
                raise ValueError(
                    "'total_bottom_detuning' must be negative "
                    f"(got {self.total_bottom_detuning})."
                )
            if (
                self.bottom_detuning
                and self.bottom_detuning < self.total_bottom_detuning
            ):
                raise ValueError(
                    f"'total_bottom_detuning' (got "
                    f"{self.total_bottom_detuning}) must be lower than "
                    f"'bottom_detuning' (got {self.bottom_detuning})."
                )
        if self.min_avg_abs_detuning < 0:
            raise ValueError(
                "'min_avg_abs_detuning' must be non-negative "
                f"(got {self.min_avg_abs_detuning})."
            )
        if (
            self.bottom_detuning
            and self.min_avg_abs_detuning >= -self.bottom_detuning
        ):
            bottom_detuning = self.bottom_detuning
            raise ValueError(
                f"'min_avg_abs_detuning' (got {self.min_avg_abs_detuning}) "
                f"must be lower than or equal to {-bottom_detuning=}."
            )

    @property
    def basis(self) -> Literal["ground-rydberg"]:
        """The addressed basis name."""
        return "ground-rydberg"

    def _undefined_fields(self) -> list[str]:
        optional = ["bottom_detuning", "max_duration", "total_bottom_detuning"]
        return [field for field in optional if getattr(self, field) is None]

    def is_virtual(self) -> bool:
        """Whether the channel is virtual (i.e. partially defined)."""
        return bool(self._undefined_fields())

    def validate_pulse(
        self,
        pulse: Pulse,
        detuning_map: DetuningMap = DetuningMap(
            trap_coordinates=[(0, 0)], weights=[1.0]
        ),
    ) -> None:
        """Checks if a pulse can be executed via this DMM on a DetuningMap.

        Args:
            pulse: The pulse to validate.
            detuning_map: The detuning map on which the pulse is applied
                (defaults to a detuning map with weight 1.0).
        """
        super().validate_pulse(pulse)
        round_detuning = pm.round(pulse.detuning.samples, 6).as_array(
            detach=True
        )
        # Check that detuning is negative
        if np.any(round_detuning > 0):
            raise ValueError("The detuning in a DMM must not be positive.")
        # Check that detuning on each atom is above bottom_detuning
        min_round_detuning = np.min(round_detuning)
        max_weight = np.max(detuning_map.weights)
        if (
            self.bottom_detuning is not None
            and max_weight * min_round_detuning < self.bottom_detuning
        ):
            raise ValueError(
                f"For a detuning map with a maximum weight of {max_weight},"
                f" a DMM pulse with minimum detuning {min_round_detuning} "
                "rad/µs goes below the local bottom "
                f"detuning of the DMM ({self.bottom_detuning} rad/µs)."
            )
        # Check that distributed detuning is above total_bottom_detuning
        sum_weight = np.sum(detuning_map.weights)
        if (
            self.total_bottom_detuning is not None
            and sum_weight * min_round_detuning < self.total_bottom_detuning
        ):
            raise ValueError(
                "For a detuning map with a total summed weight of "
                f"{sum_weight}, the total applied detuning from a DMM pulse "
                f"with minimum detuning {min_round_detuning} rad/µs goes below"
                " the total bottom detuning "
                f"of the DMM ({self.total_bottom_detuning} rad/µs)."
            )
        avg_abs_detuning = np.average(np.abs(round_detuning))
        min_non_zero_weight = np.min(
            # Can't be empty because the WeightMap enforces having at least one
            # non-zero weight
            np.array(detuning_map.weights)[np.nonzero(detuning_map.weights)]
        )
        if (
            0
            < min_non_zero_weight * avg_abs_detuning
            < self.min_avg_abs_detuning
        ):
            raise ValueError(
                "For a detuning map with a minimum non-zero weight of "
                f"{min_non_zero_weight}, a DMM pulse with an average "
                f"absolute detuning of {avg_abs_detuning:.3g} rad/µs does not"
                " respect the minimum threshold for the average absolute "
                f"detuning of the DMM ({self.min_avg_abs_detuning} rad/µs)."
            )

    def _to_abstract_repr(self, id: str) -> dict[str, Any]:
        all_fields = fields(self)
        defaults = get_dataclass_defaults(all_fields)
        params = super()._to_abstract_repr(id)
        for p in OPTIONAL_ABSTR_DMM_FIELDS:
            if params[p] == defaults[p]:
                params.pop(p, None)
        return params


def _dmm_id_from_name(dmm_name: str) -> str:
    """Converts a dmm_name into a dmm_id.

    As a reminder the dmm_name is generated automatically from dmm_id
    as dmm_id_{number of times dmm_id has been called}.

    Args:
        dmm_name: The dmm_name to convert.

    Returns:
        The associated dmm_id.
    """
    return "_".join(dmm_name.split("_")[0:2])


def _get_dmm_name(dmm_id: str, channels: list[str]) -> str:
    """Get the dmm_name to add a dmm_id to a list of channels.

    Counts the number of channels starting by dmm_id, generates the
    dmm_name as dmm_id_{number of times dmm_id has been called}.

    Args:
        dmm_id: the id of the DMM to add to the list of channels.
        channels: a list of channel names.

    Returns:
        The associated dmm_name.
    """
    dmm_count = len(
        [key for key in channels if _dmm_id_from_name(key) == dmm_id]
    )
    if dmm_count == 0:
        return dmm_id
    return dmm_id + f"_{dmm_count}"
