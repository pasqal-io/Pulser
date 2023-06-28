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
"""Defines the detuning map modulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pulser.channels.base_channel import Channel


@dataclass(init=True, repr=False, frozen=True)
class DMM(Channel):
    """Defines a Detuning Map Modulator (DMM) Channel.

    A Detuning Map Modulator can be used to define `Global` detuning Pulses
    (of zero amplitude and phase). These Pulses are locally modulated by the
    weights of a `DetuningMap`, thus providing a local control over the
    detuning. The detuning of the pulses added to a DMM has to be negative,
    between 0 and `bottom_detuning`. Channel targeting the transition between
    the ground and rydberg states, thus enconding the 'ground-rydberg' basis.

    Note:
        The protocol to add pulses to the DMM Channel is by default
        "no-delay".

    Args:
        bottom_detuning: Minimum possible detuning (in rad/µs), must be below
            zero.
        clock_period: The duration of a clock cycle (in ns). The duration of a
            pulse or delay instruction is enforced to be a multiple of the
            clock cycle.
        min_duration: The shortest duration an instruction can take.
        max_duration: The longest duration an instruction can take.
        mod_bandwidth: The modulation bandwidth at -3dB (50% reduction), in
            MHz.
    """

    bottom_detuning: Optional[float] = field(default=None, init=True)
    addressing: Literal["Global"] = field(default="Global", init=False)
    max_abs_detuning: Optional[float] = field(init=False, default=None)
    max_amp: float = field(default=0, init=False)
    min_retarget_interval: Optional[int] = field(init=False, default=None)
    fixed_retarget_t: Optional[int] = field(init=False, default=None)
    max_targets: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bottom_detuning and self.bottom_detuning > 0:
            raise ValueError("bottom_detuning must be negative.")

    @property
    def basis(self) -> Literal["ground-rydberg"]:
        """The addressed basis name."""
        return "ground-rydberg"

    # TODO: Block the use of .Global and .Local
