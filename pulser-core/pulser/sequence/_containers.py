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
"""Special containers used by the Sequence class."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple, Union

from pulser.devices._device_datacls import Device
from pulser.pulse import Pulse
from pulser.register.base_register import QubitId
from pulser.sequence._phase_tracker import _PhaseTracker


class _TimeSlot(NamedTuple):
    """Auxiliary class to store the information in the schedule."""

    type: Union[Pulse, str]
    ti: int
    tf: int
    targets: set[QubitId]


# Encodes a sequence building calls
_Call = namedtuple("_Call", ["name", "args", "kwargs"])


@dataclass
class _ChannelSchedule:
    channel_id: str
    device: Device

    def __post_init__(self):
        self.slots: list[_TimeSlot] = []

    @property
    def channel_obj(self):
        return self.device.channels[self.channel_id]

    def last_target(self) -> int:
        """Last time a target happened on the channel."""
        for slot in self.slots[::-1]:
            if slot.type == "target":
                return slot.tf
        return 0

    def get_duration(self, include_fall_time: bool = False) -> int:
        temp_tf = 0
        for i, op in enumerate(self.slots[::-1]):
            if i == 0:
                # Start with the last slot found
                temp_tf = op.tf
                if not include_fall_time:
                    break
            if isinstance(op.type, Pulse):
                temp_tf = max(
                    temp_tf, op.tf + op.type.fall_time(self.channel_obj)
                )
                break
            elif temp_tf - op.tf >= 2 * self.channel_obj.rise_time:
                # No pulse behind 'op' with a long enough fall time
                break
        return temp_tf

    def __getitem__(self, key: Union[int, slice]) -> _TimeSlot:
        return self.slots[key]

    def __len__(self) -> int:
        return len(self.slots)


class _QubitRef:
    def __init__(self):
        self.phase = _PhaseTracker(0)
        self.last_used = 0

    def increment_phase(self, phi: float) -> None:
        self.phase[self.last_used] = self.phase.last_phase + phi

    def update_last_used(self, new_t: int) -> None:
        self.last_used = max(self.last_used, new_t)
