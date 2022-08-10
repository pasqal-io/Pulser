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
"""Special containers to store the schedule of operations in the Sequence."""
from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Union, cast, overload

import numpy as np

from pulser.channels import Channel
from pulser.pulse import Pulse
from pulser.register.base_register import QubitId
from pulser.sampler.samples import ChannelSamples, _TargetSlot


class _TimeSlot(NamedTuple):
    """Auxiliary class to store the information in the schedule."""

    type: Union[Pulse, str]
    ti: int
    tf: int
    targets: set[QubitId]


@dataclass
class _ChannelSchedule:
    channel_id: str
    channel_obj: Channel

    def __post_init__(self) -> None:
        self.slots: list[_TimeSlot] = []

    def last_target(self) -> int:
        """Last time a target happened on the channel."""
        for slot in self.slots[::-1]:
            if slot.type == "target":
                return slot.tf
        return 0  # pragma: no cover

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

    def adjust_duration(self, duration: int) -> int:
        """Adjust a duration for this channel."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.channel_obj.validate_duration(
                max(duration, self.channel_obj.min_duration)
            )

    def get_samples(self, modulated: bool = False) -> ChannelSamples:
        """Returns the samples of the channel.

        Args:
            modulated: Whether to return the modulated samples.
        """
        # Keep only pulse slots
        channel_slots = [s for s in self.slots if isinstance(s.type, Pulse)]
        dt = self.get_duration()
        amp, det, phase = np.zeros(dt), np.zeros(dt), np.zeros(dt)
        slots: list[_TargetSlot] = []

        for ind, s in enumerate(channel_slots):
            pulse = cast(Pulse, s.type)
            amp[s.ti : s.tf] += pulse.amplitude.samples
            det[s.ti : s.tf] += pulse.detuning.samples
            ph_jump_t = self.channel_obj.phase_jump_time
            t_start = s.ti - ph_jump_t if ind > 0 else 0
            t_end = (
                channel_slots[ind + 1].ti - ph_jump_t
                if ind < len(channel_slots) - 1
                else dt
            )
            phase[t_start:t_end] += pulse.phase
            tf = s.tf
            if modulated:
                # Account for the extended duration of the pulses
                # after modulation, which is at most fall_time
                fall_time = pulse.fall_time(self.channel_obj)
                tf += (
                    min(fall_time, channel_slots[ind + 1].ti - s.tf)
                    if ind < len(channel_slots) - 1
                    else fall_time
                )

            slots.append(_TargetSlot(s.ti, tf, s.targets))

        ch_samples = ChannelSamples(amp, det, phase, slots)

        if modulated:
            ch_samples = ch_samples.modulate(
                self.channel_obj,
                max_duration=self.get_duration(include_fall_time=True),
            )

        return ch_samples

    @overload
    def __getitem__(self, key: int) -> _TimeSlot:
        pass

    @overload
    def __getitem__(self, key: slice) -> list[_TimeSlot]:
        pass

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[_TimeSlot, list[_TimeSlot]]:
        if key == -1 and not self.slots:
            raise ValueError("The chosen channel has no target.")
        return self.slots[key]

    def __iter__(self) -> Iterator[_TimeSlot]:
        for slot in self.slots:
            yield slot


class _Schedule(Dict[str, _ChannelSchedule]):
    def get_duration(
        self, channel: Optional[str] = None, include_fall_time: bool = False
    ) -> int:
        if channel is None:
            channels = tuple(self.keys())
            if not channels:
                return 0
        else:
            channels = (channel,)

        return max(self[id].get_duration(include_fall_time) for id in channels)

    def find_slm_mask_times(self) -> list[int]:
        # Find tentative initial and final time of SLM mask if possible
        mask_time: list[int] = []
        for ch_schedule in self.values():
            if ch_schedule.channel_obj.addressing != "Global":
                continue
            # Cycle on slots in schedule until the first pulse is found
            for slot in ch_schedule:
                if not isinstance(slot.type, Pulse):
                    continue
                ti = slot.ti
                tf = slot.tf
                if mask_time:
                    if ti < mask_time[0]:
                        mask_time = [ti, tf]
                else:
                    mask_time = [ti, tf]
                break
        return mask_time

    def add_pulse(
        self,
        pulse: Pulse,
        channel: str,
        phase_barrier_ts: list[int],
        protocol: str,
    ) -> None:
        pass
        last = self[channel][-1]
        t0 = last.tf
        current_max_t = max(t0, *phase_barrier_ts)
        phase_jump_buffer = 0
        for ch, ch_schedule in self.items():
            if protocol == "no-delay" and ch != channel:
                continue
            this_chobj = self[ch].channel_obj
            for op in ch_schedule[::-1]:
                if not isinstance(op.type, Pulse):
                    if op.tf + 2 * this_chobj.rise_time <= current_max_t:
                        # No pulse behind 'op' needing a delay
                        break
                elif ch == channel:
                    if op.type.phase != pulse.phase:
                        phase_jump_buffer = this_chobj.phase_jump_time - (
                            t0 - op.tf
                        )
                    break
                elif op.tf + op.type.fall_time(this_chobj) <= current_max_t:
                    break
                elif op.targets & last.targets or protocol == "wait-for-all":
                    current_max_t = op.tf + op.type.fall_time(this_chobj)
                    break

        delay_duration = max(current_max_t - t0, phase_jump_buffer)
        if delay_duration > 0:
            delay_duration = self[channel].adjust_duration(delay_duration)
            self.add_delay(delay_duration, channel)

        ti = t0 + delay_duration
        tf = ti + pulse.duration
        self[channel].slots.append(_TimeSlot(pulse, ti, tf, last.targets))

    def add_delay(self, duration: int, channel: str) -> None:
        last = self[channel][-1]
        ti = last.tf
        tf = ti + self[channel].channel_obj.validate_duration(duration)
        self[channel].slots.append(_TimeSlot("delay", ti, tf, last.targets))

    def add_target(self, qubits_set: set[QubitId], channel: str) -> None:
        channel_obj = self[channel].channel_obj
        if self[channel].slots:
            fall_time = (
                self[channel].get_duration(include_fall_time=True)
                - self[channel].get_duration()
            )
            if fall_time > 0:
                self.add_delay(
                    self[channel].adjust_duration(fall_time), channel
                )

            last = self[channel][-1]
            if last.targets == qubits_set:
                return
            ti = last.tf
            retarget = cast(int, channel_obj.min_retarget_interval)
            elapsed = ti - self[channel].last_target()
            delta = cast(int, np.clip(retarget - elapsed, 0, retarget))
            if channel_obj.fixed_retarget_t:
                delta = max(delta, channel_obj.fixed_retarget_t)
            if delta != 0:
                delta = self[channel].adjust_duration(delta)
            tf = ti + delta

        else:
            ti = -1
            tf = 0

        self[channel].slots.append(
            _TimeSlot("target", ti, tf, set(qubits_set))
        )
