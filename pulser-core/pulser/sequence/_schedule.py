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

from pulser.channels.base_channel import Channel
from pulser.pulse import Pulse
from pulser.register.base_register import QubitId
from pulser.sampler.samples import ChannelSamples, _TargetSlot
from pulser.waveforms import ConstantWaveform


class _TimeSlot(NamedTuple):
    """Auxiliary class to store the information in the schedule."""

    type: Union[Pulse, str]
    ti: int
    tf: int
    targets: set[QubitId]


@dataclass
class _EOMSettings:
    rabi_freq: float
    detuning_on: float
    detuning_off: float
    ti: int
    tf: Optional[int] = None


@dataclass
class _ChannelSchedule:
    channel_id: str
    channel_obj: Channel

    def __post_init__(self) -> None:
        self.slots: list[_TimeSlot] = []
        self.eom_blocks: list[_EOMSettings] = []

    def last_target(self) -> int:
        """Last time a target happened on the channel."""
        for slot in self.slots[::-1]:
            if slot.type == "target":
                return slot.tf
        return 0  # pragma: no cover

    def last_pulse_slot(self, ignore_detuned_delay: bool = False) -> _TimeSlot:
        """The last slot with a Pulse."""
        for slot in self.slots[::-1]:
            if isinstance(slot.type, Pulse) and not (
                ignore_detuned_delay and self.is_detuned_delay(slot.type)
            ):
                return slot
        raise RuntimeError("There is no slot with a pulse.")

    def in_eom_mode(self, time_slot: Optional[_TimeSlot] = None) -> bool:
        """States if a time slot is inside an EOM mode block."""
        if time_slot is None:
            return bool(self.eom_blocks) and (self.eom_blocks[-1].tf is None)
        return any(
            start <= time_slot.ti < end
            for start, end in self.get_eom_mode_intervals()
        )

    @staticmethod
    def is_detuned_delay(pulse: Pulse) -> bool:
        """Tells if a pulse is actually a delay with a constant detuning."""
        return (
            isinstance(pulse, Pulse)
            and isinstance(pulse.amplitude, ConstantWaveform)
            and pulse.amplitude[0] == 0.0
            and isinstance(pulse.detuning, ConstantWaveform)
        )

    def get_eom_mode_intervals(self) -> list[tuple[int, int]]:
        return [
            (
                block.ti,
                block.tf if block.tf is not None else self.get_duration(),
            )
            for block in self.eom_blocks
        ]

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
                    temp_tf,
                    op.tf
                    + op.type.fall_time(
                        self.channel_obj, in_eom_mode=self.in_eom_mode()
                    ),
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

    def get_samples(
        self, ignore_detuned_delay_phase: bool = True
    ) -> ChannelSamples:
        """Returns the samples of the channel."""
        # Keep only pulse slots
        channel_slots = [s for s in self.slots if isinstance(s.type, Pulse)]
        dt = self.get_duration()
        amp, det, phase = np.zeros(dt), np.zeros(dt), np.zeros(dt)
        slots: list[_TargetSlot] = []

        for ind, s in enumerate(channel_slots):
            pulse = cast(Pulse, s.type)
            amp[s.ti : s.tf] += pulse.amplitude.samples
            det[s.ti : s.tf] += pulse.detuning.samples

            tf = s.tf
            # Account for the extended duration of the pulses
            # after modulation, which is at most fall_time
            fall_time = pulse.fall_time(
                self.channel_obj, in_eom_mode=self.in_eom_mode(time_slot=s)
            )
            tf += (
                min(fall_time, channel_slots[ind + 1].ti - s.tf)
                if ind < len(channel_slots) - 1
                else fall_time
            )
            slots.append(_TargetSlot(s.ti, tf, s.targets))

            if ignore_detuned_delay_phase and self.is_detuned_delay(pulse):
                # The phase of detuned delays is not considered
                continue

            ph_jump_t = self.channel_obj.phase_jump_time
            for last_pulse_ind in range(ind - 1, -1, -1):  # From ind-1 to 0
                last_pulse_slot = channel_slots[last_pulse_ind]
                # Skips over detuned delay pulses
                if not (
                    ignore_detuned_delay_phase
                    and self.is_detuned_delay(
                        cast(Pulse, last_pulse_slot.type)
                    )
                ):
                    # Accounts for when pulse is added with 'no-delay'
                    # i.e. there is no phase_jump_time in between a phase jump
                    t_start = max(s.ti - ph_jump_t, last_pulse_slot.tf)
                    break
            else:
                t_start = 0
            # Overrides all values from t_start on. The next pulses will do
            # the same, so the last phase is automatically kept till the end
            phase[t_start:] = pulse.phase

        return ChannelSamples(amp, det, phase, slots, self.eom_blocks)

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

    def enable_eom(
        self,
        channel_id: str,
        amp_on: float,
        detuning_on: float,
        detuning_off: float,
        _skip_buffer: bool = False,
    ) -> None:
        channel_obj = self[channel_id].channel_obj
        # Adds a buffer unless the channel is empty or _skip_buffer = True
        if not _skip_buffer and self.get_duration(channel_id):
            # Wait for the last pulse to ramp down (if needed)
            self.wait_for_fall(channel_id)
            # Account for time needed to ramp to desired amplitude
            # By definition, rise_time goes from 10% to 90%
            # Roughly 2*rise_time is enough to go from 0% to 100%
            if detuning_off != 0:
                self.add_pulse(
                    Pulse.ConstantPulse(
                        2 * channel_obj.rise_time,
                        0.0,
                        detuning_off,
                        self._get_last_pulse_phase(channel_id),
                    ),
                    channel_id,
                    phase_barrier_ts=[0],
                    protocol="no-delay",
                )
            else:
                self.add_delay(2 * channel_obj.rise_time, channel_id)

        # Set up the EOM
        eom_settings = _EOMSettings(
            rabi_freq=amp_on,
            detuning_on=detuning_on,
            detuning_off=detuning_off,
            ti=self[channel_id][-1].tf,
        )

        self[channel_id].eom_blocks.append(eom_settings)

    def disable_eom(self, channel_id: str, _skip_buffer: bool = False) -> None:
        self[channel_id].eom_blocks[-1].tf = self[channel_id][-1].tf
        if not _skip_buffer:
            self.wait_for_fall(channel_id)

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
        # Buffer to add between pulses of different phase
        phase_jump_buffer = 0
        if protocol != "no-delay":
            current_max_t = self._find_add_delay(
                current_max_t, channel, protocol
            )
            try:
                # Gets the last pulse on the channel
                last_pulse_slot = self[channel].last_pulse_slot(
                    ignore_detuned_delay=True
                )
                last_pulse = cast(Pulse, last_pulse_slot.type)
                # Checks if the current pulse changes the phase
                if last_pulse.phase != pulse.phase:
                    # Subtracts the time that has already elapsed since the
                    # last pulse from the phase_jump_time and adds the
                    # fall_time to let the last pulse ramp down
                    ch_obj = self[channel].channel_obj
                    phase_jump_buffer = (
                        ch_obj.phase_jump_time
                        + last_pulse.fall_time(
                            ch_obj, in_eom_mode=self[channel].in_eom_mode()
                        )
                        - (t0 - last_pulse_slot.tf)
                    )
            except RuntimeError:
                # No previous pulse
                pass

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
        if (
            self[channel].in_eom_mode()
            and self[channel].eom_blocks[-1].detuning_off != 0
        ):
            phase = self._get_last_pulse_phase(channel)
            delay_pulse = Pulse.ConstantPulse(
                tf - ti, 0.0, self[channel].eom_blocks[-1].detuning_off, phase
            )
            self[channel].slots.append(
                _TimeSlot(delay_pulse, ti, tf, last.targets)
            )
        else:
            self[channel].slots.append(
                _TimeSlot("delay", ti, tf, last.targets)
            )

    def add_target(self, qubits_set: set[QubitId], channel: str) -> None:
        channel_obj = self[channel].channel_obj
        if self[channel].slots:
            self.wait_for_fall(channel)

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

    def wait_for_fall(self, channel: str) -> None:
        """Adds a delay to let the channel's amplitude ramp down."""
        # Extra time needed for the output to finish
        fall_time = (
            self[channel].get_duration(include_fall_time=True)
            - self[channel].get_duration()
        )
        # If there is a fall time, a delay is added to account for it
        if fall_time > 0:
            self.add_delay(self[channel].adjust_duration(fall_time), channel)

    def _find_add_delay(self, t0: int, channel: str, protocol: str) -> int:
        current_max_t = t0
        for ch, ch_schedule in self.items():
            if ch == channel:
                continue
            this_chobj = self[ch].channel_obj
            in_eom_mode = self[ch].in_eom_mode()
            for op in ch_schedule[::-1]:
                if not isinstance(op.type, Pulse):
                    if op.tf + 2 * this_chobj.rise_time <= current_max_t:
                        # No pulse behind 'op' needing a delay
                        break
                elif (
                    op.tf
                    + op.type.fall_time(this_chobj, in_eom_mode=in_eom_mode)
                    <= current_max_t
                ):
                    break
                elif (
                    op.targets & self[channel][-1].targets
                    or protocol == "wait-for-all"
                ):
                    current_max_t = op.tf + op.type.fall_time(
                        this_chobj, in_eom_mode=in_eom_mode
                    )
                    break

        return current_max_t

    def _get_last_pulse_phase(self, channel: str) -> float:
        try:
            last_pulse = cast(Pulse, self[channel].last_pulse_slot().type)
            phase = last_pulse.phase
        except RuntimeError:
            phase = 0.0
        return phase
