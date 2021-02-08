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

from collections import namedtuple
from collections.abc import Iterable
import copy
import warnings

import numpy as np

import pulser
from pulser.pulse import Pulse
from pulser.devices import MockDevice
from pulser._seq_drawer import draw_sequence
from pulser.utils import validate_duration

# Auxiliary class to store the information in the schedule
_TimeSlot = namedtuple('_TimeSlot', ['type', 'ti', 'tf', 'targets'])


class Sequence:
    """A sequence of operations on a device.

    A sequence is composed by

        - The device in which we want to implement it
        - The register of qubits on which to act
        - The device's channels that are used
        - The schedule of operations on each channel

    Args:
        register(Register): The atom register on which to apply the pulses.
        device(PasqalDevice): A valid device in which to execute the Sequence
            (import it from ``pulser.devices``).
    """
    def __init__(self, register, device):
        """Initializes a new pulse sequence."""
        cond1 = device not in pulser.devices._valid_devices
        cond2 = device != MockDevice
        if cond1 and cond2:
            names = [d.name for d in pulser.devices._valid_devices]
            error_msg = ("The Sequence's device has to be imported from "
                         + "pasqal.devices. Choose 'MockDevice' or between the"
                         + " following real devices:\n" + "\n".join(names))
            raise ValueError(error_msg)
        # Checks if register is compatible with the device
        device.validate_register(register)

        self._register = register
        self._device = device
        self._channels = {}
        self._schedule = {}
        self._phase_ref = {}  # The phase reference of each channel
        # Stores the ids of selected channels and their declared names
        self._taken_channels = {}
        self._qids = set(self.qubit_info.keys())  # IDs of all qubits in device
        self._last_used = {}    # Last time each qubit was used, by basis
        self._last_target = {}  # Last time a target happened, by channel

    @property
    def qubit_info(self):
        """Dictionary with the qubit's IDs and positions."""
        return self._register.qubits

    @property
    def declared_channels(self):
        """Channels declared in this Sequence."""
        return dict(self._channels)

    @property
    def available_channels(self):
        """Channels still available for declaration."""
        return {id: ch for id, ch in self._device.channels.items()
                if id not in self._taken_channels
                or self._device == MockDevice}

    def current_phase_ref(self, qubit, basis='digital'):
        """Current phase reference of a specific qubit for a given basis.

        Args:
            qubit (str): The id of the qubit whose phase shift is desired.

        Keyword args:
            basis (str): The basis (i.e. electronic transition) the phase
                reference is associated with. Must correspond to the basis of a
                declared channel.

        Returns:
            float: Current phase reference of 'qubit' in 'basis'.

        """
        if qubit not in self._qids:
            raise ValueError("'qubit' must be the id of a qubit declared in "
                             "this sequence's device.")

        if basis not in self._phase_ref:
            raise ValueError("No declared channel targets the given 'basis'.")

        return self._phase_ref[basis][qubit].last_phase

    def declare_channel(self, name, channel_id, initial_target=None):
        """Declares a new channel to the Sequence.

        Args:
            name (str): Unique name for the channel in the sequence.
            channel_id (str): How the channel is identified in the device.
                Consult ``Sequence.available_channels`` to see which channel
                ID's are still available and the associated channel's
                description.

        Keyword Args:
            initial_target (set, default=None): For 'Local' adressing channels
                only. Declares the initial target of the channel. If left as
                None, the initial target will have to be set manually as the
                first addition to this channel.
        """

        if name in self._channels:
            raise ValueError("The given name is already in use.")

        if channel_id not in self._device.channels:
            raise ValueError("No channel %s in the device." % channel_id)

        if channel_id not in self.available_channels:
            raise ValueError("Channel %s is not available." % channel_id)

        ch = self._device.channels[channel_id]
        self._channels[name] = ch
        self._taken_channels[channel_id] = name
        self._schedule[name] = []
        self._last_target[name] = 0

        if ch.basis not in self._phase_ref:
            self._phase_ref[ch.basis] = {q: _PhaseTracker(0)
                                         for q in self._qids}
            self._last_used[ch.basis] = {q: 0 for q in self._qids}

        if ch.addressing == 'Global':
            self._add_to_schedule(name, _TimeSlot('target', -1, 0, self._qids))
        elif initial_target is not None:
            self.target(initial_target, name)

    def add(self, pulse, channel, protocol='min-delay'):
        """Adds a pulse to a channel.

        Args:
            pulse (pulser.Pulse): The pulse object to add to the channel.
            channel (str): The channel's name provided when declared.

        Keyword Args:
            protocol (default='min-delay'): Stipulates how to deal with
                eventual conflicts with other channels, specifically in terms
                of having to channels act on the same target simultaneously.

                - 'min-delay'
                    Before adding the pulse, introduces the smallest
                    possible delay that avoids all exisiting conflicts.
                - 'no-delay'
                    Adds the pulse to the channel, regardless of
                    existing conflicts.
                - 'wait-for-all'
                    Before adding the pulse, adds a delay that
                    idles the channel until the end of the other channels'
                    latest pulse.
        """

        last = self._last(channel)
        self._validate_pulse(pulse, channel)

        valid_protocols = ['min-delay', 'no-delay', 'wait-for-all']
        if protocol not in valid_protocols:
            raise ValueError(f"Invalid protocol '{protocol}', only accepts "
                             "protocols: " + ", ".join(valid_protocols))

        t0 = last.tf    # Preliminary ti
        basis = self._channels[channel].basis
        phase_barriers = [self._phase_ref[basis][q].last_time
                          for q in last.targets]
        current_max_t = max(t0, *phase_barriers)
        if protocol != 'no-delay':
            for ch, seq in self._schedule.items():
                if ch == channel:
                    continue
                for op in self._schedule[ch][::-1]:
                    if op.tf <= current_max_t:
                        break
                    if not isinstance(op.type, Pulse):
                        continue
                    if op.targets & last.targets or protocol == 'wait-for-all':
                        current_max_t = op.tf
                        break
        ti = current_max_t
        tf = ti + pulse.duration
        if ti > t0:
            self.delay(ti-t0, channel)

        prs = {self._phase_ref[basis][q].last_phase for q in last.targets}
        if len(prs) != 1:
            raise ValueError("Cannot do a multiple-target pulse on qubits "
                             "with different phase references for the same "
                             "basis.")
        else:
            phase_ref = prs.pop()

        if phase_ref != 0:
            # Has to copy to keep the original pulse intact
            pulse = copy.deepcopy(pulse)
            pulse.phase = (pulse.phase + phase_ref) % (2 * np.pi)

        self._add_to_schedule(channel, _TimeSlot(pulse, ti, tf, last.targets))

        for q in last.targets:
            if self._last_used[basis][q] < tf:
                self._last_used[basis][q] = tf

        if pulse.post_phase_shift:
            self.phase_shift(pulse.post_phase_shift, *last.targets,
                             basis=basis)

    def target(self, qubits, channel):
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits (hashable, iterable): The new target for this channel. Must
                correspond to a qubit ID in device or an iterable of qubit IDs,
                when multi-qubit adressing is possible.
            channel (str): The channel's name provided when declared. Must be
                a channel with 'Local' addressing.
         """

        if channel not in self._channels:
            raise ValueError("Use the name of a declared channel.")

        if isinstance(qubits, Iterable) and not isinstance(qubits, str):
            qs = set(qubits)
        else:
            qs = {qubits}

        if not qs.issubset(self._qids):
            raise ValueError("The given qubits have to belong to the device.")

        if self._channels[channel].addressing != 'Local':
            raise ValueError("Can only choose target of 'Local' channels.")
        elif len(qs) > self._channels[channel].max_targets:
            raise ValueError(
                "This channel can target at most "
                f"{self._channels[channel].max_targets} qubits at a time"
            )

        basis = self._channels[channel].basis
        phase_refs = {self._phase_ref[basis][q].last_phase for q in qs}
        if len(phase_refs) != 1:
            raise ValueError("Cannot target multiple qubits with different "
                             "phase references for the same basis.")

        try:
            last = self._last(channel)
            if last.targets == qs:
                warnings.warn("The provided qubits are already the target. "
                              "Skipping this target instruction.")
                return
            ti = last.tf
            retarget = self._channels[channel].retarget_time
            elapsed = ti - self._last_target[channel]
            delta = np.clip(retarget - elapsed, 0, retarget)
            if delta != 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    delta = validate_duration(np.clip(delta, 16, np.inf))
            tf = ti + delta

        except ValueError:
            ti = -1
            tf = 0

        self._last_target[channel] = tf
        self._add_to_schedule(channel, _TimeSlot('target', ti, tf, qs))

    def delay(self, duration, channel):
        """Idles a given channel for a specific duration.

        Args:
            duration (int): Time to delay (in multiples of 4 ns).
            channel (str): The channel's name provided when declared.
        """
        last = self._last(channel)
        ti = last.tf
        tf = ti + validate_duration(duration)
        self._add_to_schedule(channel,
                              _TimeSlot('delay', ti, tf, last.targets))

    def measure(self, basis='ground-rydberg'):
        """Measures in a valid basis.

        Args:
            basis (str): Valid basis for measurement (consult the
                'supported_bases' attribute of the selected device for
                the available options).
        """
        available = self._device.supported_bases
        if basis not in available:
            raise ValueError(f"The basis '{basis}' is not supported by the "
                             "selected device. The available options are: "
                             + ", ".join(list(available)))

        if hasattr(self, '_measurement'):
            raise SystemError("The sequence has already been measured.")

        self._measurement = basis

    def phase_shift(self, phi, *targets, basis='digital'):
        r"""Shifts the phase of a qubit's reference by 'phi', for a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi (float): The intended phase shift (in rads).
            targets: The ids of the qubits on which to apply the phase
                shift.

        Keyword Args:
            basis(str): The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.
        """
        if phi % (2*np.pi) == 0:
            warnings.warn("A phase shift of 0 is meaningless, "
                          "it will be ommited.")
            return
        if not set(targets) <= self._qids:
            raise ValueError("All given targets have to be qubit ids declared"
                             " in this sequence's device.")

        if basis not in self._phase_ref:
            raise ValueError("No declared channel targets the given 'basis'.")

        for q in targets:
            t = self._last_used[basis][q]
            new_phase = self._phase_ref[basis][q].last_phase + phi
            self._phase_ref[basis][q][t] = new_phase

    def align(self, *channels):
        """Aligns multiple channels in time.

        Introduces delays that align the provided channels with the one that
        finished the latest, such that the next action added to any of them
        will start right after the latest channel has finished.

        Args:
            channels (str): The names of the channels to align, as given upon
                declaration.
        """

        ch_set = set(channels)
        # channels have to be a subset of the declared channels
        if not ch_set <= set(self._channels):
            raise ValueError("All channel names must correspond to declared"
                             " channels.")
        if len(channels) != len(ch_set):
            raise ValueError("The same channel was provided more than once.")

        if len(channels) < 2:
            raise ValueError("Needs at least two channels for alignment.")

        last_ts = {id: self._last(id).tf for id in channels}
        tf = max(last_ts.values())

        for id in channels:
            delta = tf - last_ts[id]
            if delta > 0:
                self.delay(delta, id)

    def draw(self):
        """Draws the sequence in its current sequence."""
        draw_sequence(self)

    def __str__(self):
        full = ""
        pulse_line = "t: {}->{} | {} | Targets: {}\n"
        target_line = "t: {}->{} | Target: {} | Phase Reference: {}\n"
        delay_line = "t: {}->{} | Delay \n"
        # phase_line = "t: {} | Phase shift of: {:.3f} | Targets: {}\n"
        for ch, seq in self._schedule.items():
            basis = self._channels[ch].basis
            full += f"Channel: {ch}\n"
            first_slot = True
            for ts in seq:
                if ts.type == 'delay':
                    full += delay_line.format(ts.ti, ts.tf)
                    continue

                tgts = list(ts.targets)
                tgt_txt = ", ".join([str(t) for t in tgts])
                if isinstance(ts.type, Pulse):
                    full += pulse_line.format(ts.ti, ts.tf, ts.type, tgt_txt)
                elif ts.type == 'target':
                    phase = self._phase_ref[basis][tgts[0]][ts.tf]
                    if first_slot:
                        full += (f"t: 0 | Initial targets: {tgt_txt} | " +
                                 f"Phase Reference: {phase} \n")
                        first_slot = False
                    else:
                        full += target_line.format(ts.ti, ts.tf, tgt_txt,
                                                   phase)

            full += "\n"

        if hasattr(self, "_measurement"):
            full += f"Measured in basis: {self._measurement}"

        return full

    def _add_to_schedule(self, channel, timeslot):
        if hasattr(self, "_measurement"):
            raise SystemError("The sequence has already been measured. "
                              "Nothing more can be added.")
        self._schedule[channel].append(timeslot)

    def _last(self, channel):
        """Shortcut to last element in the channel's schedule."""
        if channel not in self._schedule:
            raise ValueError("Use the name of a declared channel.")
        try:
            return self._schedule[channel][-1]
        except IndexError:
            raise ValueError("The chosen channel has no target.")

    def _validate_pulse(self, pulse, channel):
        if not isinstance(pulse, Pulse):
            raise TypeError("pulse input must be of type Pulse, not of type "
                            "{}.".format(type(pulse)))

        ch = self._channels[channel]
        if np.any(pulse.amplitude.samples > ch.max_amp):
            raise ValueError("The pulse's amplitude goes over the maximum "
                             "value allowed for the chosen channel.")
        if np.any(np.round(np.abs(pulse.detuning.samples),
                           decimals=6) > ch.max_abs_detuning):
            raise ValueError("The pulse's detuning values go out of the range "
                             "allowed for the chosen channel.")


class _PhaseTracker:
    """Tracks a phase reference over time."""

    def __init__(self, initial_phase):
        self._times = [0]
        self._phases = [self._format(initial_phase)]

    @property
    def last_time(self):
        return self._times[-1]

    @property
    def last_phase(self):
        return self._phases[-1]

    def changes(self, ti, tf, time_scale=1):
        """Changes in phases within ]ti, tf]."""
        start, end = np.searchsorted(
                self._times, (ti * time_scale, tf * time_scale), side='right')
        for i in range(start, end):
            change = self._phases[i] - self._phases[i-1]
            yield (self._times[i] / time_scale, change)

    def _format(self, phi):
        return phi % (2 * np.pi)

    def __setitem__(self, t, phi):
        phase = self._format(phi)
        if t in self._times:
            ind = self._times.index(t)
            self._phases[ind] = phase
        else:
            ind = np.searchsorted(self._times, t, side='right')
            self._times.insert(ind, t)
            self._phases.insert(ind, phase)

    def __getitem__(self, t):
        ind = np.searchsorted(self._times, t, side='right') - 1
        return self._phases[ind]
