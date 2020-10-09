import warnings
import numpy as np
from collections import namedtuple

from devices import PasqalDevice
from pulses import Pulse
from utils import validate_duration

# Auxiliary class to store the information in the schedule
TimeSlot = namedtuple('TimeSlot', ['type', 'ti', 'tf', 'targets'])


class Sequence:
    """A sequence of operations on a device.

    A sequence is composed by
        - The device in which we want to implement it
        - The device's channels that are used
        - The schedule of operations on each channel
    """
    def __init__(self, device):
        if not isinstance(device, PasqalDevice):
            raise TypeError("The Sequence's device has to be a PasqalDevice.")
        self._device = device
        self._channels = {}
        self._schedule = {}
        self._taken_channels = []   # Stores the ids of selected channels
        self._qids = set(self.qubit_info.keys())  # IDs of all qubits in device

    @property
    def qubit_info(self):
        """Returns the dictionary with the qubit's IDs and positions."""
        return self._device.qubits

    @property
    def declared_channels(self):
        return dict(self._channels)

    @property
    def available_channels(self):
        return {id: ch for id, ch in self._device.channels.items()
                if id not in self._taken_channels}

    def declare_channel(self, name, channel_id, initial_target=None):
        """Declare a new channel to the Sequence.

        Args:
            name (str): Unique name for the channel in the sequence.
            channel_id (str): How the channel is identified in the device.

        Keyword Args:
            initial_target (set, default=None): For 'local' adressing channels
                where a target has to be defined, it can be done when the
                channel is first declared. If left as None, this will have to
                be done manually as the first addition to a channel.
        """

        if name in self._channels:
            raise ValueError("The given name is already in use.")

        if channel_id not in self._device.channels:
            raise ValueError("No channel %s in the device." % channel_id)

        if channel_id in self._taken_channels:
            raise ValueError("Channel %s has already been added." % channel_id)

        ch = self._device.channels[channel_id]
        self._channels[name] = ch
        self._taken_channels.append(channel_id)
        self._schedule[name] = []

        if ch.addressing == 'global':
            self._schedule[name].append(TimeSlot('target', -1, 0, self._qids))
        elif initial_target is not None:
            self.target(initial_target, name)

    def add(self, pulse, channel, protocol='min-delay'):
        """Add a pulse to a channel.

        Args:
            pulse (Pulse): The pulse object to add to the channel.
            channel (str): The channel's name provided when declared.

        Keyword Args:
            protocol (default='min-delay'): Stipulates how to deal with
                eventual conflicts with other channels, specifically in terms
                of having to channels act on the same target simultaneously.
                'min-delay': Before adding the pulse, introduces the smallest
                    possible delay that avoids all exisiting conflicts.
                'no-delay': Adds the pulse to the channel, regardless of
                    existing conflicts.
                'wait-for-all': Before adding the pulse, adds a delay that
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
        current_max_t = t0  # Stores the maximum tf found so far
        if protocol != 'no-delay':
            for ch, seq in self._schedule.items():
                if ch == channel:
                    continue
                for op in self._schedule[ch][::-1]:
                    if op.tf <= current_max_t:
                        break
                    if op.type in ['delay', 'target']:
                        continue
                    if op.targets & last.targets or protocol == 'wait-for-all':
                        current_max_t = op.tf
                        break

        ti = current_max_t
        tf = ti + pulse.duration
        if ti > t0:
            self.delay(ti-t0, channel)

        self._schedule[channel].append(TimeSlot(pulse, ti, tf, last.targets))

    def target(self, qubits, channel):
        """Changes the target qubit of a 'local' channel.

        Args:
            qubits (set(str)): The new target for this channel.
            channel (str): The channel's name provided when declared.
        """

        if channel not in self._channels:
            raise ValueError("Use the name of a declared channel.")

        qs = {qubits}
        if not qs.issubset(self._qids):
            raise ValueError("The given qubits have to belong to the device.")

        if self._channels[channel].addressing != 'local':
            raise ValueError("Can only choose target of 'local' channels.")
        elif len(qs) != 1:
            raise ValueError("This channel takes only a single target qubit.")

        try:
            last = self._last(channel)
            if last.targets == qs:
                warnings.warn("The provided qubits are already the target. "
                              "Skipping this target instruction.")
                return
            ti = last.tf
            tf = ti + self._channels[channel].retarget_time
        except ValueError:
            ti = -1
            tf = 0

        self._schedule[channel].append(TimeSlot('target', ti, tf, qs))

    def delay(self, duration, channel):
        """Idle a given choosen for a specific duration.

        Args:
            duration (int): Time to delay (in ns).
            channel (str): The channel's name provided when declared.
        """
        last = self._last(channel)
        ti = last.tf
        tf = ti + validate_duration(duration)
        self._schedule[channel].append(TimeSlot('delay', ti, tf, last.targets))

    def __str__(self):
        full = ""
        #header = "Time (ns) \t Element \t Targets\n"
        line = "t: {}->{} | {} | Targets: {}\n"
        for ch, seq in self._schedule.items():
            full += f"Channel: {ch}\n"
            #full += header
            for ts in seq:
                full += line.format(ts.ti, ts.tf, ts.type, ts.targets)
            full += "\n"

        return full

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
        if np.any(np.abs(pulse.detuning.samples) > ch.max_abs_detuning):
            raise ValueError("The pulse's detuning values go out of the range "
                             "allowed for the chosen channel.")
