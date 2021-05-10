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
from functools import wraps
from itertools import chain
import json
import warnings

import numpy as np

import pulser
from pulser.pulse import Pulse
from pulser.devices import MockDevice
from pulser.devices._device_datacls import Device
from pulser.json.coders import PulserEncoder, PulserDecoder
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, Variable
from pulser._seq_drawer import draw_sequence

# Auxiliary class to store the information in the schedule
_TimeSlot = namedtuple('_TimeSlot', ['type', 'ti', 'tf', 'targets'])
# Encodes a sequence building calls
_Call = namedtuple("_Call", ['name', 'args', 'kwargs'])


def _screen(func):
    """Blocks the call to a function if the Sequence is parametrized."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_parametrized():
            raise RuntimeError(f"Sequence.{func.__name__} can't be called in"
                               + " parametrized sequences.")
        return func(self, *args, **kwargs)
    return wrapper


def _store(func):
    """Stores any Sequence building call for deferred execution."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        def verify_variable(x):
            if isinstance(x, Parametrized):
                # If not already, the sequence becomes parametrized
                self._building = False
                for name, var in x.variables.items():
                    if name not in self._variables:
                        raise ValueError(f"Unknown variable '{name}'.")
                    elif self._variables[name] is not var:
                        raise ValueError(
                            f"{x} has variables that don't come from this "
                            "Sequence. Use only what's returned by this"
                            "Sequence's 'declare_variable' method as your"
                            "variables."
                            )
            elif isinstance(x, Iterable) and not isinstance(x, str):
                # Recursively look for parametrized objs inside the arguments
                for y in x:
                    verify_variable(y)

        if self._is_measured and self.is_parametrized():
            raise SystemError("The sequence has been measured, no further "
                              "changes are allowed.")
        # Check if all Parametrized inputs stem from declared variables
        for x in chain(args, kwargs.values()):
            verify_variable(x)
        storage = self._calls if self._building else self._to_build_calls
        func(self, *args, **kwargs)
        storage.append(_Call(func.__name__, args, kwargs))
    return wrapper


class Sequence:
    """A sequence of operations on a device.

    A sequence is composed by

        - The device in which we want to implement it
        - The register of qubits on which to act
        - The device's channels that are used
        - The schedule of operations on each channel


    A Sequence also supports variable parameters, which have to be obtained
    through ``Sequence.declare_variable()``. From the moment a variable is
    declared, a ``Sequence`` becomes **parametrized** and stops being built on
    the fly, instead storing the sequence building calls for later execution.
    This forgoes some specific functionalities of a "regular" ``Sequence``,
    like the ability to validate a ``Pulse`` or to draw the sequence as it is
    being built. Instead, all validation happens upon building (through
    ``Sequence.build()``), where values for all declared variables have to be
    specified and a "regular" ``Sequence`` is created and returned. By
    changing the values given to the variables, multiple sequences can be
    generated from a single "parametrized" ``Sequence``.

    Args:
        register(Register): The atom register on which to apply the pulses.
        device(Device): A valid device in which to execute the Sequence (import
            it from ``pulser.devices``).

    Note:
        The register and device do not support variable parameters. As such,
        they are the same for all Sequences built from a parametrized Sequence.
    """
    def __init__(self, register, device):
        """Initializes a new pulse sequence."""
        if not isinstance(device, Device):
            raise TypeError("'device' must be of type 'Device'. Import a valid"
                            " device from 'pulser.devices'.")
        cond1 = device not in pulser.devices._valid_devices
        cond2 = device != MockDevice
        if cond1 and cond2:
            names = [d.name for d in pulser.devices._valid_devices]
            warns_msg = ("The Sequence's device should be imported from "
                         + "'pulser.devices'. Correct operation is not ensured"
                         + " for custom devices. Choose 'MockDevice' or one of"
                         + " the following real devices:\n" + "\n".join(names))
            warnings.warn(warns_msg)

        # Checks if register is compatible with the device
        device.validate_register(register)

        self._register = register
        self._device = device
        self._in_xy = False
        self._calls = [_Call("__init__", (register, device), {})]
        self._channels = {}
        self._schedule = {}
        self._phase_ref = {}  # The phase reference of each channel
        # Stores the names and corresponding ids of declared channels
        self._taken_channels = {}
        self._qids = set(self.qubit_info.keys())  # IDs of all qubits in device
        self._last_used = {}    # Last time each qubit was used, by basis
        self._last_target = {}  # Last time a target happened, by channel

        # Initializes all parametrized Sequence related attributes
        self._reset_parametrized()

    @property
    def qubit_info(self):
        """Dictionary with the qubit's IDs and positions."""
        return self._register.qubits

    @property
    def declared_channels(self):
        """Channels declared in this Sequence."""
        return dict(self._channels)

    @property
    def declared_variables(self):
        """Variables declared in this Sequence."""
        return dict(self._variables)

    @property
    def available_channels(self):
        """Channels still available for declaration."""
        # Show all channels if none are declared, otherwise filter depending
        # on whether the sequence is working on XY mode
        if not self._channels:
            return dict(self._device.channels)
        else:
            return {id: ch for id, ch in self._device.channels.items()
                    if (id not in self._taken_channels.values()
                    or self._device == MockDevice)
                    and (ch.basis == "xy" if self._in_xy else ch.basis != "xy")
                    }

    def is_parametrized(self):
        """States whether the sequence is parametrized.

        A parametrized sequence is one that depends on the values assigned to
        variables declared within it. Sequence-building calls are not executed
        right away, but rather stored for deferred execution when all variables
        are given a value (when ``Sequence.build()`` is called).

        Returns:
            bool: Whether the sequence is parametrized.
        """
        return not self._building

    @_screen
    def current_phase_ref(self, qubit, basis='digital'):
        """Current phase reference of a specific qubit for a given basis.

        Args:
            qubit (hashable): The id of the qubit whose phase shift is desired.

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
            initial_target (set, default=None): For 'Local' addressing channels
                only. Declares the initial target of the channel. If left as
                None, the initial target will have to be set manually as the
                first addition to this channel.
        """

        if name in self._channels:
            raise ValueError("The given name is already in use.")

        if channel_id not in self._device.channels:
            raise ValueError("No channel %s in the device." % channel_id)

        ch = self._device.channels[channel_id]
        if channel_id not in self.available_channels:
            if self._in_xy and ch.basis != "xy":
                raise ValueError(f"Channel '{ch}' cannot work simultaneously "
                                 "with the declared 'Microwave' channel."
                                 )
            elif not self._in_xy and ch.basis == "xy":
                raise ValueError("Channel of type 'Microwave' cannot work "
                                 "simultaneously with the declared channels.")
            else:
                raise ValueError(f"Channel {channel_id} is not available.")

        if ch.basis == "xy" and not self._in_xy:
            self._in_xy = True
        self._channels[name] = ch
        self._taken_channels[name] = channel_id
        self._schedule[name] = []
        self._last_target[name] = 0

        if ch.basis not in self._phase_ref:
            self._phase_ref[ch.basis] = {q: _PhaseTracker(0)
                                         for q in self._qids}
            self._last_used[ch.basis] = {q: 0 for q in self._qids}

        if ch.addressing == 'Global':
            self._add_to_schedule(name, _TimeSlot('target', -1, 0, self._qids))
        elif initial_target is not None:
            try:
                cond = any(isinstance(t, Parametrized) for t in initial_target)
            except TypeError:
                cond = isinstance(initial_target, Parametrized)
            if cond:
                self._building = False

            if self.is_parametrized():
                # Do not store "initial_target" in a _call when parametrized
                # It is stored as a _to_build_call when target is called
                self.target(initial_target, name)
                initial_target = None
            else:
                # "_target" call is not saved
                self._target(initial_target, name)

        # Manually store the channel declaration as a regular call
        self._calls.append(_Call(
                            "declare_channel",
                            (name, channel_id),
                            {"initial_target": initial_target}))

    def declare_variable(self, name, size=1, dtype=float):
        """Declare a new variable within this Sequence.

        The declared variables can be used to create parametrized versions of
        ``Waveform`` and ``Pulse`` objects, which in turn can be added to the
        ``Sequence``. Additionally, simple arithmetic operations involving
        variables are also supported and will return parametrized objects that
        are dependent on the involved variables.

        Args:
            name(str): The name for the variable. Must be unique within a
                Sequence.

        Keyword Args:
            size(int=1): The number of entries stored in the variable.
            dtype(default=float): The type of the data that will be assigned
                to the variable. Must be ``float``, ``int`` or ``str``.

        Returns:
            Variable: The declared Variable instance.

        Note:
            To avoid confusion, it is recommended to store the returned
            Variable instance in a Python variable with the same name.
        """
        if name in self._variables:
            raise ValueError("Name for variable is already being used.")
        var = Variable(name, dtype, size=size)
        self._variables[name] = var
        return var

    @_store
    def add(self, pulse, channel, protocol='min-delay'):
        """Adds a pulse to a channel.

        Args:
            pulse (pulser.Pulse): The pulse object to add to the channel.
            channel (str): The channel's name provided when declared.

        Keyword Args:
            protocol (default='min-delay'): Stipulates how to deal with
                eventual conflicts with other channels, specifically in terms
                of having multiple channels act on the same target
                simultaneously.

                - ``'min-delay'``
                    Before adding the pulse, introduces the smallest
                    possible delay that avoids all exisiting conflicts.
                - ``'no-delay'``
                    Adds the pulse to the channel, regardless of
                    existing conflicts.
                - ``'wait-for-all'``
                    Before adding the pulse, adds a delay that
                    idles the channel until the end of the other channels'
                    latest pulse.
        """
        self._validate_channel(channel)

        valid_protocols = ['min-delay', 'no-delay', 'wait-for-all']
        if protocol not in valid_protocols:
            raise ValueError(f"Invalid protocol '{protocol}', only accepts "
                             "protocols: " + ", ".join(valid_protocols))

        if self.is_parametrized():
            if not isinstance(pulse, Parametrized):
                self._validate_pulse(pulse, channel)
            return

        if not isinstance(pulse, Pulse):
            raise TypeError("pulse input must be of type Pulse, not of type "
                            "{}.".format(type(pulse)))

        channel_obj = self._channels[channel]
        _duration = channel_obj.validate_duration(pulse.duration)
        if _duration != pulse.duration:
            try:
                pulse = Pulse(pulse.amplitude.change_duration(_duration),
                              pulse.detuning.change_duration(_duration),
                              pulse.phase,
                              pulse.post_phase_shift)
            except NotImplementedError:
                raise TypeError("Failed to automatically adjust one of the "
                                "pulse's waveforms to the channel duration "
                                "constraints. Choose a duration that is a "
                                f"multiple of {channel_obj.clock_period} ns.")

        self._validate_pulse(pulse, channel)
        last = self._last(channel)
        t0 = last.tf    # Preliminary ti
        basis = channel_obj.basis
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
            self._delay(ti-t0, channel)

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
            self._phase_shift(pulse.post_phase_shift, *last.targets,
                              basis=basis)

    @_store
    def target(self, qubits, channel):
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits (hashable, iterable): The new target for this channel. Must
                correspond to a qubit ID in device or an iterable of qubit IDs,
                when multi-qubit addressing is possible.
            channel (str): The channel's name provided when declared. Must be
                a channel with 'Local' addressing.
         """
        self._target(qubits, channel)

    @_store
    def delay(self, duration, channel):
        """Idles a given channel for a specific duration.

        Args:
            duration (int): Time to delay (in multiples of 4 ns).
            channel (str): The channel's name provided when declared.
        """
        self._delay(duration, channel)

    @_store
    def measure(self, basis='ground-rydberg'):
        """Measures in a valid basis.

        Args:
            basis (str): Valid basis for measurement (consult the
                ``supported_bases`` attribute of the selected device for
                the available options).
        """
        available = self._device.supported_bases
        if basis not in available:
            raise ValueError(f"The basis '{basis}' is not supported by the "
                             "selected device. The available options are: "
                             + ", ".join(list(available)))

        if hasattr(self, "_measurement"):
            raise SystemError("The sequence has already been measured.")

        if self.is_parametrized():
            self._is_measured = True
        else:
            self._measurement = basis

    @_store
    def phase_shift(self, phi, *targets, basis='digital'):
        r"""Shifts the phase of a qubit's reference by 'phi', for a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi (float): The intended phase shift (in rads).
            targets (hashable): The ids of the qubits on which to apply the
                phase shift.

        Keyword Args:
            basis(str): The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.
        """
        self._phase_shift(phi, *targets, basis=basis)

    @_store
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

        if self.is_parametrized():
            return

        last_ts = {id: self._last(id).tf for id in channels}
        tf = max(last_ts.values())

        for id in channels:
            delta = tf - last_ts[id]
            if delta > 0:
                self._delay(delta, id)

    def build(self, **vars):
        """Builds a sequence from the programmed instructions.

        Keyword Args:
            vars: The values for all the variables declared in this Sequence
                instance, indexed by the name given upon declaration. Check
                ``Sequence.declared_variables`` to see all the variables.

        Returns:
            Sequence: The Sequence built with the given variable values.

        Example:
            ::

                # Check which variables are declared
                >>> print(seq.declared_variables)
                {'x': Variable(name='x', dtype=<class 'float'>, size=1),
                 'y': Variable(name='y', dtype=<class 'int'>, size=3)}
                # Build a sequence with specific values for both variables
                >>> seq1 = seq.build(x=0.5, y=[1, 2, 3])
        """
        if not self.is_parametrized():
            warnings.warn("Building a non-parametrized sequence simply returns"
                          " a copy of itself.")
            return copy.copy(self)
        all_keys, given_keys = self._variables.keys(), vars.keys()
        if given_keys != all_keys:
            invalid_vars = given_keys - all_keys
            if invalid_vars:
                warnings.warn("No declared variables named: "
                              + ", ".join(invalid_vars))
                for k in invalid_vars:
                    vars.pop(k, None)
            missing_vars = all_keys - given_keys
            if missing_vars:
                raise TypeError("Did not receive values for variables: "
                                + ", ".join(missing_vars))

        for name, value in vars.items():
            self._variables[name]._assign(value)

        # Shallow copy with stored parametrized objects
        seq = copy.copy(self)
        # Eliminates the source of recursiveness errors
        seq._reset_parametrized()
        # Deepcopy the base sequence (what remains)
        seq = copy.deepcopy(seq)

        for call in self._to_build_calls:
            args_ = [arg.build() if isinstance(arg, Parametrized) else arg
                     for arg in call.args]
            kwargs_ = {key: val.build() if isinstance(val, Parametrized)
                       else val for key, val in call.kwargs.items()}
            getattr(seq, call.name)(*args_, **kwargs_)

        return seq

    def serialize(self, **kwargs):
        """Serializes the Sequence into a JSON formatted string.

        Other Parameters:
            kwargs: Valid keyword-arguments for ``json.dumps()``, except for
                ``cls``.

        Returns:
            str: The sequence encoded in a JSON formatted string.

        See also:
            ``json.dumps``: Built-in function for serialization to a JSON
            formatted string.
        """
        return json.dumps(self, cls=PulserEncoder, **kwargs)

    @staticmethod
    def deserialize(obj, **kwargs):
        """Deserializes a JSON formatted string.

        Args:
            obj(str): The JSON formatted string to deserialize, coming from the
                serialization of a ``Sequence`` through
                ``Sequence.serialize()``.

        Other Parameters:
            kwargs: Valid keyword-arguments for ``json.loads()``, except for
                ``cls`` and ``object_hook``.

        Returns:
            Sequence: The deserialized Sequence object.

        See also:
            ``json.loads``: Built-in function for deserialization from a JSON
            formatted string.
        """
        if "Sequence" not in obj:
            warnings.warn("The given JSON formatted string does not encode a "
                          "Sequence.")

        return json.loads(obj, cls=PulserDecoder, **kwargs)

    @_screen
    def draw(self):
        """Draws the sequence in its current state."""
        draw_sequence(self)

    def _target(self, qubits, channel):
        self._validate_channel(channel)

        try:
            qs = set(qubits) if not isinstance(qubits, str) else {qubits}
        except TypeError:
            qs = {qubits}

        if self._channels[channel].addressing != 'Local':
            raise ValueError("Can only choose target of 'Local' channels.")
        elif len(qs) > self._channels[channel].max_targets:
            raise ValueError(
                "This channel can target at most "
                f"{self._channels[channel].max_targets} qubits at a time"
            )

        if self.is_parametrized():
            for q in qs:
                if q not in self._qids and not isinstance(q, Parametrized):
                    raise ValueError("All non-variable qubits must belong to "
                                     "the register.")
            return

        elif not qs.issubset(self._qids):
            raise ValueError("All given qubits must belong to the register.")

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
                    delta = self._channels[channel].validate_duration(
                        np.clip(delta, 16, np.inf)
                        )
            tf = ti + delta

        except ValueError:
            ti = -1
            tf = 0

        self._last_target[channel] = tf
        self._add_to_schedule(channel, _TimeSlot('target', ti, tf, qs))

    def _delay(self, duration, channel):
        self._validate_channel(channel)
        if self.is_parametrized():
            return

        last = self._last(channel)
        ti = last.tf
        tf = ti + self._channels[channel].validate_duration(duration)
        self._add_to_schedule(channel,
                              _TimeSlot('delay', ti, tf, last.targets))

    def _phase_shift(self, phi, *targets, basis='digital'):
        if basis not in self._phase_ref:
            raise ValueError("No declared channel targets the given 'basis'.")
        if self.is_parametrized():
            for t in targets:
                if t not in self._qids and not isinstance(t, Parametrized):
                    raise ValueError("All non-variable targets must belong to "
                                     "the register.")
            return

        elif not set(targets) <= self._qids:
            raise ValueError("All given targets have to be qubit ids declared"
                             " in this sequence's register.")

        if phi % (2*np.pi) == 0:
            warnings.warn("A phase shift of 0 is meaningless, "
                          "it will be ommited.")
            return

        for q in targets:
            t = self._last_used[basis][q]
            new_phase = self._phase_ref[basis][q].last_phase + phi
            self._phase_ref[basis][q][t] = new_phase

    def _to_dict(self):
        d = obj_to_dict(self, *self._calls[0].args, **self._calls[0].kwargs)
        d["__version__"] = pulser.__version__
        d["calls"] = self._calls[1:]
        d["vars"] = self._variables
        d["to_build_calls"] = self._to_build_calls
        return d

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

        if self.is_parametrized():
            prelude = "Prelude\n-------\n" + full
            lines = ["Stored calls\n------------"]
            for i, c in enumerate(self._to_build_calls, 1):
                args = [str(a) for a in c.args]
                kwargs = [f"{key}={str(value)}"
                          for key, value in c.kwargs.items()]
                lines.append(f"{i}. {c.name}({', '.join(args+kwargs)})")
            full = prelude + "\n\n".join(lines)

        return full

    def _add_to_schedule(self, channel, timeslot):
        if hasattr(self, "_measurement"):
            raise SystemError("The sequence has already been measured. "
                              "Nothing more can be added.")
        self._schedule[channel].append(timeslot)

    def _last(self, channel):
        """Shortcut to last element in the channel's schedule."""
        try:
            return self._schedule[channel][-1]
        except IndexError:
            raise ValueError("The chosen channel has no target.")

    def _validate_channel(self, channel):
        if channel not in self._channels:
            raise ValueError("Use the name of a declared channel.")

    def _validate_pulse(self, pulse, channel):
        self._device.validate_pulse(pulse, self._taken_channels[channel])

    def _reset_parametrized(self):
        """Resets all attributes related to parametrization."""
        # Signals the sequence as actively "building" ie not parametrized
        self._building = True
        self._is_measured = False
        self._variables = {}
        self._to_build_calls = []


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
