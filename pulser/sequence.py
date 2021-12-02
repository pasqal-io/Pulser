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
"""The Sequence class, where a pulse sequence is defined."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Generator, Iterable
import copy
from functools import wraps
from itertools import chain
import json
from sys import version_info
from typing import Any, cast, NamedTuple, Optional, Tuple, Union
import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import pulser
from pulser.channels import Channel
from pulser.devices import MockDevice
from pulser.devices._device_datacls import Device
from pulser.json.coders import PulserEncoder, PulserDecoder
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, Variable
from pulser.pulse import Pulse
from pulser.register import BaseRegister
from pulser._seq_drawer import draw_sequence

if version_info[:2] >= (3, 8):  # pragma: no cover
    from typing import Literal, get_args
else:  # pragma: no cover
    try:
        from typing_extensions import Literal, get_args  # type: ignore
    except ImportError:
        raise ImportError(
            "Using pulser with Python version 3.7 requires the"
            " `typing_extensions` module. Install it by running"
            " `pip install typing-extensions`."
        )


QubitId = Union[int, str]
PROTOCOLS = Literal["min-delay", "no-delay", "wait-for-all"]


class _TimeSlot(NamedTuple):
    """Auxiliary class to store the information in the schedule."""

    type: Union[Pulse, str]
    ti: int
    tf: int
    targets: set[QubitId]


# Encodes a sequence building calls
_Call = namedtuple("_Call", ["name", "args", "kwargs"])


def _screen(func: Callable) -> Callable:
    """Blocks the call to a function if the Sequence is parametrized."""

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        if self.is_parametrized():
            raise RuntimeError(
                f"Sequence.{func.__name__} can't be called in"
                " parametrized sequences."
            )
        return func(self, *args, **kwargs)

    return wrapper


def _store(func: Callable) -> Callable:
    """Stores any Sequence building call for deferred execution."""

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        def verify_variable(x: Any) -> None:
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
            raise RuntimeError(
                "The sequence has been measured, no further "
                "changes are allowed."
            )
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
        register(BaseRegister): The atom register on which to apply the pulses.
        device(Device): A valid device in which to execute the Sequence (import
            it from ``pulser.devices``).

    Note:
        The register and device do not support variable parameters. As such,
        they are the same for all Sequences built from a parametrized Sequence.
    """

    def __init__(self, register: BaseRegister, device: Device):
        """Initializes a new pulse sequence."""
        if not isinstance(device, Device):
            raise TypeError(
                "'device' must be of type 'Device'. Import a valid"
                " device from 'pulser.devices'."
            )
        cond1 = device not in pulser.devices._valid_devices
        cond2 = device not in pulser.devices._mock_devices
        if cond1 and cond2:
            names = [d.name for d in pulser.devices._valid_devices]
            warns_msg = (
                "The Sequence's device should be imported from "
                + "'pulser.devices'. Correct operation is not ensured"
                + " for custom devices. Choose 'MockDevice'"
                + " or one of the following real devices:\n"
                + "\n".join(names)
            )
            warnings.warn(warns_msg, stacklevel=2)

        # Checks if register is compatible with the device
        device.validate_register(register)

        self._register: BaseRegister = register
        self._device: Device = device
        self._in_xy: bool = False
        self._mag_field: Optional[tuple[float, float, float]] = None
        self._calls: list[_Call] = [_Call("__init__", (register, device), {})]
        self._channels: dict[str, Channel] = {}
        self._schedule: dict[str, list[_TimeSlot]] = {}
        # The phase reference of each channel
        self._phase_ref: dict[str, dict[QubitId, _PhaseTracker]] = {}
        # Stores the names and dict ids of declared channels
        self._taken_channels: dict[str, str] = {}
        # IDs of all qubits in device
        self._qids: set[QubitId] = set(self.qubit_info.keys())
        # Last time each qubit was used, by basis
        self._last_used: dict[str, dict[QubitId, int]] = {}
        # Last time a target happened, by channel
        self._last_target: dict[str, int] = {}
        self._variables: dict[str, Variable] = {}
        self._to_build_calls: list[_Call] = []
        self._building: bool = True
        # Marks the sequence as empty until the first pulse is added
        self._empty_sequence: bool = True
        # SLM mask targets and on/off times
        self._slm_mask_targets: set[QubitId] = set()
        self._slm_mask_time: list[int] = []

        # Initializes all parametrized Sequence related attributes
        self._reset_parametrized()

    @property
    def qubit_info(self) -> dict[QubitId, np.ndarray]:
        """Dictionary with the qubit's IDs and positions."""
        return self._register.qubits

    @property
    def declared_channels(self) -> dict[str, Channel]:
        """Channels declared in this Sequence."""
        return dict(self._channels)

    @property
    def declared_variables(self) -> dict[str, Variable]:
        """Variables declared in this Sequence."""
        return dict(self._variables)

    @property
    def available_channels(self) -> dict[str, Channel]:
        """Channels still available for declaration."""
        # Show all channels if none are declared, otherwise filter depending
        # on whether the sequence is working on XY mode
        # If already in XY mode, filter right away
        if not self._channels and not self._in_xy:
            return dict(self._device.channels)
        else:
            # MockDevice channels can be declared multiple times
            return {
                id: ch
                for id, ch in self._device.channels.items()
                if (
                    id not in self._taken_channels.values()
                    or self._device == MockDevice
                )
                and (ch.basis == "XY" if self._in_xy else ch.basis != "XY")
            }

    @property
    def magnetic_field(self) -> np.ndarray:
        """The magnetic field acting on the array of atoms.

        The magnetic field vector is defined on the reference frame of the
        atoms in the Register (with the z-axis coming outside of the plane).

        Note:
            Only defined in "XY Mode", the default value being (0, 0, 30) G.
        """
        if not self._in_xy:
            raise AttributeError(
                "The magnetic field is only defined when the "
                "sequence is in 'XY Mode'."
            )
        return np.array(self._mag_field)

    def is_parametrized(self) -> bool:
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
    def get_duration(self, channel: Optional[str] = None) -> int:
        """Returns the current duration of a channel or the whole sequence.

        Keyword Args:
            channel (Optional[str]): A specific channel to return the duration
                of. If left as None, it will return the duration of the whole
                sequence.

        Returns:
            int: The duration of the channel or sequence, in ns.
        """
        if channel is None:
            durations = [
                self._last(ch).tf
                for ch in self._schedule
                if self._schedule[ch]
            ]
            return 0 if not durations else max(durations)

        self._validate_channel(channel)
        return self._last(channel).tf if self._schedule[channel] else 0

    @_screen
    def current_phase_ref(
        self, qubit: QubitId, basis: str = "digital"
    ) -> float:
        """Current phase reference of a specific qubit for a given basis.

        Args:
            qubit (Union[int, str]): The id of the qubit whose phase shift is
                desired.
            basis (str): The basis (i.e. electronic transition) the phase
                reference is associated with. Must correspond to the basis of a
                declared channel.

        Returns:
            float: Current phase reference of 'qubit' in 'basis'.
        """
        if qubit not in self._qids:
            raise ValueError(
                "'qubit' must be the id of a qubit declared in "
                "this sequence's device."
            )

        if basis not in self._phase_ref:
            raise ValueError("No declared channel targets the given 'basis'.")

        return self._phase_ref[basis][qubit].last_phase

    def set_magnetic_field(
        self, bx: float = 0.0, by: float = 0.0, bz: float = 30.0
    ) -> None:
        """Sets the magnetic field acting on the entire array.

        The magnetic field vector is defined on the reference frame of the
        atoms in the Register (with the z-axis coming outside of the plane).
        Can only be defined before there are pulses added to the sequence.

        Note:
            The magnetic field only work in the "XY Mode". If not already
            defined through the declaration of a Microwave channel, calling
            this function will enable the "XY Mode".

        Keyword Args:
            bx (float): The magnetic field in the x direction (in Gauss).
            by (float): The magnetic field in the y direction (in Gauss).
            bz (float): The magnetic field in the z direction (in Gauss).
        """
        if not self._in_xy:
            if self._channels:
                raise ValueError(
                    "The magnetic field can only be set in 'XY " "Mode'."
                )
            # No channels declared yet
            self._in_xy = True
        elif not self._empty_sequence:
            # Not all channels are empty
            raise ValueError(
                "The magnetic field can only be set on an empty " "sequence."
            )

        mag_vector = (bx, by, bz)
        if np.linalg.norm(mag_vector) == 0.0:
            raise ValueError(
                "The magnetic field must have a magnitude greater" " than 0."
            )
        self._mag_field = mag_vector

        # No parametrization -> Always stored as a regular call
        self._calls.append(_Call("set_magnetic_field", mag_vector, {}))

    @_store
    def config_slm_mask(self, qubits: Iterable[QubitId]) -> None:
        """Setup an SLM mask by specifying the qubits it targets.

        Args:
            qubits (Iterable[QubitId]): Iterable of qubit ID's to mask during
                the first global pulse of the sequence.
        """
        try:
            targets = set(qubits)
        except TypeError:
            raise TypeError("The SLM targets must be castable to set")

        if not targets.issubset(self._qids):
            raise ValueError("SLM mask targets must exist in the register")

        if self.is_parametrized():
            return

        if self._slm_mask_targets:
            raise ValueError("SLM mask can be configured only once.")

        # If checks have passed, set the SLM mask targets
        self._slm_mask_targets = targets

        # Find tentative initial and final time of SLM mask if possible
        for channel in self._channels:
            if not self._channels[channel].addressing == "Global":
                continue
            # Cycle on slots in schedule until the first pulse is found
            for slot in self._schedule[channel]:
                if not isinstance(slot.type, Pulse):
                    continue
                ti = slot.ti
                tf = slot.tf
                if self._slm_mask_time:
                    if ti < self._slm_mask_time[0]:
                        self._slm_mask_time = [ti, tf]
                else:
                    self._slm_mask_time = [ti, tf]
                break

    def declare_channel(
        self,
        name: str,
        channel_id: str,
        initial_target: Optional[
            Union[
                Iterable[Union[QubitId, Parametrized]],
                Union[QubitId, Parametrized],
            ]
        ] = None,
    ) -> None:
        """Declares a new channel to the Sequence.

        The first declared channel implicitly defines the sequence's mode of
        operation (i.e. the underlying Hamiltonian). In particular, if the
        first declared channel is of type ``Microwave``, the sequence will work
        in "XY Mode" and will not allow declaration of channels that do not
        address the 'XY' basis. Inversely, declaration of a channel of another
        type will block the declaration of ``Microwave`` channels.

        Note:
            Regular devices only allow a channel to be declared once, but
            ``MockDevice`` channels can be repeatedly declared if needed.

        Args:
            name (str): Unique name for the channel in the sequence.
            channel_id (str): How the channel is identified in the device.
                Consult ``Sequence.available_channels`` to see which channel
                ID's are still available and the associated channel's
                description.
            initial_target (Optional[Union[int, str, Iterable]]): For 'Local'
                addressing channels only. Declares the initial target of the
                channel. If left as None, the initial target will have to be
                set manually as the first addition to this channel.
        """
        if name in self._channels:
            raise ValueError("The given name is already in use.")

        if channel_id not in self._device.channels:
            raise ValueError("No channel %s in the device." % channel_id)

        ch = self._device.channels[channel_id]
        if channel_id not in self.available_channels:
            if self._in_xy and ch.basis != "XY":
                raise ValueError(
                    f"Channel '{ch}' cannot work simultaneously "
                    "with the declared 'Microwave' channel."
                )
            elif not self._in_xy and ch.basis == "XY":
                raise ValueError(
                    "Channel of type 'Microwave' cannot work "
                    "simultaneously with the declared channels."
                )
            else:
                raise ValueError(f"Channel {channel_id} is not available.")

        if ch.basis == "XY" and not self._in_xy:
            self._in_xy = True
            self.set_magnetic_field()
        self._channels[name] = ch
        self._taken_channels[name] = channel_id
        self._schedule[name] = []
        self._last_target[name] = 0

        if ch.basis not in self._phase_ref:
            self._phase_ref[ch.basis] = {
                q: _PhaseTracker(0) for q in self._qids
            }
            self._last_used[ch.basis] = {q: 0 for q in self._qids}

        if ch.addressing == "Global":
            self._add_to_schedule(name, _TimeSlot("target", -1, 0, self._qids))
        elif initial_target is not None:
            try:
                cond = any(
                    isinstance(t, Parametrized)
                    for t in cast(Iterable, initial_target)
                )
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
                self._target(
                    cast(Union[Iterable, QubitId], initial_target), name
                )

        # Manually store the channel declaration as a regular call
        self._calls.append(
            _Call(
                "declare_channel",
                (name, channel_id),
                {"initial_target": initial_target},
            )
        )

    def declare_variable(
        self,
        name: str,
        size: int = 1,
        dtype: Union[type[int], type[float], type[str]] = float,
    ) -> Variable:
        """Declare a new variable within this Sequence.

        The declared variables can be used to create parametrized versions of
        ``Waveform`` and ``Pulse`` objects, which in turn can be added to the
        ``Sequence``. Additionally, simple arithmetic operations involving
        variables are also supported and will return parametrized objects that
        are dependent on the involved variables.

        Args:
            name (str): The name for the variable. Must be unique within a
                Sequence.

        Keyword Args:
            size (int=1): The number of entries stored in the variable.
            dtype (default=float): The type of the data that will be assigned
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
    def add(
        self,
        pulse: Union[Pulse, Parametrized],
        channel: Union[str, Parametrized],
        protocol: PROTOCOLS = "min-delay",
    ) -> None:
        """Adds a pulse to a channel.

        Args:
            pulse (pulser.Pulse): The pulse object to add to the channel.
            channel (str): The channel's name provided when declared.
            protocol (str, default='min-delay'): Stipulates how to deal with
                eventual conflicts with other channels, specifically in terms
                of having multiple channels act on the same target
                simultaneously.

                - ``'min-delay'``: Before adding the pulse, introduces the
                  smallest possible delay that avoids all exisiting conflicts.

                - ``'no-delay'``: Adds the pulse to the channel, regardless of
                  existing conflicts.

                - ``'wait-for-all'``: Before adding the pulse, adds a delay
                  that idles the channel until the end of the other channels'
                  latest pulse.
        """
        pulse = cast(Pulse, pulse)
        channel = cast(str, channel)

        self._validate_channel(channel)

        valid_protocols = get_args(PROTOCOLS)
        if protocol not in valid_protocols:
            raise ValueError(
                f"Invalid protocol '{protocol}', only accepts protocols: "
                + ", ".join(valid_protocols)
            )

        if self.is_parametrized():
            if not isinstance(pulse, Parametrized):
                self._validate_pulse(pulse, channel)
            # Sequence is marked as non-empty on the first added pulse
            if self._empty_sequence:
                self._empty_sequence = False
            return

        if not isinstance(pulse, Pulse):
            raise TypeError(
                f"'pulse' must be of type Pulse, not of type {type(pulse)}."
            )

        channel_obj = self._channels[channel]
        _duration = channel_obj.validate_duration(pulse.duration)
        if _duration != pulse.duration:
            try:
                pulse = Pulse(
                    pulse.amplitude.change_duration(_duration),
                    pulse.detuning.change_duration(_duration),
                    pulse.phase,
                    pulse.post_phase_shift,
                )
            except NotImplementedError:
                raise TypeError(
                    "Failed to automatically adjust one of the pulse's "
                    "waveforms to the channel duration constraints. Choose a "
                    "duration that is a multiple of "
                    f"{channel_obj.clock_period} ns."
                )

        self._validate_pulse(pulse, channel)
        last = self._last(channel)
        t0 = last.tf  # Preliminary ti
        basis = channel_obj.basis
        phase_barriers = [
            self._phase_ref[basis][q].last_time for q in last.targets
        ]
        current_max_t = max(t0, *phase_barriers)
        if protocol != "no-delay":
            for ch, seq in self._schedule.items():
                if ch == channel:
                    continue
                for op in self._schedule[ch][::-1]:
                    if op.tf <= current_max_t:
                        break
                    if not isinstance(op.type, Pulse):
                        continue
                    if op.targets & last.targets or protocol == "wait-for-all":
                        current_max_t = op.tf
                        break
        ti = current_max_t
        if ti > t0:
            # Insert a delay
            delay_duration = ti - t0

            # Delay must not be shorter than the min duration for this channel
            min_duration = self._channels[channel].min_duration
            if delay_duration < min_duration:
                ti += min_duration - delay_duration
                delay_duration = min_duration

            self._delay(delay_duration, channel)

        tf = ti + pulse.duration

        prs = {self._phase_ref[basis][q].last_phase for q in last.targets}
        if len(prs) != 1:
            raise ValueError(
                "Cannot do a multiple-target pulse on qubits with different "
                "phase references for the same basis."
            )
        else:
            phase_ref = prs.pop()

        if phase_ref != 0:
            # Has to recriate the original pulse with a new phase
            pulse = Pulse(
                pulse.amplitude,
                pulse.detuning,
                pulse.phase + phase_ref,
                post_phase_shift=pulse.post_phase_shift,
            )

        self._add_to_schedule(channel, _TimeSlot(pulse, ti, tf, last.targets))

        for qubit in last.targets:
            if self._last_used[basis][qubit] < tf:
                self._last_used[basis][qubit] = tf

        if pulse.post_phase_shift:
            self._phase_shift(
                pulse.post_phase_shift, *last.targets, basis=basis
            )

        # Sequence is marked as non-empty on the first added pulse
        if self._empty_sequence:
            self._empty_sequence = False

        # If the added pulse starts earlier than all previously added pulses,
        # update SLM mask initial and final time
        if self._slm_mask_targets:
            try:
                if self._slm_mask_time[0] > ti:
                    self._slm_mask_time = [ti, tf]
            except IndexError:
                self._slm_mask_time = [ti, tf]

    @_store
    def target(
        self,
        qubits: Union[QubitId, Iterable[QubitId], Parametrized],
        channel: Union[str, Parametrized],
    ) -> None:
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits (Union[int, str, Iterable]): The new target for this
                channel. Must correspond to a qubit ID in device or an iterable
                of qubit IDs, when multi-qubit addressing is possible.
            channel (str): The channel's name provided when declared. Must be
                a channel with 'Local' addressing.
        """
        qubits = cast(QubitId, qubits)
        channel = cast(str, channel)

        self._target(qubits, channel)

    @_store
    def delay(
        self,
        duration: Union[int, Parametrized],
        channel: Union[str, Parametrized],
    ) -> None:
        """Idles a given channel for a specific duration.

        Args:
            duration (int): Time to delay (in multiples of 4 ns).
            channel (str): The channel's name provided when declared.
        """
        duration = cast(int, duration)
        channel = cast(str, channel)

        self._delay(duration, channel)

    @_store
    def measure(
        self, basis: Union[str, Parametrized] = "ground-rydberg"
    ) -> None:
        """Measures in a valid basis.

        Note:
            In addition to the supported bases of the selected device, allowed
            measurement bases will depend on the mode of operation. In
            particular, if using ``Microwave`` channels (XY mode), only
            measuring in the 'XY' basis is allowed. Inversely, it is not
            possible to measure in the 'XY' basis outside of XY mode.

        Args:
            basis (str): Valid basis for measurement (consult the
                ``supported_bases`` attribute of the selected device for
                the available options).
        """
        available = (
            self._device.supported_bases - {"XY"}
            if not self._in_xy
            else {"XY"}
        )
        if basis not in available:
            raise ValueError(
                f"The basis '{basis}' is not supported by the "
                "selected device and operation mode. The "
                "available options are: " + ", ".join(list(available))
            )

        if hasattr(self, "_measurement"):
            raise RuntimeError("The sequence has already been measured.")

        if self.is_parametrized():
            self._is_measured = True
        else:
            self._measurement = basis

    @_store
    def phase_shift(
        self,
        phi: Union[float, Parametrized],
        *targets: Union[QubitId, Parametrized],
        basis: Union[str, Parametrized] = "digital",
    ) -> None:
        r"""Shifts the phase of a qubit's reference by 'phi', for a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi (float): The intended phase shift (in rads).
            targets (Union[int, str]): The ids of the qubits to apply the phase
                shift to.
            basis (str): The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.
        """
        phi = cast(float, phi)
        basis = cast(str, basis)
        targets = cast(Tuple[QubitId], targets)

        self._phase_shift(phi, *targets, basis=basis)

    @_store
    def align(self, *channels: Union[str, Parametrized]) -> None:
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
            raise ValueError(
                "All channel names must correspond to declared" " channels."
            )
        if len(channels) != len(ch_set):
            raise ValueError("The same channel was provided more than once.")

        if len(channels) < 2:
            raise ValueError("Needs at least two channels for alignment.")

        if self.is_parametrized():
            return

        last_ts = {id: self._last(cast(str, id)).tf for id in channels}
        tf = max(last_ts.values())

        for id in channels:
            delta = tf - last_ts[id]
            if delta > 0:
                self._delay(delta, cast(str, id))

    def build(self, **vars: Union[ArrayLike, float, int, str]) -> Sequence:
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
            warnings.warn(
                "Building a non-parametrized sequence simply returns"
                " a copy of itself.",
                stacklevel=2,
            )
            return copy.copy(self)
        all_keys, given_keys = self._variables.keys(), vars.keys()
        if given_keys != all_keys:
            invalid_vars = given_keys - all_keys
            if invalid_vars:
                warnings.warn(
                    "No declared variables named: " + ", ".join(invalid_vars),
                    stacklevel=2,
                )
                for k in invalid_vars:
                    vars.pop(k, None)
            missing_vars = all_keys - given_keys
            if missing_vars:
                raise TypeError(
                    "Did not receive values for variables: "
                    + ", ".join(missing_vars)
                )

        for name, value in vars.items():
            self._variables[name]._assign(value)

        # Shallow copy with stored parametrized objects
        seq = copy.copy(self)
        # Eliminates the source of recursiveness errors
        seq._reset_parametrized()
        # Deepcopy the base sequence (what remains)
        seq = copy.deepcopy(seq)

        for call in self._to_build_calls:
            args_ = [
                arg.build() if isinstance(arg, Parametrized) else arg
                for arg in call.args
            ]
            kwargs_ = {
                key: val.build() if isinstance(val, Parametrized) else val
                for key, val in call.kwargs.items()
            }
            getattr(seq, call.name)(*args_, **kwargs_)

        return seq

    def serialize(self, **kwargs: Any) -> str:
        """Serializes the Sequence into a JSON formatted string.

        Other Parameters:
            kwargs: Valid keyword-arguments for ``json.dumps()``, except for
                ``cls``.

        Returns:
            str: The sequence encoded in a JSON formatted string.

        See Also:
            ``json.dumps``: Built-in function for serialization to a JSON
            formatted string.
        """
        return json.dumps(self, cls=PulserEncoder, **kwargs)

    @staticmethod
    def deserialize(obj: str, **kwargs: Any) -> Sequence:
        """Deserializes a JSON formatted string.

        Args:
            obj (str): The JSON formatted string to deserialize, coming from
                the serialization of a ``Sequence`` through
                ``Sequence.serialize()``.

        Other Parameters:
            kwargs: Valid keyword-arguments for ``json.loads()``, except for
                ``cls`` and ``object_hook``.

        Returns:
            Sequence: The deserialized Sequence object.

        See Also:
            ``json.loads``: Built-in function for deserialization from a JSON
            formatted string.
        """
        if "Sequence" not in obj:
            warnings.warn(
                "The given JSON formatted string does not encode a Sequence.",
                stacklevel=2,
            )

        return cast(Sequence, json.loads(obj, cls=PulserDecoder, **kwargs))

    @_screen
    def draw(
        self,
        draw_phase_area: bool = False,
        draw_interp_pts: bool = True,
        draw_phase_shifts: bool = False,
        draw_register: bool = False,
        fig_name: str = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the sequence in its current state.

        Keyword Args:
            draw_phase_area (bool): Whether phase and area values need to be
                shown as text on the plot, defaults to False.
            draw_interp_pts (bool): When the sequence has pulses with waveforms
                of type InterpolatedWaveform, draws the points of interpolation
                on top of the respective waveforms (defaults to True).
            draw_phase_shifts (bool): Whether phase shift and reference
                information should be added to the plot, defaults to False.
            draw_register (bool): Whether to draw the register before the pulse
                sequence, with a visual indication (square halo) around the
                qubits masked by the SLM, defaults to False.
            fig_name(str, default=None): The name on which to save the
                figure. If `draw_register` is True, both pulses and register
                will be saved as figures, with a suffix ``_pulses`` and
                ``_register`` in the file name. If `draw_register` is False,
                only the pulses are saved, with no suffix. If `fig_name` is
                None, no figure is saved.
            kwargs_savefig(dict, default={}): Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        See Also:
            Simulation.draw(): Draws the provided sequence and the one used by
            the solver.
        """
        fig_reg, fig = draw_sequence(
            self,
            draw_phase_area=draw_phase_area,
            draw_interp_pts=draw_interp_pts,
            draw_phase_shifts=draw_phase_shifts,
            draw_register=draw_register,
        )
        if fig_name is not None and draw_register:
            name, ext = os.path.splitext(fig_name)
            fig.savefig(name + "_pulses" + ext, **kwargs_savefig)
            fig_reg.savefig(name + "_register" + ext, **kwargs_savefig)
        elif fig_name:
            fig.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def _target(
        self, qubits: Union[Iterable[QubitId], QubitId], channel: str
    ) -> None:
        self._validate_channel(channel)

        try:
            qubits_set = (
                set(cast(Iterable, qubits))
                if not isinstance(qubits, str)
                else {qubits}
            )
        except TypeError:
            qubits_set = {qubits}

        if self._channels[channel].addressing != "Local":
            raise ValueError("Can only choose target of 'Local' channels.")
        elif len(qubits_set) > cast(int, self._channels[channel].max_targets):
            raise ValueError(
                "This channel can target at most "
                f"{self._channels[channel].max_targets} qubits at a time"
            )

        if self.is_parametrized():
            for q in qubits_set:
                if q not in self._qids and not isinstance(q, Parametrized):
                    raise ValueError(
                        "All non-variable qubits must belong to the register."
                    )
            return

        elif not qubits_set.issubset(self._qids):
            raise ValueError("All given qubits must belong to the register.")

        basis = self._channels[channel].basis
        phase_refs = {self._phase_ref[basis][q].last_phase for q in qubits_set}
        if len(phase_refs) != 1:
            raise ValueError(
                "Cannot target multiple qubits with different "
                "phase references for the same basis."
            )

        try:
            last = self._last(channel)
            if last.targets == qubits_set:
                return
            ti = last.tf
            retarget = cast(int, self._channels[channel].retarget_time)
            elapsed = ti - self._last_target[channel]
            delta = cast(int, np.clip(retarget - elapsed, 0, retarget))
            if delta != 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    delta = self._channels[channel].validate_duration(
                        16 if delta < 16 else delta
                    )
            tf = ti + delta

        except ValueError:
            ti = -1
            tf = 0

        self._last_target[channel] = tf
        self._add_to_schedule(channel, _TimeSlot("target", ti, tf, qubits_set))

    def _delay(self, duration: int, channel: str) -> None:
        self._validate_channel(channel)
        if self.is_parametrized():
            return

        last = self._last(channel)
        ti = last.tf
        tf = ti + self._channels[channel].validate_duration(duration)
        self._add_to_schedule(
            channel, _TimeSlot("delay", ti, tf, last.targets)
        )

    def _phase_shift(
        self, phi: float, *targets: QubitId, basis: str = "digital"
    ) -> None:
        if basis not in self._phase_ref:
            raise ValueError("No declared channel targets the given 'basis'.")
        if self.is_parametrized():
            for t in targets:
                if t not in self._qids and not isinstance(t, Parametrized):
                    raise ValueError(
                        "All non-variable targets must belong to the register."
                    )
            return

        elif not set(targets) <= self._qids:
            raise ValueError(
                "All given targets have to be qubit ids declared"
                " in this sequence's register."
            )

        if phi % (2 * np.pi) == 0:
            return

        for qubit in targets:
            last_used = self._last_used[basis][qubit]
            new_phase = self._phase_ref[basis][qubit].last_phase + phi
            self._phase_ref[basis][qubit][last_used] = new_phase

    def _to_dict(self) -> dict[str, Any]:
        d = obj_to_dict(self, *self._calls[0].args, **self._calls[0].kwargs)
        d["__version__"] = pulser.__version__
        d["calls"] = self._calls[1:]
        d["vars"] = self._variables
        d["to_build_calls"] = self._to_build_calls
        return d

    def __str__(self) -> str:
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
                if ts.type == "delay":
                    full += delay_line.format(ts.ti, ts.tf)
                    continue

                tgts = list(ts.targets)
                tgt_txt = ", ".join([str(t) for t in tgts])
                if isinstance(ts.type, Pulse):
                    full += pulse_line.format(ts.ti, ts.tf, ts.type, tgt_txt)
                elif ts.type == "target":
                    phase = self._phase_ref[basis][tgts[0]][ts.tf]
                    if first_slot:
                        full += (
                            f"t: 0 | Initial targets: {tgt_txt} | "
                            + f"Phase Reference: {phase} \n"
                        )
                        first_slot = False
                    else:
                        full += target_line.format(
                            ts.ti, ts.tf, tgt_txt, phase
                        )
            full += "\n"

        if hasattr(self, "_measurement"):
            full += f"Measured in basis: {self._measurement}"

        if self.is_parametrized():
            prelude = "Prelude\n-------\n" + full
            lines = ["Stored calls\n------------"]
            for i, c in enumerate(self._to_build_calls, 1):
                args = [str(a) for a in c.args]
                kwargs = [
                    f"{key}={str(value)}" for key, value in c.kwargs.items()
                ]
                lines.append(f"{i}. {c.name}({', '.join(args+kwargs)})")
            full = prelude + "\n\n".join(lines)

        return full

    def _add_to_schedule(self, channel: str, timeslot: _TimeSlot) -> None:
        if hasattr(self, "_measurement"):
            raise RuntimeError(
                "The sequence has already been measured. "
                "Nothing more can be added."
            )
        self._schedule[channel].append(timeslot)

    def _min_pulse_duration(self) -> float:
        duration_list = []
        for ch_schedule in self._schedule.values():
            for slot in ch_schedule:
                if isinstance(slot.type, Pulse):
                    duration_list.append(slot.tf - slot.ti)
        return min(duration_list)

    def _last(self, channel: str) -> _TimeSlot:
        """Shortcut to last element in the channel's schedule."""
        try:
            return self._schedule[channel][-1]
        except IndexError:
            raise ValueError("The chosen channel has no target.")

    def _validate_channel(self, channel: str) -> None:
        if channel not in self._channels:
            raise ValueError("Use the name of a declared channel.")

    def _validate_pulse(self, pulse: Pulse, channel: str) -> None:
        self._device.validate_pulse(pulse, self._taken_channels[channel])

    def _reset_parametrized(self) -> None:
        """Resets all attributes related to parametrization."""
        # Signals the sequence as actively "building" ie not parametrized
        self._building = True
        self._is_measured = False
        self._variables = {}
        self._to_build_calls = []


class _PhaseTracker:
    """Tracks a phase reference over time."""

    def __init__(self, initial_phase: float):
        self._times: list[int] = [0]
        self._phases: list[float] = [self._format(initial_phase)]

    @property
    def last_time(self) -> int:
        return self._times[-1]

    @property
    def last_phase(self) -> float:
        return self._phases[-1]

    def changes(
        self,
        ti: Union[float, int],
        tf: Union[float, int],
        time_scale: float = 1.0,
    ) -> Generator[tuple[float, float], None, None]:
        """Changes in phases within ]ti, tf]."""
        start, end = np.searchsorted(
            self._times, (ti * time_scale, tf * time_scale), side="right"
        )
        for i in range(start, end):
            change = self._phases[i] - self._phases[i - 1]
            yield (self._times[i] / time_scale, change)

    def _format(self, phi: float) -> float:
        return phi % (2 * np.pi)

    def __setitem__(self, t: int, phi: float) -> None:
        phase = self._format(phi)
        if t in self._times:
            ind = self._times.index(t)
            self._phases[ind] = phase
        else:
            ind = int(np.searchsorted(self._times, t, side="right"))
            self._times.insert(ind, t)
            self._phases.insert(ind, phase)

    def __getitem__(self, t: int) -> float:
        ind = int(np.searchsorted(self._times, t, side="right")) - 1
        return self._phases[ind]
