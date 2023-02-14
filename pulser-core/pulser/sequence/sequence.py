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

import copy
import json
import os
import warnings
from collections.abc import Iterable, Mapping
from sys import version_info
from typing import Any, Optional, Tuple, Union, cast, overload

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import pulser
import pulser.sequence._decorators as seq_decorators
from pulser.channels.base_channel import Channel
from pulser.channels.eom import RydbergEOM
from pulser.devices._device_datacls import BaseDevice
from pulser.json.abstract_repr.deserializer import (
    deserialize_abstract_sequence,
)
from pulser.json.abstract_repr.serializer import serialize_abstract_sequence
from pulser.json.coders import PulserDecoder, PulserEncoder
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, Variable
from pulser.parametrized.variable import VariableItem
from pulser.pulse import Pulse
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.mappable_reg import MappableRegister
from pulser.sequence._basis_ref import _QubitRef
from pulser.sequence._call import _Call
from pulser.sequence._schedule import _ChannelSchedule, _Schedule, _TimeSlot
from pulser.sequence._seq_drawer import Figure, draw_sequence
from pulser.sequence._seq_str import seq_to_str

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


PROTOCOLS = Literal["min-delay", "no-delay", "wait-for-all"]


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
        register: The atom register on which to apply the pulses. If given as
            a MappableRegister instance, the traps corrresponding to each
            qubit ID must be given when building the sequence.
        device: A valid device in which to execute the Sequence (import
            it from ``pulser.devices``).

    Note:
        The register and device do not support variable parameters. As such,
        they are the same for all Sequences built from a parametrized Sequence.
    """

    def __init__(
        self,
        register: Union[BaseRegister, MappableRegister],
        device: BaseDevice,
    ):
        """Initializes a new pulse sequence."""
        if not isinstance(device, BaseDevice):
            raise TypeError(
                f"'device' must be of type 'BaseDevice', not {type(device)}."
            )

        # Checks if register is compatible with the device
        if isinstance(register, MappableRegister):
            device.validate_layout(register.layout)
            device.validate_layout_filling(register)
        else:
            device.validate_register(register)

        self._register: Union[BaseRegister, MappableRegister] = register
        self._device: BaseDevice = device
        self._in_xy: bool = False
        self._mag_field: Optional[tuple[float, float, float]] = None
        self._calls: list[_Call] = [
            _Call("__init__", (), {"register": register, "device": device})
        ]
        self._schedule: _Schedule = _Schedule()
        self._basis_ref: dict[str, dict[QubitId, _QubitRef]] = {}
        # IDs of all qubits in device
        self._qids: set[QubitId] = set(self._register.qubit_ids)
        # Last time each qubit was used, by basis
        self._variables: dict[str, Variable] = {}
        self._to_build_calls: list[_Call] = []
        self._building: bool = True
        # Marks the sequence as empty until the first pulse is added
        self._empty_sequence: bool = True
        # SLM mask targets and on/off times
        self._slm_mask_targets: set[QubitId] = set()

        # Initializes all parametrized Sequence related attributes
        self._reset_parametrized()

    @property
    def _slm_mask_time(self) -> list[int]:
        """The initial and final time when the SLM mask is on."""
        return (
            []
            if not self._slm_mask_targets
            else self._schedule.find_slm_mask_times()
        )

    @property
    def qubit_info(self) -> dict[QubitId, np.ndarray]:
        """Dictionary with the qubit's IDs and positions."""
        if self.is_register_mappable():
            raise RuntimeError(
                "Can't access the qubit information when the register is "
                "mappable."
            )
        return cast(BaseRegister, self._register).qubits

    @property
    def device(self) -> BaseDevice:
        """Device that the sequence is using."""
        return self._device

    @property
    def register(self) -> BaseRegister:
        """Register with the qubit's IDs and positions."""
        if self.is_register_mappable():
            raise RuntimeError(
                "Can't access the sequence's register because the register "
                "is mappable."
            )
        return cast(BaseRegister, self._register)

    @property
    def declared_channels(self) -> dict[str, Channel]:
        """Channels declared in this Sequence."""
        return {name: cs.channel_obj for name, cs in self._schedule.items()}

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
        if not self._schedule and not self._in_xy:
            return dict(self._device.channels)
        else:
            # MockDevice channels can be declared multiple times
            occupied_ch_ids = [cs.channel_id for cs in self._schedule.values()]
            return {
                id: ch
                for id, ch in self._device.channels.items()
                if (
                    id not in occupied_ch_ids or self._device.reusable_channels
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
            Whether the sequence is parametrized.
        """
        return not self._building

    def is_in_eom_mode(self, channel: str) -> bool:
        """States whether a channel is currently in EOM mode.

        Args:
            channel: The name of the declared channel to inspect.

        Returns:
            Whether the channel is in EOM mode.

        """
        self._validate_channel(channel)
        if not self.is_parametrized():
            return self._schedule[channel].in_eom_mode()

        # Look for the latest stored EOM mode enable/disable
        for call in reversed(self._calls + self._to_build_calls):
            if call.name not in ("enable_eom_mode", "disable_eom_mode"):
                continue
            # Channel is the first positional arg in both methods
            ch_arg = call.args[0] if call.args else call.kwargs["channel"]
            if ch_arg == channel:
                # If it's not "enable_eom_mode", then it's "disable_eom_mode"
                return cast(bool, call.name == "enable_eom_mode")
        # If it reaches here, there were no EOM calls found
        return False

    def is_register_mappable(self) -> bool:
        """States whether the sequence's register is mappable.

        A sequence with a mappable register will require its qubit Id's to be
        mapped to trap Ids of its associated RegisterLayout through the
        `Sequence.build()` call.

        Returns:
            Whether the register is a MappableRegister.
        """
        return isinstance(self._register, MappableRegister)

    def is_measured(self) -> bool:
        """States whether the sequence has been measured."""
        return (
            self._is_measured
            if self.is_parametrized()
            else hasattr(self, "_measurement")
        )

    @seq_decorators.screen
    def get_duration(
        self, channel: Optional[str] = None, include_fall_time: bool = False
    ) -> int:
        """Returns the current duration of a channel or the whole sequence.

        Args:
            channel: A specific channel to return the duration of. If left as
                None, it will return the duration of the whole sequence.
            include_fall_time: Whether to include in the duration the
                extra time needed by the last pulse to finish, if there is
                modulation.

        Returns:
            The duration of the channel or sequence, in ns.
        """
        if channel is not None:
            self._validate_channel(channel)

        return self._schedule.get_duration(channel, include_fall_time)

    @seq_decorators.screen
    def current_phase_ref(
        self, qubit: QubitId, basis: str = "digital"
    ) -> float:
        """Current phase reference of a specific qubit for a given basis.

        Args:
            qubit: The id of the qubit whose phase shift is desired.
            basis: The basis (i.e. electronic transition) the phase
                reference is associated with. Must correspond to the basis of a
                declared channel.

        Returns:
            Current phase reference of 'qubit' in 'basis'.
        """
        if qubit not in self._qids:
            raise ValueError(
                "'qubit' must be the id of a qubit declared in "
                "this sequence's register."
            )

        if basis not in self._basis_ref:
            raise ValueError("No declared channel targets the given 'basis'.")

        return self._basis_ref[basis][qubit].phase.last_phase

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

        Args:
            bx: The magnetic field in the x direction (in Gauss).
            by: The magnetic field in the y direction (in Gauss).
            bz: The magnetic field in the z direction (in Gauss).
        """
        if not self._in_xy:
            if self._schedule:
                raise ValueError(
                    "The magnetic field can only be set in 'XY Mode'."
                )
            # No channels declared yet
            self._in_xy = True
        elif not self._empty_sequence:
            # Not all channels are empty
            raise ValueError(
                "The magnetic field can only be set on an empty sequence."
            )

        mag_vector = (bx, by, bz)
        if np.linalg.norm(mag_vector) == 0.0:
            raise ValueError(
                "The magnetic field must have a magnitude greater than 0."
            )
        self._mag_field = mag_vector

        # No parametrization -> Always stored as a regular call
        self._calls.append(_Call("set_magnetic_field", mag_vector, {}))

    @seq_decorators.store
    def config_slm_mask(self, qubits: Iterable[QubitId]) -> None:
        """Setup an SLM mask by specifying the qubits it targets.

        Args:
            qubits: Iterable of qubit ID's to mask during the first global
                pulse of the sequence.
        """
        if not self._device.supports_slm_mask:
            raise ValueError(
                f"The '{self._device}' device does not have an SLM mask."
            )
        try:
            targets = set(qubits)
        except TypeError:
            raise TypeError("The SLM targets must be castable to set.")

        if not targets.issubset(self._qids):
            raise ValueError("SLM mask targets must exist in the register.")

        if self.is_parametrized():
            return

        if self._slm_mask_targets:
            raise ValueError("SLM mask can be configured only once.")

        # If checks have passed, set the SLM mask targets
        self._slm_mask_targets = targets

    @seq_decorators.screen
    def switch_device(
        self, new_device: BaseDevice, strict: bool = False
    ) -> Sequence:
        """Switch the device of a sequence.

        Args:
            new_device: The target device instance.
            strict: Enforce a strict match between devices and channels to
                guarantee the pulse sequence is left unchanged.

        Returns:
            The sequence on the new device, using the match channels of
            the former device declared in the sequence.
        """
        # Check if the device is new or not

        if self._device == new_device:
            warnings.warn(
                "Switching a sequence to the same device"
                + " returns the sequence unchanged.",
                stacklevel=2,
            )
            return self

        if new_device.rydberg_level != self._device.rydberg_level:
            if strict:
                raise ValueError(
                    "Strict device match failed because the"
                    + " devices have different Rydberg levels."
                )
            warnings.warn(
                "Switching to a device with a different Rydberg level,"
                " check that the expected Rydberg interactions still hold.",
                stacklevel=2,
            )

        # Channel match
        channel_match: dict[str, Any] = {}
        strict_error_message = ""
        ch_type_er_mess = ""
        for o_d_ch_name, o_d_ch_obj in self.declared_channels.items():
            channel_match[o_d_ch_name] = None
            for n_d_ch_id, n_d_ch_obj in new_device.channels.items():
                if (
                    not new_device.reusable_channels
                    and n_d_ch_id in channel_match.values()
                ):
                    # Channel already matched and can't be reused
                    continue
                # Find the corresponding channel on the new device
                # We verify the channel class then
                # check whether the addressing Global or local
                basis_match = o_d_ch_obj.basis == n_d_ch_obj.basis
                addressing_match = (
                    o_d_ch_obj.addressing == n_d_ch_obj.addressing
                )
                base_msg = f"No match for channel {o_d_ch_name}"
                if not (basis_match and addressing_match):
                    # If there already is a message, keeps it
                    ch_type_er_mess = ch_type_er_mess or (
                        base_msg + " with the right basis and addressing."
                    )
                    continue
                if self._schedule[o_d_ch_name].eom_blocks:
                    if n_d_ch_obj.eom_config is None:
                        ch_type_er_mess = (
                            base_msg + " with an EOM configuration."
                        )
                        continue
                    if (
                        n_d_ch_obj.eom_config != o_d_ch_obj.eom_config
                        and strict
                    ):
                        strict_error_message = (
                            base_msg + " with the same EOM configuration."
                        )
                        continue
                if not strict:
                    channel_match[o_d_ch_name] = n_d_ch_id
                    break
                if n_d_ch_obj.mod_bandwidth != o_d_ch_obj.mod_bandwidth:
                    strict_error_message = strict_error_message or (
                        base_msg + " with the same mod_bandwidth."
                    )
                    continue
                if n_d_ch_obj.fixed_retarget_t != o_d_ch_obj.fixed_retarget_t:
                    strict_error_message = strict_error_message or (
                        base_msg + " with the same fixed_retarget_t."
                    )
                    continue

                # Clock_period check
                if o_d_ch_obj.clock_period == n_d_ch_obj.clock_period:
                    channel_match[o_d_ch_name] = n_d_ch_id
                    break
                strict_error_message = strict_error_message or (
                    base_msg + " with the same clock_period."
                )

        if None in channel_match.values():
            if strict_error_message:
                raise ValueError(strict_error_message)
            else:
                raise TypeError(ch_type_er_mess)
        # Initialize the new sequence (works for Sequence subclasses too)
        new_seq = type(self)(register=self.register, device=new_device)

        for call in self._calls[1:]:
            if not (call.name == "declare_channel"):
                getattr(new_seq, call.name)(*call.args, **call.kwargs)
                continue
            # Switch the old id with the correct id
            sw_channel_args = list(call.args)
            sw_channel_kw_args = call.kwargs.copy()
            if "name" in sw_channel_kw_args:  # pragma: no cover
                sw_channel_kw_args["channel_id"] = channel_match[
                    sw_channel_kw_args["name"]
                ]
            elif "channel_id" in sw_channel_kw_args:  # pragma: no cover
                sw_channel_kw_args["channel_id"] = channel_match[
                    sw_channel_args[0]
                ]
            else:
                sw_channel_args[1] = channel_match[sw_channel_args[0]]

            new_seq.declare_channel(*sw_channel_args, **sw_channel_kw_args)
        return new_seq

    @seq_decorators.block_if_measured
    def declare_channel(
        self,
        name: str,
        channel_id: str,
        initial_target: Optional[Union[QubitId, Iterable[QubitId]]] = None,
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
            name: Unique name for the channel in the sequence.
            channel_id: How the channel is identified in the device.
                Consult ``Sequence.available_channels`` to see which channel
                ID's are still available and the associated channel's
                description.
            initial_target: For 'Local' addressing channels only. Declares the
                initial target of the channel. If left as None, the initial
                target will have to be set manually as the first addition
                to this channel.
        """
        if name in self._schedule:
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

        if initial_target is not None:
            try:
                cond = any(
                    isinstance(t, Parametrized)
                    for t in cast(Iterable, initial_target)
                )
            except TypeError:
                cond = isinstance(initial_target, Parametrized)
            if cond:
                raise TypeError("The initial_target cannot be parametrized")

        if ch.basis == "XY" and not self._in_xy:
            self._in_xy = True
            self.set_magnetic_field()

        self._schedule[name] = _ChannelSchedule(channel_id, ch)

        if ch.basis not in self._basis_ref:
            self._basis_ref[ch.basis] = {q: _QubitRef() for q in self._qids}

        if ch.addressing == "Global":
            self._add_to_schedule(name, _TimeSlot("target", -1, 0, self._qids))
        elif initial_target is not None:
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

    @overload
    def declare_variable(
        self,
        name: str,
        *,
        dtype: Union[type[int], type[float]] = float,
    ) -> VariableItem:
        pass

    @overload
    def declare_variable(
        self,
        name: str,
        *,
        size: int,
        dtype: Union[type[int], type[float]] = float,
    ) -> Variable:
        pass

    def declare_variable(
        self,
        name: str,
        size: Optional[int] = None,
        dtype: Union[type[int], type[float]] = float,
    ) -> Union[Variable, VariableItem]:
        """Declare a new variable within this Sequence.

        The declared variables can be used to create parametrized versions of
        ``Waveform`` and ``Pulse`` objects, which in turn can be added to the
        ``Sequence``. Additionally, simple arithmetic operations involving
        variables are also supported and will return parametrized objects that
        are dependent on the involved variables.

        Args:
            name: The name for the variable. Must be unique within a
                Sequence.

        Args:
            size: The number of entries stored in the variable. If defined,
                returns an array of variables with the given size. If left
                as ``None``, returns a single variable.
            dtype: The type of the data that will be assigned
                to the variable. Must be ``float`` or ``int``.

        Returns:
            The declared Variable instance.

        Note:
            To avoid confusion, it is recommended to store the returned
            Variable instance in a Python variable with the same name.
        """
        if name in ("qubits", "seq_name"):
            raise ValueError(
                f"'{name}' is a protected name. Please choose a different name"
                " for the variable."
            )

        if name in self._variables:
            raise ValueError("Name for variable is already being used.")

        if size is None:
            var = self.declare_variable(name, size=1, dtype=dtype)
            return var[0]
        else:
            var = Variable(name, dtype, size=size)
            self._variables[name] = var
            return var

    @seq_decorators.store
    @seq_decorators.block_if_measured
    def enable_eom_mode(
        self,
        channel: str,
        amp_on: Union[float, Parametrized],
        detuning_on: Union[float, Parametrized],
        optimal_detuning_off: Union[float, Parametrized] = 0.0,
    ) -> None:
        """Puts a channel in EOM mode operation.

        For channels with a finite modulation bandwidth and an EOM, operation
        in EOM mode allows for the execution of square pulses with a higher
        bandwidth than that which is tipically available. It can be turned on
        and off through the `Sequence.enable_eom_mode()` and
        `Sequence.disable_eom_mode()` methods.
        A channel in EOM mode can only execute square pulses with a given
        amplitude (`amp_on`) and detuning (`detuning_on`), which are
        chosen at the moment the EOM mode is enabled. Furthermore, the
        detuning when there is no pulse being played (`detuning_off`) is
        restricted to a set of values that depends on `amp_on` and
        `detuning_on`.
        While in EOM mode, one can only add pulses of variable duration
        (through `Sequence.add_eom_pulse()`) or delays.

        Note:
            Enabling the EOM mode will automatically enforce a buffer unless
            the channel is empty. The detuning will go to the `detuning_off`
            value during this buffer. This buffer will not wait for pulses
            on other channels to finish, so calling `Sequence.align()` or
            `Sequence.delay()` before enabling the EOM mode is necessary to
            avoid eventual conflicts.

        Args:
            channel: The name of the channel to put in EOM mode.
            amp_on: The amplitude of the EOM pulses (in rad/µs).
            detuning_on: The detuning of the EOM pulses (in rad/µs).
            optimal_detuning_off: The optimal value of detuning (in rad/µs)
                when there is no pulse being played. It will choose the closest
                value among the existing options.
        """
        if self.is_in_eom_mode(channel):
            raise RuntimeError(
                f"The '{channel}' channel is already in EOM mode."
            )
        channel_obj = self.declared_channels[channel]
        if not channel_obj.supports_eom():
            raise TypeError(f"Channel '{channel}' does not have an EOM.")

        on_pulse = Pulse.ConstantPulse(
            channel_obj.min_duration, amp_on, detuning_on, 0.0
        )
        if not isinstance(on_pulse, Parametrized):
            channel_obj.validate_pulse(on_pulse)
            amp_on = cast(float, amp_on)
            detuning_on = cast(float, detuning_on)

            off_options = cast(
                RydbergEOM, channel_obj.eom_config
            ).detuning_off_options(amp_on, detuning_on)

            if not isinstance(optimal_detuning_off, Parametrized):
                closest_option = np.abs(
                    off_options - optimal_detuning_off
                ).argmin()
                detuning_off = off_options[closest_option]
                off_pulse = Pulse.ConstantPulse(
                    channel_obj.min_duration, 0.0, detuning_off, 0.0
                )
                channel_obj.validate_pulse(off_pulse)

            if not self.is_parametrized():
                self._schedule.enable_eom(
                    channel, amp_on, detuning_on, detuning_off
                )

    @seq_decorators.store
    @seq_decorators.block_if_measured
    def disable_eom_mode(self, channel: str) -> None:
        """Takes a channel out of EOM mode operation.

        For channels with a finite modulation bandwidth and an EOM, operation
        in EOM mode allows for the execution of square pulses with a higher
        bandwidth than that which is tipically available. It can be turned on
        and off through the `Sequence.enable_eom_mode()` and
        `Sequence.disable_eom_mode()` methods.
        A channel in EOM mode can only execute square pulses with a given
        amplitude (`amp_on`) and detuning (`detuning_on`), which are
        chosen at the moment the EOM mode is enabled. Furthermore, the
        detuning when there is no pulse being played (`detuning_off`) is
        restricted to a set of values that depends on `amp_on` and
        `detuning_on`.
        While in EOM mode, one can only add pulses of variable duration
        (through `Sequence.add_eom_pulse()`) or delays.

        Note:
            Disabling the EOM mode will automatically enforce a buffer time
            from the moment it is turned off.

        Args:
            channel: The name of the channel to take out of EOM mode.
        """
        if not self.is_in_eom_mode(channel):
            raise RuntimeError(f"The '{channel}' channel is not in EOM mode.")
        if not self.is_parametrized():
            self._schedule.disable_eom(channel)

    @seq_decorators.store
    @seq_decorators.mark_non_empty
    @seq_decorators.block_if_measured
    def add_eom_pulse(
        self,
        channel: str,
        duration: Union[int, Parametrized],
        phase: Union[float, Parametrized],
        post_phase_shift: Union[float, Parametrized] = 0.0,
        protocol: PROTOCOLS = "min-delay",
    ) -> None:
        """Adds a square pulse to a channel in EOM mode.

        For channels with a finite modulation bandwidth and an EOM, operation
        in EOM mode allows for the execution of square pulses with a higher
        bandwidth than that which is tipically available. It can be turned on
        and off through the `Sequence.enable_eom_mode()` and
        `Sequence.disable_eom_mode()` methods.
        A channel in EOM mode can only execute square pulses with a given
        amplitude (`amp_on`) and detuning (`detuning_on`), which are
        chosen at the moment the EOM mode is enabled. Furthermore, the
        detuning when there is no pulse being played (`detuning_off`) is
        restricted to a set of values that depends on `amp_on` and
        `detuning_on`.
        While in EOM mode, one can only add pulses of variable duration
        (through `Sequence.add_eom_pulse()`) or delays.

        Note:
            When the phase between pulses is changed, the necessary buffer
            time for a phase jump will still be enforced (unless
            ``protocol='no-delay'``).

        Args:
            channel: The name of the channel to add the pulse to.
            duration: The duration of the pulse (in ns).
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a phase shift (in rads)
                immediately after the end of the pulse.
            protocol: Stipulates how to deal with eventual conflicts with
                other channels (see `Sequence.add()` for more details).
        """
        if not self.is_in_eom_mode(channel):
            raise RuntimeError(f"Channel '{channel}' must be in EOM mode.")

        if self.is_parametrized():
            self._validate_add_protocol(protocol)
            # Test the parameters
            if not isinstance(duration, Parametrized):
                channel_obj = self.declared_channels[channel]
                channel_obj.validate_duration(duration)
            for arg in (phase, post_phase_shift):
                if not isinstance(arg, (float, int)):
                    raise TypeError("Phase values must be a numeric value.")
            return

        eom_settings = self._schedule[channel].eom_blocks[-1]
        eom_pulse = Pulse.ConstantPulse(
            duration,
            eom_settings.rabi_freq,
            eom_settings.detuning_on,
            phase,
            post_phase_shift=post_phase_shift,
        )
        self._add(eom_pulse, channel, protocol)

    @seq_decorators.store
    @seq_decorators.mark_non_empty
    @seq_decorators.block_if_measured
    def add(
        self,
        pulse: Union[Pulse, Parametrized],
        channel: str,
        protocol: PROTOCOLS = "min-delay",
    ) -> None:
        """Adds a pulse to a channel.

        Args:
            pulse: The pulse object to add to the channel.
            channel: The channel's name provided when declared.
            protocol: Stipulates how to deal with
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

        Note:
            When the phase of the pulse to add is different than the phase of
            the previous pulse on the channel, a delay between the two pulses
            might be automatically added to ensure the channel's
            `phase_jump_time` is respected. To override this behaviour, use
            the ``'no-delay'`` protocol.
        """
        self._validate_channel(channel, block_eom_mode=True)
        self._add(pulse, channel, protocol)

    @seq_decorators.store
    def target(
        self,
        qubits: Union[QubitId, Iterable[QubitId]],
        channel: str,
    ) -> None:
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits: The new target for this channel. Must correspond to a
                qubit ID in device or an iterable of qubit IDs, when
                multi-qubit addressing is possible.
            channel: The channel's name provided when declared. Must be
                a channel with 'Local' addressing.
        """
        self._target(qubits, channel)

    @seq_decorators.store
    @seq_decorators.check_allow_qubit_index
    def target_index(
        self,
        qubits: Union[int, Iterable[int], Parametrized],
        channel: str,
    ) -> None:
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits: The new target for this channel. Must correspond to a
                qubit index or an iterable of qubit indices, when multi-qubit
                addressing is possible.
                A qubit index is a number between 0 and the number of qubits.
                It is then converted to a Qubit ID using the order in which
                they were declared when instantiating the ``Register``
                or ``MappableRegister``.
            channel: The channel's name provided when declared. Must be
                a channel with 'Local' addressing.

        Note:
            Cannot be used on non-parametrized sequences using a mappable
            register.
        """
        self._target(qubits, channel, _index=True)

    @seq_decorators.store
    def delay(
        self,
        duration: Union[int, Parametrized],
        channel: str,
    ) -> None:
        """Idles a given channel for a specific duration.

        Args:
            duration: Time to delay (in ns).
            channel: The channel's name provided when declared.
        """
        self._delay(duration, channel)

    @seq_decorators.store
    @seq_decorators.block_if_measured
    def measure(self, basis: str = "ground-rydberg") -> None:
        """Measures in a valid basis.

        Note:
            In addition to the supported bases of the selected device, allowed
            measurement bases will depend on the mode of operation. In
            particular, if using ``Microwave`` channels (XY mode), only
            measuring in the 'XY' basis is allowed. Inversely, it is not
            possible to measure in the 'XY' basis outside of XY mode.

        Args:
            basis: Valid basis for measurement (consult the
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

        if self.is_parametrized():
            self._is_measured = True
        else:
            self._measurement = basis

    @seq_decorators.store
    def phase_shift(
        self,
        phi: Union[float, Parametrized],
        *targets: QubitId,
        basis: str = "digital",
    ) -> None:
        r"""Shifts the phase of a qubit's reference by 'phi', on a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi: The intended phase shift (in rads).
            targets: The ids of the qubits to apply the phase shift to.
            basis: The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.
        """
        self._phase_shift(phi, *targets, basis=basis)

    @seq_decorators.store
    @seq_decorators.check_allow_qubit_index
    def phase_shift_index(
        self,
        phi: Union[float, Parametrized],
        *targets: Union[int, Parametrized],
        basis: str = "digital",
    ) -> None:
        r"""Shifts the phase of a qubit's reference by 'phi', on a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi: The intended phase shift (in rads).
            targets: The indices of the qubits to apply the phase shift to.
                A qubit index is a number between 0 and the number of qubits.
                It is then converted to a Qubit ID using the order in which
                they were declared when instantiating the ``Register``
                or ``MappableRegister``.
            basis: The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.

        Note:
            Cannot be used on non-parametrized sequences using a mappable
            register.
        """
        self._phase_shift(phi, *targets, basis=basis, _index=True)

    @seq_decorators.store
    @seq_decorators.block_if_measured
    def align(self, *channels: str) -> None:
        """Aligns multiple channels in time.

        Introduces delays that align the provided channels with the one that
        finished the latest, such that the next action added to any of them
        will start right after the latest channel has finished.

        Args:
            channels: The names of the channels to align, as given upon
                declaration.
        """
        ch_set = set(channels)
        # channels have to be a subset of the declared channels
        if not ch_set <= set(self._schedule):
            raise ValueError(
                "All channel names must correspond to declared channels."
            )
        if len(channels) != len(ch_set):
            raise ValueError("The same channel was provided more than once.")

        if len(channels) < 2:
            raise ValueError("Needs at least two channels for alignment.")

        if self.is_parametrized():
            return

        last_ts = {
            id: self.get_duration(id, include_fall_time=True)
            for id in channels
        }
        tf = max(last_ts.values())

        for id in channels:
            delta = tf - last_ts[id]
            if delta > 0:
                self._delay(
                    self._schedule[id].adjust_duration(delta),
                    id,
                )

    def build(
        self,
        *,
        qubits: Optional[Mapping[QubitId, int]] = None,
        **vars: Union[ArrayLike, float, int],
    ) -> Sequence:
        """Builds a sequence from the programmed instructions.

        Args:
            qubits: A mapping between qubit IDs and trap IDs used to define
                the register. Must only be provided when the sequence is
                initialized with a MappableRegister.
            vars: The values for all the variables declared in this Sequence
                instance, indexed by the name given upon declaration. Check
                ``Sequence.declared_variables`` to see all the variables.

        Returns:
            The Sequence built with the given variable values.

        Example:
            ::

                # Check which variables are declared
                >>> print(seq.declared_variables)
                {'x': Variable(name='x', dtype=<class 'float'>, size=1),
                 'y': Variable(name='y', dtype=<class 'int'>, size=3)}
                # Build a sequence with specific values for both variables
                >>> seq1 = seq.build(x=0.5, y=[1, 2, 3])
        """
        if self.is_register_mappable():
            if qubits is None:
                raise ValueError(
                    "'qubits' must be specified when the sequence is created "
                    "with a MappableRegister."
                )

        elif qubits is not None:
            raise ValueError(
                "'qubits' must not be specified when the sequence already has "
                "a concrete register."
            )

        self._cross_check_vars(vars)

        # Shallow copy with stored parametrized objects (if any)
        # NOTE: While seq is a shallow copy, be extra careful with changes to
        # attributes of seq pointing to mutable objects, as they might be
        # inadvertedly done to self too
        seq = copy.copy(self)

        # Eliminates the source of recursiveness errors
        seq._reset_parametrized()

        # Deepcopy the base sequence (what remains)
        seq = copy.deepcopy(seq)
        # NOTE: Changes to seq are now safe to do

        if not (self.is_parametrized() or self.is_register_mappable()):
            warnings.warn(
                "Building a non-parametrized sequence simply returns"
                " a copy of itself.",
                stacklevel=2,
            )
            return seq

        for name, value in vars.items():
            self._variables[name]._assign(value)

        if qubits:
            reg = cast(MappableRegister, self._register).build_register(qubits)
            self._set_register(seq, reg)

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
            The sequence encoded in a JSON formatted string.

        See Also:
            ``json.dumps``: Built-in function for serialization to a JSON
            formatted string.
        """
        return json.dumps(self, cls=PulserEncoder, **kwargs)

    def to_abstract_repr(
        self, seq_name: str = "pulser-exported", **defaults: Any
    ) -> str:
        """Serializes the Sequence into an abstract JSON object.

        Keyword Args:
            seq_name (str): A name for the sequence. If not defined, defaults
                to "pulser-exported".
            defaults: The default values for all the variables declared in this
                Sequence instance, indexed by the name given upon declaration.
                Check ``Sequence.declared_variables`` to see all the variables.
                When using a MappableRegister, the Qubit IDs to trap IDs
                mapping must also be provided under the `qubits` keyword.

        Note:
            Providing the `defaults` is optional but, when done, it is
            mandatory to give default values for all the expected parameters.

        Returns:
            str: The sequence encoded as an abstract JSON object.

        See Also:
            ``serialize``
        """
        return serialize_abstract_sequence(self, seq_name, **defaults)

    @staticmethod
    def deserialize(obj: str, **kwargs: Any) -> Sequence:
        """Deserializes a JSON formatted string.

        Args:
            obj: The JSON formatted string to deserialize, coming from
                the serialization of a ``Sequence`` through
                ``Sequence.serialize()``.

        Other Parameters:
            kwargs: Valid keyword-arguments for ``json.loads()``, except for
                ``cls`` and ``object_hook``.

        Returns:
            The deserialized Sequence object.

        See Also:
            ``json.loads``: Built-in function for deserialization from a JSON
            formatted string.
        """
        if "Sequence" not in obj:
            raise ValueError(
                "The given JSON formatted string does not encode a Sequence."
            )

        return cast(Sequence, json.loads(obj, cls=PulserDecoder, **kwargs))

    @staticmethod
    def from_abstract_repr(obj_str: str) -> Sequence:
        """Deserialize a sequence from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the sequence encoded
                in the abstract JSON format.

        Returns:
            Sequence: The Pulser sequence.
        """
        return deserialize_abstract_sequence(obj_str)

    @seq_decorators.screen
    def draw(
        self,
        mode: str = "input+output",
        draw_phase_area: bool = False,
        draw_interp_pts: bool = True,
        draw_phase_shifts: bool = False,
        draw_register: bool = False,
        draw_phase_curve: bool = False,
        fig_name: str = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the sequence in its current state.

        Args:
            mode: The curves to draw. 'input'
                draws only the programmed curves, 'output' the excepted curves
                after modulation. 'input+output' will draw both curves except
                for channels without a defined modulation bandwidth, in which
                case only the input is drawn.
            draw_phase_area: Whether phase and area values need to be
                shown as text on the plot, defaults to False. Doesn't work in
                'output' mode. If `draw_phase_curve=True`, phase values are
                ommited.
            draw_interp_pts: When the sequence has pulses with waveforms
                of type InterpolatedWaveform, draws the points of interpolation
                on top of the respective input waveforms (defaults to True).
                Doesn't work in 'output' mode.
            draw_phase_shifts: Whether phase shift and reference
                information should be added to the plot, defaults to False.
            draw_register: Whether to draw the register before the pulse
                sequence, with a visual indication (square halo) around the
                qubits masked by the SLM, defaults to False. Can't be set to
                True if the sequence is defined with a mappable register.
            draw_phase_curve: Draws the changes in phase in its own curve
                (ignored if the phase doesn't change throughout the channel).
            fig_name: The name on which to save the
                figure. If `draw_register` is True, both pulses and register
                will be saved as figures, with a suffix ``_pulses`` and
                ``_register`` in the file name. If `draw_register` is False,
                only the pulses are saved, with no suffix. If `fig_name` is
                None, no figure is saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        See Also:
            Simulation.draw(): Draws the provided sequence and the one used by
            the solver.
        """
        valid_modes = ("input", "output", "input+output")
        if mode not in valid_modes:
            raise ValueError(
                f"'mode' must be one of {valid_modes}, not '{mode}'."
            )
        if mode == "output":
            if draw_phase_area:
                warnings.warn(
                    "'draw_phase_area' doesn't work in 'output' mode, so it "
                    "will default to 'False'.",
                    stacklevel=2,
                )
                draw_phase_area = False
            if draw_interp_pts:
                warnings.warn(
                    "'draw_interp_pts' doesn't work in 'output' mode, so it "
                    "will default to 'False'.",
                    stacklevel=2,
                )
                draw_interp_pts = False
        if draw_register and self.is_register_mappable():
            raise ValueError(
                "Can't draw the register for a sequence without a defined "
                "register."
            )
        fig_reg, fig = self._plot(
            draw_phase_area=draw_phase_area,
            draw_interp_pts=draw_interp_pts,
            draw_phase_shifts=draw_phase_shifts,
            draw_register=draw_register,
            draw_input="input" in mode,
            draw_modulation="output" in mode,
            draw_phase_curve=draw_phase_curve,
        )
        if fig_name is not None and fig_reg is not None:
            name, ext = os.path.splitext(fig_name)
            fig.savefig(name + "_pulses" + ext, **kwargs_savefig)
            fig_reg.savefig(name + "_register" + ext, **kwargs_savefig)
        elif fig_name:
            fig.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def _plot(self, **draw_options: bool) -> tuple[Figure | None, Figure]:
        return draw_sequence(self, **draw_options)

    def _add(
        self,
        pulse: Union[Pulse, Parametrized],
        channel: str,
        protocol: PROTOCOLS,
    ) -> None:
        self._validate_add_protocol(protocol)
        if self.is_parametrized():
            if not isinstance(pulse, Parametrized):
                self._validate_and_adjust_pulse(pulse, channel)
            return

        pulse = cast(Pulse, pulse)
        channel_obj = self._schedule[channel].channel_obj
        last = self._last(channel)
        basis = channel_obj.basis

        ph_refs = {
            self._basis_ref[basis][q].phase.last_phase for q in last.targets
        }
        if len(ph_refs) != 1:
            raise ValueError(
                "Cannot do a multiple-target pulse on qubits with different "
                "phase references for the same basis."
            )
        else:
            phase_ref = ph_refs.pop()

        pulse = self._validate_and_adjust_pulse(pulse, channel, phase_ref)

        phase_barriers = [
            self._basis_ref[basis][q].phase.last_time for q in last.targets
        ]

        self._schedule.add_pulse(pulse, channel, phase_barriers, protocol)

        true_finish = self._last(channel).tf + pulse.fall_time(
            channel_obj, in_eom_mode=self.is_in_eom_mode(channel)
        )
        for qubit in last.targets:
            self._basis_ref[basis][qubit].update_last_used(true_finish)

        if pulse.post_phase_shift:
            self._phase_shift(
                pulse.post_phase_shift, *last.targets, basis=basis
            )

    @seq_decorators.block_if_measured
    def _target(
        self,
        qubits: Union[Iterable[QubitId], QubitId, Parametrized],
        channel: str,
        _index: bool = False,
    ) -> None:
        self._validate_channel(channel, block_eom_mode=True)
        channel_obj = self._schedule[channel].channel_obj
        try:
            qubits_set = (
                set(cast(Iterable, qubits))
                if not isinstance(qubits, str)
                else {qubits}
            )
        except TypeError:
            qubits_set = {qubits}

        if channel_obj.addressing != "Local":
            raise ValueError("Can only choose target of 'Local' channels.")
        elif (
            channel_obj.max_targets is not None
            and len(qubits_set) > channel_obj.max_targets
        ):
            raise ValueError(
                f"This channel can target at most {channel_obj.max_targets} "
                "qubits at a time."
            )
        qubit_ids_set = self._check_qubits_give_ids(*qubits_set, _index=_index)

        if not self.is_parametrized():
            basis = channel_obj.basis
            phase_refs = {
                self._basis_ref[basis][q].phase.last_phase
                for q in qubit_ids_set
            }
            if len(phase_refs) != 1:
                raise ValueError(
                    "Cannot target multiple qubits with different "
                    "phase references for the same basis."
                )
            self._schedule.add_target(qubit_ids_set, channel)

    def _check_qubits_give_ids(
        self, *qubits: Union[QubitId, Parametrized], _index: bool = False
    ) -> set[QubitId]:
        if _index:
            if self.is_parametrized():
                nb_of_indices = len(self._register.qubit_ids)
                allowed_indices = range(nb_of_indices)
                for i in qubits:
                    if i not in allowed_indices and not isinstance(
                        i, Parametrized
                    ):
                        raise ValueError(
                            f"All non-variable targets must be indices valid "
                            f"for the register, between 0 and "
                            f"{nb_of_indices - 1}. Wrong index: {i!r}."
                        )
                return set()
            else:
                qubits = cast(Tuple[int, ...], qubits)
                try:
                    return {self.register.qubit_ids[index] for index in qubits}
                except IndexError:
                    raise IndexError("Indices must exist for the register.")
        ids = set(cast(Tuple[QubitId, ...], qubits))
        if not ids <= self._qids:
            raise ValueError(
                "All given ids have to be qubit ids declared"
                " in this sequence's register."
            )
        return ids

    @seq_decorators.block_if_measured
    def _delay(self, duration: Union[int, Parametrized], channel: str) -> None:
        self._validate_channel(channel)
        if self.is_parametrized():
            return
        self._schedule.add_delay(cast(int, duration), channel)

    def _phase_shift(
        self,
        phi: Union[float, Parametrized],
        *targets: Union[QubitId, Parametrized],
        basis: str,
        _index: bool = False,
    ) -> None:
        if basis not in self._basis_ref:
            raise ValueError("No declared channel targets the given 'basis'.")
        target_ids = self._check_qubits_give_ids(*targets, _index=_index)

        if not self.is_parametrized():
            phi = cast(float, phi)
            if phi % (2 * np.pi) == 0:
                return

            for qubit in target_ids:
                self._basis_ref[basis][qubit].increment_phase(phi)

    def _to_dict(self, _module: str = "pulser.sequence") -> dict[str, Any]:
        d = obj_to_dict(
            self,
            *self._calls[0].args,
            _module=_module,
            **self._calls[0].kwargs,
        )
        d["__version__"] = pulser.__version__
        d["calls"] = self._calls[1:]
        d["vars"] = self._variables
        d["to_build_calls"] = self._to_build_calls
        return d

    def __str__(self) -> str:
        return seq_to_str(self)

    def _add_to_schedule(self, channel: str, timeslot: _TimeSlot) -> None:
        # Maybe get rid of this
        self._schedule[channel].slots.append(timeslot)

    def _last(self, channel: str) -> _TimeSlot:
        """Shortcut to last element in the channel's schedule."""
        return self._schedule[channel][-1]

    def _validate_channel(
        self, channel: str, block_eom_mode: bool = False
    ) -> None:
        if isinstance(channel, Parametrized):
            raise NotImplementedError(
                "Using parametrized objects or variables to refer to channels "
                "is not supported."
            )
        if channel not in self._schedule:
            raise ValueError("Use the name of a declared channel.")
        if block_eom_mode and self.is_in_eom_mode(channel):
            raise RuntimeError("The chosen channel is in EOM mode.")

    def _validate_and_adjust_pulse(
        self, pulse: Pulse, channel: str, phase_ref: Optional[float] = None
    ) -> Pulse:
        channel_obj = self._schedule[channel].channel_obj
        channel_obj.validate_pulse(pulse)
        _duration = channel_obj.validate_duration(pulse.duration)
        new_phase = pulse.phase + (phase_ref if phase_ref else 0)
        if _duration != pulse.duration:
            try:
                new_amp = pulse.amplitude.change_duration(_duration)
                new_det = pulse.detuning.change_duration(_duration)
            except NotImplementedError:
                raise TypeError(
                    "Failed to automatically adjust one of the pulse's "
                    "waveforms to the channel duration constraints. Choose a "
                    "duration that is a multiple of "
                    f"{channel_obj.clock_period} ns."
                )
        else:
            new_amp = pulse.amplitude
            new_det = pulse.detuning

        return Pulse(new_amp, new_det, new_phase, pulse.post_phase_shift)

    def _validate_add_protocol(self, protocol: str) -> None:
        valid_protocols = get_args(PROTOCOLS)
        if protocol not in valid_protocols:
            raise ValueError(
                f"Invalid protocol '{protocol}', only accepts protocols: "
                + ", ".join(valid_protocols)
            )

    def _reset_parametrized(self) -> None:
        """Resets all attributes related to parametrization."""
        # Signals the sequence as actively "building" ie not parametrized
        self._building = True
        self._is_measured = False
        self._variables = {}
        self._to_build_calls = []

    def _set_register(self, seq: Sequence, reg: BaseRegister) -> None:
        """Sets the register on a sequence who had a mappable register."""
        self._device.validate_register(reg)
        qids = set(reg.qubit_ids)
        used_qubits = set()
        for ch, ch_schedule in self._schedule.items():
            # Correct the targets of global channels
            if ch_schedule.channel_obj.addressing == "Global":
                for i, slot in enumerate(self._schedule[ch]):
                    stored_values = slot._asdict()
                    stored_values["targets"] = qids
                    seq._schedule[ch].slots[i] = _TimeSlot(**stored_values)
            else:
                # Make sure all explicit targets are in the register
                for slot in self._schedule[ch]:
                    used_qubits.update(slot.targets)

        if not used_qubits <= qids:
            raise ValueError(
                f"Qubits {used_qubits - qids} are being targeted but"
                " have not been assigned a trap."
            )
        seq._register = reg
        seq._qids = qids
        seq._calls[0] = _Call("__init__", (seq._register, seq._device), {})

    def _cross_check_vars(self, vars: dict[str, Any]) -> None:
        """Checks if values are given to all and only declared vars."""
        all_keys, given_keys = self._variables.keys(), vars.keys()
        if given_keys != all_keys:
            invalid_vars = given_keys - all_keys
            if invalid_vars:
                warnings.warn(
                    "No declared variables named: " + ", ".join(invalid_vars),
                    stacklevel=3,
                )
                for k in invalid_vars:
                    vars.pop(k, None)
            missing_vars = all_keys - given_keys
            if missing_vars:
                raise TypeError(
                    "Did not receive values for variables: "
                    + ", ".join(missing_vars)
                )
