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
from collections.abc import Collection, Mapping
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)

import jsonschema
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import pulser
import pulser.devices as devices
import pulser.math as pm
import pulser.sequence._decorators as seq_decorators
from pulser.channels.base_channel import Channel, States, get_states_from_bases
from pulser.channels.dmm import DMM, _dmm_id_from_name, _get_dmm_name
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices._device_datacls import BaseDevice
from pulser.json.abstract_repr.deserializer import (
    deserialize_abstract_sequence,
)
from pulser.json.abstract_repr.serializer import serialize_abstract_sequence
from pulser.json.coders import PulserDecoder, PulserEncoder
from pulser.json.exceptions import AbstractReprError
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, Variable
from pulser.parametrized.variable import VariableItem
from pulser.pulse import Pulse
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.mappable_reg import MappableRegister
from pulser.register.weight_maps import DetuningMap
from pulser.sequence._basis_ref import _QubitRef
from pulser.sequence._call import _Call
from pulser.sequence._schedule import (
    _ChannelSchedule,
    _DMMSchedule,
    _PhaseDriftParams,
    _Schedule,
    _TimeSlot,
)
from pulser.sequence._seq_drawer import Figure, draw_sequence
from pulser.sequence.helpers._seq_str import seq_to_str
from pulser.sequence.helpers._switch_device import switch_device
from pulser.waveforms import Waveform

DeviceType = TypeVar("DeviceType", bound=BaseDevice)

PROTOCOLS = Literal["min-delay", "no-delay", "wait-for-all"]


class Sequence(Generic[DeviceType]):
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
        device: DeviceType,
    ):
        """Initializes a new pulse sequence."""
        if not isinstance(device, BaseDevice):
            raise TypeError(
                f"'device' must be of type 'BaseDevice', not {type(device)}."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if device == devices.Chadoq2:
                warnings.warn(
                    "The 'Chadoq2' device has been deprecated. For a "
                    "similar device combining global and local addressing, "
                    "consider using `DigitalAnalogDevice`.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )

            if device == devices.IroiseMVP:
                warnings.warn(
                    "The 'IroiseMVP' device has been deprecated. For a "
                    "similar analog device consider using `AnalogDevice`.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )

        # Checks if register is compatible with the device
        if isinstance(register, MappableRegister):
            device.validate_layout(register.layout)
            device.validate_layout_filling(register)
        else:
            device.validate_register(register)

        self._register: Union[BaseRegister, MappableRegister] = register
        self._device = device
        self._in_xy: bool = False
        self._in_ising_value: bool = False
        self._mag_field: Optional[tuple[float, float, float]] = None
        self._calls: list[_Call] = [
            _Call("__init__", (), {"register": register, "device": device})
        ]
        self._schedule: _Schedule = _Schedule(
            max_duration=device.max_sequence_duration
        )
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
        self._slm_mask_dmm: str | None = None
        # Initializes all parametrized Sequence related attributes
        self._reset_parametrized()

    @property
    def _slm_mask_time(self) -> list[int]:
        """The initial and final time when the SLM mask is on."""
        if (
            self._in_ising
            and self._slm_mask_dmm
            and not cast(
                _DMMSchedule, self._schedule[self._slm_mask_dmm]
            )._waiting_for_first_pulse
        ):
            slm_slot = self._schedule[self._slm_mask_dmm].slots[1]
            return [slm_slot.ti, slm_slot.tf]
        return (
            []
            if not self._slm_mask_targets
            else self._schedule.find_slm_mask_times()
        )

    @property
    def _in_ising(self) -> bool:
        return self._in_ising_value

    @_in_ising.setter
    def _in_ising(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("_in_ising must be a bool.")
        if self._in_ising == value:
            # If the value doesn't change, do nothing
            return
        if self._in_ising:  # ie value = False
            # Trying to switch off ising
            raise ValueError("Cannot quit ising.")
        # At this point, value = True
        if self._in_xy:
            raise ValueError("Cannot be in ising if in xy.")
        self._in_ising_value = True
        if self._slm_mask_dmm:
            self._set_slm_mask_dmm(self._slm_mask_dmm, self._slm_mask_targets)

    @property
    def qubit_info(self) -> dict[QubitId, pm.AbstractArray]:
        """Dictionary with the qubit's IDs and positions."""
        if self.is_register_mappable():
            raise RuntimeError(
                "Can't access the qubit information when the register is "
                "mappable."
            )
        return cast(BaseRegister, self._register).qubits

    @property
    def device(self) -> DeviceType:
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

    @overload
    def get_register(self, include_mappable: Literal[False]) -> BaseRegister:
        pass

    @overload
    def get_register(
        self, include_mappable: Literal[True]
    ) -> BaseRegister | MappableRegister:
        pass

    def get_register(
        self, include_mappable: bool = True
    ) -> BaseRegister | MappableRegister:
        """The atom register on which to apply the pulses."""
        return self._register if include_mappable else self.register

    def _get_dmm_id_detuning_map(self, call: _Call) -> tuple[str, DetuningMap]:
        dmm_id: str
        det_map: DetuningMap
        # Get DMM name
        if "dmm_id" in call.kwargs:
            dmm_id = call.kwargs["dmm_id"]
        elif len(call.args) > 1:
            dmm_id = call.args[1]
        else:
            dmm_id = "dmm_0"
        # Get DetuningMap
        if "detuning_map" in call.kwargs:
            det_map = call.kwargs["detuning_map"]
        elif isinstance(call.args[0], DetuningMap):
            det_map = call.args[0]
        else:  # SLM case:
            det_map = self._slm_detuning_map(set(call.args[0]))
        return (dmm_id, det_map)

    @property
    def declared_channels(self) -> dict[str, Channel]:
        """Channels declared in this Sequence."""
        all_declared_channels = {
            name: cs.channel_obj for name, cs in self._schedule.items()
        }
        # Add DMM and SLM whose configuration is stored
        for call in self._to_build_calls:
            if (
                call.name == "config_slm_mask"
                or call.name == "config_detuning_map"
            ):
                (dmm_id, _) = self._get_dmm_id_detuning_map(call)
                dmm_name = _get_dmm_name(
                    dmm_id, list(all_declared_channels.keys())
                )
                all_declared_channels[dmm_name] = self.device.dmm_channels[
                    dmm_id
                ]
        return all_declared_channels

    @property
    def declared_variables(self) -> dict[str, Variable]:
        """Variables declared in this Sequence."""
        return dict(self._variables)

    @property
    def available_channels(self) -> dict[str, Channel]:
        """Channels still available for declaration."""
        all_channels = {**self._device.channels, **self._device.dmm_channels}
        if not self._in_xy and not self._in_ising:
            # If no channel has been declared nor any DMM configured, and if
            # device is physical, don't show the DMM used for the SLM Mask
            if (
                self._slm_mask_dmm is not None
                and not self._device.reusable_channels
            ):
                return {
                    id: ch
                    for id, ch in all_channels.items()
                    if id != self._slm_mask_dmm
                }
            return all_channels
        else:
            occupied_ch_ids = [
                (
                    self._schedule[ch_name].channel_id
                    if ch_name in self._schedule
                    else _dmm_id_from_name(ch_name)
                )
                for ch_name in self.declared_channels.keys()
            ]
            return {
                id: ch
                for id, ch in all_channels.items()
                if (
                    # MockDevice channels can be declared multiple times
                    (
                        id not in occupied_ch_ids
                        or self._device.reusable_channels
                    )
                    and (
                        # If we are in XY mode, the dmm channels are available
                        # to configure a SLM mask if no slm mask was defined
                        ch.basis == "XY"
                        or (isinstance(ch, DMM) and self._slm_mask_dmm is None)
                        if self._in_xy
                        else ch.basis != "XY"
                    )
                )
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
            bool(self._param_measurement)
            if self.is_parametrized()
            else hasattr(self, "_measurement")
        )

    def get_measurement_basis(self) -> str:
        """Gets the sequence's measurement basis.

        Raises:
            RuntimeError: When the sequence has not been measured.
        """
        if not self.is_measured():
            raise RuntimeError("The sequence has not been measured.")
        return (
            self._param_measurement
            if self.is_parametrized()
            else self._measurement
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

    def get_addressed_bases(self) -> tuple[str, ...]:
        """Returns the bases addressed by the declared channels."""
        return tuple(self._basis_ref)

    def get_addressed_states(self) -> list[States]:
        """Returns the states addressed by the declared channels."""
        return get_states_from_bases(self.get_addressed_bases())

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
            raise ValueError(
                f"No declared channel targets the given 'basis' ('{basis}')."
            )

        return float(self._basis_ref[basis][qubit].phase.last_phase)

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

    def _slm_detuning_map(self, targets: set[QubitId]) -> DetuningMap:
        return self.register.define_detuning_map(
            {
                qubit: (1.0 if qubit in targets else 0)
                for qubit in self.register.qubit_ids
            }
        )

    def _set_slm_mask_dmm(self, dmm_id: str, targets: set[QubitId]) -> None:
        detuning_map = self._slm_detuning_map(targets)
        self._config_detuning_map(detuning_map, dmm_id)
        # Find the name of the dmm in the declared channels.
        for key in reversed(self.declared_channels.keys()):
            if dmm_id == _dmm_id_from_name(key):
                self._slm_mask_dmm = key
                break
        # Modulate the dmm if pulses have already been added to Global Channels
        slm_mask_times = self._schedule.find_slm_mask_times()
        if slm_mask_times:
            max_amp = max(
                [
                    np.max(ch_schedule.get_samples().amp[: slm_mask_times[1]])
                    for ch_schedule in self._schedule.values()
                    if not isinstance(ch_schedule, _DMMSchedule)
                    and ch_schedule.channel_obj.addressing == "Global"
                ]
            )
            self._modulate_slm_mask_dmm(slm_mask_times[1], max_amp)
        else:
            # Block the modulation of this dmm
            cast(
                _DMMSchedule, self._schedule[key]
            )._waiting_for_first_pulse = True

    @seq_decorators.store
    def config_slm_mask(
        self, qubits: Collection[QubitId], dmm_id: str = "dmm_0"
    ) -> None:
        """Setup an SLM mask by specifying the qubits it targets.

        If the sequence is in XY mode, masked qubits don't interact with
        the incoming pulses until the end of the first pulse of the global
        channel starting the earliest in the schedule.

        If the sequence is in Ising, the SLM Mask is a DetuningMap where
        the detuning of each masked qubit is 1.0. DMM "dmm_id" is
        configured using this Detuning Map, and modulated by a pulse having
        a large negative detuning and either a duration defined from pulses
        already present in the sequence (same as in XY mode) or by the first
        pulse added after this operation.

        Args:
            qubits: Collection of qubit ID's to mask during the first global
                pulse of the sequence.
            dmm_id: Id of the DMM channel to use in the device.
        """
        if not self._device.supports_slm_mask:
            raise ValueError(
                f"The '{self._device}' device does not have an SLM mask."
            )

        if self.is_register_mappable():
            raise RuntimeError(
                "The SLM mask can't be combined with a mappable register."
            )

        try:
            targets = set(qubits)
        except TypeError:
            raise TypeError("The SLM targets must be castable to set.")

        if not targets.issubset(self._qids):
            raise ValueError("SLM mask targets must exist in the register.")

        # If sequence is parametrized slm is configured at build
        if self.is_parametrized():
            return

        if self._slm_mask_targets:
            raise ValueError("SLM mask can be configured only once.")

        if self._in_xy or (not self._in_xy and not self._in_ising):
            if dmm_id not in self._device.dmm_channels:
                raise ValueError(f"No DMM {dmm_id} in the device.")
            self._slm_mask_dmm = dmm_id
        if not self._in_xy and self._in_ising:
            self._set_slm_mask_dmm(dmm_id, targets)
        self._slm_mask_targets = targets

    @seq_decorators.store
    @seq_decorators.block_if_measured
    def config_detuning_map(
        self,
        detuning_map: DetuningMap,
        dmm_id: str,
    ) -> None:
        """Declares a new DMM channel to the Sequence.

        Associates a DetuningMap to a DMM channel of the Device.

        Note:
            Regular devices only allow a DMM to be declared once, but
            ``MockDevice`` DMM can be repeatedly declared if needed.

        Args:
            detuning_map: A DetuningMap defining the amount of detuning each
                atom receives.
            dmm_id: How the channel is identified in the device.
                See in ``Sequence.available_channels`` which DMM IDs are still
                available (start by "dmm" ) and the associated description.
        """
        self._config_detuning_map(detuning_map, dmm_id)

    def _config_detuning_map(
        self,
        detuning_map: DetuningMap,
        dmm_id: str,
    ) -> None:
        if dmm_id not in self._device.dmm_channels:
            raise ValueError(f"No DMM {dmm_id} in the device.")

        dmm_ch = self._device.dmm_channels[dmm_id]
        if self._in_xy:
            raise ValueError(
                f"DMM '{dmm_ch}' cannot work simultaneously "
                "with the declared 'Microwave' channel."
            )
        if dmm_id not in self.available_channels:
            raise ValueError(f"DMM {dmm_id} is not available.")

        # Configures the DMM implementing an SLM mask if configured before
        self._in_ising = True

        if self.is_parametrized():
            return
        # Add a suffix to the DMM id if repetition in the declared channels
        dmm_name = dmm_id
        if dmm_id in self.declared_channels:
            assert self._device.reusable_channels
            dmm_name = _get_dmm_name(
                dmm_id, list(self.declared_channels.keys())
            )

        self._schedule[dmm_name] = _DMMSchedule(
            dmm_id, dmm_ch, detuning_map=detuning_map
        )
        if "ground-rydberg" not in self._basis_ref:
            self._basis_ref["ground-rydberg"] = {
                q: _QubitRef() for q in self._qids
            }

        # DMM has Global addressing
        self._add_to_schedule(dmm_name, _TimeSlot("target", -1, 0, self._qids))

    def switch_register(
        self, new_register: BaseRegister | MappableRegister
    ) -> Sequence:
        """Replicate the sequence with a different register.

        The new sequence is reconstructed with the provided register by
        replicating all the instructions used to build the original sequence.
        This means that operations referecing specific qubits IDs
        (eg. `Sequence.target()`) expect to find the same qubit IDs in the new
        register. By the same token, switching from a register to a mappable
        register might fail if one of the instructions does not work with
        mappable registers (e.g. `Sequence.configure_slm_mask()`).

        Warns:
            UserWarning: If the sequence is configuring a detuning map, a
            warning is raised to remind the user that the detuning map is
            unchanged and might no longer be aligned with the qubits in
            the new register.

        Args:
            new_register: The new register to give the sequence.

        Returns:
            The sequence with the new register.
        """
        new_seq = type(self)(register=new_register, device=self._device)
        # Copy the variables to the new sequence
        new_seq._variables = self.declared_variables
        for call in self._calls[1:] + self._to_build_calls:
            if call.name == "config_detuning_map":
                warnings.warn(
                    "Switching the register of a sequence that configures"
                    " a detuning map. Please ensure that the new qubit"
                    " positions are still aligned.",
                    stacklevel=2,
                )
            getattr(new_seq, call.name)(*call.args, **call.kwargs)
        return new_seq

    def switch_device(
        self, new_device: DeviceType, strict: bool = False
    ) -> Sequence:
        """Replicate the sequence with a different device.

        This method is designed to replicate the sequence with as few changes
        to the original contents as possible.
        If the `strict` option is chosen, the device switch will fail whenever
        it cannot guarantee that the new sequence's contents will not be
        modified in the process.

        Args:
            new_device: The target device instance.
            strict: Enforce a strict match between devices and channels to
                guarantee the pulse sequence is left unchanged.

        Returns:
            The sequence on the new device, using the match channels of
            the former device declared in the sequence.
        """
        return switch_device(self, new_device, strict)

    @seq_decorators.block_if_measured
    def declare_channel(
        self,
        name: str,
        channel_id: str,
        initial_target: Optional[Union[QubitId, Collection[QubitId]]] = None,
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
        if name.startswith("dmm_"):
            raise ValueError(
                "Name starting by 'dmm_' are reserved for DMM channels."
            )
        if name in self._schedule:
            raise ValueError("The given name is already in use.")

        if channel_id not in self._device.channels:
            raise ValueError(f"No channel {channel_id} in the device.")

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
                    for t in cast(Collection, initial_target)
                )
            except TypeError:
                cond = isinstance(initial_target, Parametrized)
            if cond:
                raise TypeError("The initial_target cannot be parametrized")

        if ch.basis == "XY":
            if not self._in_xy:
                self.set_magnetic_field()
                self._in_xy = True
        else:
            self._in_ising = True
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
                    cast(Union[Collection, QubitId], initial_target), name
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
        if name in ("qubits", "seq_name", "json_dumps_options"):
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

    @seq_decorators.verify_parametrization
    @seq_decorators.block_if_measured
    def enable_eom_mode(
        self,
        channel: str,
        amp_on: Union[float, pm.TensorLike, Parametrized],
        detuning_on: Union[float, pm.TensorLike, Parametrized],
        optimal_detuning_off: Union[float, Parametrized] = 0.0,
        correct_phase_drift: bool = False,
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
            correct_phase_drift: Performs a phase shift to correct for the
                phase drift incurred while turning on the EOM mode.
        """
        if self.is_in_eom_mode(channel):
            raise RuntimeError(
                f"The '{channel}' channel is already in EOM mode."
            )

        channel_obj = self.declared_channels[channel]
        if not channel_obj.supports_eom():
            raise TypeError(f"Channel '{channel}' does not have an EOM.")

        detuning_off, switching_beams = self._process_eom_parameters(
            channel_obj, amp_on, detuning_on, optimal_detuning_off
        )
        if not self.is_parametrized():
            assert not isinstance(amp_on, Parametrized)
            amp_on_ = pm.AbstractArray(amp_on)
            assert not isinstance(detuning_on, Parametrized)
            detuning_on_ = pm.AbstractArray(detuning_on)
            assert not isinstance(detuning_off, Parametrized)
            detuning_off_ = pm.AbstractArray(detuning_off)

            phase_drift_params = _PhaseDriftParams(
                drift_rate=-detuning_off_,
                # enable_eom() calls wait for fall, so the block only
                # starts after fall time
                ti=self.get_duration(channel, include_fall_time=True),
            )
            self._schedule.enable_eom(
                channel,
                amp_on_,
                detuning_on_,
                detuning_off_,
                switching_beams,
            )
            if correct_phase_drift:
                buffer_slot = self._last(channel)
                drift = phase_drift_params.calc_phase_drift(buffer_slot.tf)
                self._phase_shift(
                    -float(drift),
                    *buffer_slot.targets,
                    basis=channel_obj.basis,
                )

        # Manually store the call to "enable_eom_mode" so that the updated
        # 'optimal_detuning_off' is stored
        call_container = (
            self._to_build_calls if self.is_parametrized() else self._calls
        )
        call_container.append(
            _Call(
                "enable_eom_mode",
                (),
                dict(
                    channel=channel,
                    amp_on=amp_on,
                    detuning_on=detuning_on,
                    optimal_detuning_off=(
                        detuning_off
                        if isinstance(detuning_off, Parametrized)
                        else float(detuning_off)
                    ),
                    correct_phase_drift=correct_phase_drift,
                ),
            )
        )

    @seq_decorators.store
    @seq_decorators.block_if_measured
    def disable_eom_mode(
        self, channel: str, correct_phase_drift: bool = False
    ) -> None:
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
            correct_phase_drift: Performs a phase shift to correct for the
                phase drift that occured since the last pulse (or the start of
                the EOM mode, if no pulse was added).
        """
        if not self.is_in_eom_mode(channel):
            raise RuntimeError(f"The '{channel}' channel is not in EOM mode.")
        if not self.is_parametrized():
            self._schedule.disable_eom(channel)
            if correct_phase_drift:
                ch_schedule = self._schedule[channel]
                # EOM mode has just been disabled, so tf is defined
                last_eom_block_tf = cast(int, ch_schedule.eom_blocks[-1].tf)
                drift_params = self._get_last_eom_pulse_phase_drift(channel)
                self._phase_shift(
                    -float(drift_params.calc_phase_drift(last_eom_block_tf)),
                    *ch_schedule[-1].targets,
                    basis=ch_schedule.channel_obj.basis,
                )

    @seq_decorators.verify_parametrization
    @seq_decorators.block_if_measured
    def modify_eom_setpoint(
        self,
        channel: str,
        amp_on: Union[float, pm.TensorLike, Parametrized],
        detuning_on: Union[float, pm.TensorLike, Parametrized],
        optimal_detuning_off: Union[float, Parametrized] = 0.0,
        correct_phase_drift: bool = False,
    ) -> None:
        """Modifies the setpoint of an ongoing EOM mode operation.

        Note:
            Modifying the EOM setpoint will automatically enforce a buffer.
            The detuning will go to the `detuning_off` value during
            this buffer. This buffer will not wait for pulses on other
            channels to finish, so calling `Sequence.align()` or
            `Sequence.delay()` beforehand is necessary to avoid eventual
            conflicts.

        Args:
            channel: The name of the channel currently in EOM mode.
            amp_on: The new amplitude of the EOM pulses (in rad/µs).
            detuning_on: The new detuning of the EOM pulses (in rad/µs).
            optimal_detuning_off: The new optimal value of detuning (in rad/µs)
                when there is no pulse being played. It will choose the closest
                value among the existing options.
            correct_phase_drift: Performs a phase shift to correct for the
                phase drift incurred while modifying the EOM setpoint.
        """
        if not self.is_in_eom_mode(channel):
            raise RuntimeError(f"The '{channel}' channel is not in EOM mode.")

        channel_obj = self.declared_channels[channel]
        detuning_off, switching_beams = self._process_eom_parameters(
            channel_obj, amp_on, detuning_on, optimal_detuning_off
        )

        if not self.is_parametrized():
            assert not isinstance(amp_on, Parametrized)
            amp_on_ = pm.AbstractArray(amp_on)
            assert not isinstance(detuning_on, Parametrized)
            detuning_on_ = pm.AbstractArray(detuning_on)
            assert not isinstance(detuning_off, Parametrized)
            detuning_off_ = pm.AbstractArray(detuning_off)

            self._schedule.disable_eom(channel, _skip_buffer=True)
            old_phase_drift_params = self._get_last_eom_pulse_phase_drift(
                channel
            )
            new_phase_drift_params = _PhaseDriftParams(
                drift_rate=-detuning_off_,
                ti=self.get_duration(channel, include_fall_time=False),
            )
            self._schedule.enable_eom(
                channel,
                amp_on_,
                detuning_on_,
                detuning_off_,
                switching_beams,
                _skip_wait_for_fall=True,
            )
            if correct_phase_drift:
                buffer_slot = self._last(channel)
                drift = old_phase_drift_params.calc_phase_drift(
                    buffer_slot.ti
                ) + new_phase_drift_params.calc_phase_drift(buffer_slot.tf)
                self._phase_shift(
                    -float(drift),
                    *buffer_slot.targets,
                    basis=channel_obj.basis,
                )

        # Manually store the call to "modify_eom_setpoint" so that the updated
        # 'optimal_detuning_off' is stored
        call_container = (
            self._to_build_calls if self.is_parametrized() else self._calls
        )
        call_container.append(
            _Call(
                "modify_eom_setpoint",
                (),
                dict(
                    channel=channel,
                    amp_on=amp_on,
                    detuning_on=detuning_on,
                    optimal_detuning_off=(
                        detuning_off
                        if isinstance(detuning_off, Parametrized)
                        else float(detuning_off)
                    ),
                    correct_phase_drift=correct_phase_drift,
                ),
            )
        )

    @seq_decorators.store
    @seq_decorators.mark_non_empty
    @seq_decorators.block_if_measured
    def add_eom_pulse(
        self,
        channel: str,
        duration: Union[int, Parametrized],
        phase: Union[float, pm.TensorLike, Parametrized],
        post_phase_shift: Union[float, Parametrized] = 0.0,
        protocol: PROTOCOLS = "min-delay",
        correct_phase_drift: bool = False,
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
            post_phase_shift: Optionally lets you add a phase shift (in rad)
                immediately after the end of the pulse.
            protocol: Stipulates how to deal with eventual conflicts with
                other channels (see `Sequence.add()` for more details).
            correct_phase_drift: Adjusts the phase to correct for the phase
                drift that occured since the last pulse (or the start of the
                EOM mode, if adding the first pulse). This effectively
                changes the phase of the EOM pulse, so an extra delay might
                be added to enforce the phase jump time.
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
                if isinstance(arg, Parametrized):
                    continue
                try:
                    if isinstance(arg, str):
                        raise TypeError
                    float(pm.AbstractArray(arg, dtype=float))
                except TypeError:
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
        phase_drift_params = (
            self._get_last_eom_pulse_phase_drift(channel)
            if correct_phase_drift
            else None
        )
        self._add(
            eom_pulse, channel, protocol, phase_drift_params=phase_drift_params
        )

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
        self._validate_channel(
            channel,
            block_eom_mode=True,
            block_if_slm=channel.startswith("dmm_"),
        )
        if isinstance(self.declared_channels[channel], DMM):
            raise ValueError(
                "`Sequence.add()` can't be used on a DMM channel. "
                "Use `Sequence.add_dmm_detuning()` instead."
            )
        self._add(pulse, channel, protocol)

    @seq_decorators.store
    @seq_decorators.mark_non_empty
    @seq_decorators.block_if_measured
    def add_dmm_detuning(
        self,
        waveform: Union[Waveform, Parametrized],
        dmm_name: str,
        protocol: PROTOCOLS = "no-delay",
    ) -> None:
        """Add a waveform to the detuning of a DMM.

        Args:
            waveform: The waveform to add to the detuning of the DMM.
            dmm_name: The name of the DMM channel to modulate.
            protocol: Stipulates how to deal with
                eventual conflicts with other channels, specifically in terms
                of having multiple channels act on the same target
                simultaneously (defaults to "no-delay").

                - ``'min-delay'``: Before adding the pulse, introduces the
                  smallest possible delay that avoids all exisiting conflicts.
                - ``'no-delay'``: Adds the pulse to the channel, regardless of
                  existing conflicts.
                - ``'wait-for-all'``: Before adding the pulse, adds a delay
                  that idles the channel until the end of the other channels'
                  latest pulse.
        """
        self._validate_channel(dmm_name, block_if_slm=True)
        if not isinstance(self.declared_channels[dmm_name], DMM):
            raise ValueError(f"'{dmm_name}' is not the name of a DMM channel.")
        self._add(
            Pulse.ConstantAmplitude(0, waveform, 0),
            dmm_name,
            protocol,
        )

    @seq_decorators.store
    def target(
        self,
        qubits: Union[QubitId, Collection[QubitId]],
        channel: str,
    ) -> None:
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits: The new target for this channel. Must correspond to a
                qubit ID in device or a collection of qubit IDs, when
                multi-qubit addressing is possible.
            channel: The channel's name provided when declared. Must be
                a channel with 'Local' addressing.
        """
        self._target(qubits, channel)

    @seq_decorators.store
    def target_index(
        self,
        qubits: Union[int, Collection[int], Parametrized],
        channel: str,
    ) -> None:
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits: The new target for this channel. Must correspond to a
                qubit index or an collection of qubit indices, when multi-qubit
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
        at_rest: bool = False,
    ) -> None:
        """Idles a given channel for a specific duration.

        Args:
            duration: Time to delay (in ns).
            channel: The channel's name provided when declared.
            at_rest: Whether to wait until the previous pulse on the
                channel has finished (including output modulation) before
                starting the delay.

        Note:
            Delays added automatically by other instructions will generally
            take into account the output modulation.
        """
        self._delay(duration, channel, at_rest)

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
        elif basis not in self.get_addressed_bases():
            warnings.warn(
                f"The desired measurement basis '{basis}' is not being "
                "addressed by any channel in the sequence.",
                stacklevel=2,
            )

        if self.is_parametrized():
            self._param_measurement = basis
        else:
            self._measurement = basis

    @seq_decorators.store
    def phase_shift(
        self,
        phi: float | Parametrized,
        *targets: QubitId,
        basis: str = "digital",
    ) -> None:
        r"""Shifts the phase of a qubit's reference by 'phi', on a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi: The intended phase shift (in rad).
            targets: The ids of the qubits to apply the phase shift to.
            basis: The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.
        """
        self._phase_shift(phi, *targets, basis=basis)

    @seq_decorators.store
    def phase_shift_index(
        self,
        phi: float | Parametrized,
        *targets: int | Parametrized,
        basis: str = "digital",
    ) -> None:
        r"""Shifts the phase of a qubit's reference by 'phi', on a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi: The intended phase shift (in rad).
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
    def align(self, *channels: str, at_rest: bool = True) -> None:
        """Aligns multiple channels in time.

        Introduces delays that align the provided channels with the one that
        finished the latest, such that the next action added to any of them
        will start right after the latest channel has finished.

        Args:
            channels: The names of the channels to align, as given upon
                declaration.
            at_rest: Whether to consider the output modulation of a channel's
                contents when determining that it has finished.
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
            id: self.get_duration(id, include_fall_time=at_rest)
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
        **vars: Union[ArrayLike, pm.TensorLike, float, int],
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

        # Recreate the base sequence (what remains)
        temp_seq = type(seq)(register=seq._register, device=seq._device)
        assert not seq._to_build_calls
        for call in seq._calls[1:]:
            getattr(temp_seq, call.name)(*call.args, **call.kwargs)
        seq = temp_seq

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

        Warning:
            This method has been deprecated and is scheduled for removal
            in Pulser v1.0.0. For sequence serialization and deserialization,
            use ``Sequence.to_abstract_repr()`` and
            ``Sequence.from_abstract_repr()`` instead.

        See Also:
            ``json.dumps``: Built-in function for serialization to a JSON
            formatted string.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                DeprecationWarning(
                    "`Sequence.serialize()` and `Sequence.deserialize()` have "
                    "been deprecated and will be removed in Pulser v1.0.0. "
                    "Use `Sequence.to_abstract_repr()` and "
                    "`Sequence.from_abstract_repr()` instead."
                )
            )

        return self._serialize(**kwargs)

    def _serialize(self, **kwargs: Any) -> str:
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
        self,
        seq_name: str = "pulser-exported",
        json_dumps_options: dict[str, Any] = {},
        skip_validation: bool = False,
        **defaults: Any,
    ) -> str:
        """Serializes the Sequence into an abstract JSON object.

        Keyword Args:
            seq_name (str): A name for the sequence. If not defined, defaults
                to "pulser-exported".
            json_dumps_options: A mapping between optional parameters of
                ``json.dumps()`` (as string) and their value (parameter cannot
                be "cls").
            skip_validation: Whether to skip the validation of the serialized
                sequence against the abstract representation's JSON schema.
                Skipping the validation is useful to cut down on execution
                time, as this step takes significantly longer than the
                serialization itself; it is also low risk, as the validation
                is only defensively checking that there are no bugs in the
                serialized sequence.
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
        """
        try:
            return serialize_abstract_sequence(
                self,
                seq_name=seq_name,
                json_dumps_options=json_dumps_options,
                skip_validation=skip_validation,
                **defaults,
            )
        except jsonschema.exceptions.ValidationError as e:
            if self.is_parametrized():
                raise AbstractReprError(
                    "The serialization of the parametrized sequence failed, "
                    "potentially due to an error that only appears at build "
                    "time. Check that no errors appear when building with "
                    "`Sequence.build()` or when providing the `defaults` to "
                    "`Sequence.to_abstract_repr()`."
                ) from e
            raise e  # pragma: no cover

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

        Warning:
            This method has been deprecated and is scheduled for removal
            in Pulser v1.0.0. For sequence serialization and deserialization,
            use ``Sequence.to_abstract_repr()`` and
            ``Sequence.from_abstract_repr()`` instead.

        See Also:
            ``json.loads``: Built-in function for deserialization from a JSON
            formatted string.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                DeprecationWarning(
                    "`Sequence.serialize()` and `Sequence.deserialize()` have "
                    "been deprecated and will be removed in Pulser v1.0.0. "
                    "Use `Sequence.to_abstract_repr()` and "
                    "`Sequence.from_abstract_repr()` instead."
                )
            )
        return Sequence._deserialize(obj, **kwargs)

    @staticmethod
    def _deserialize(obj: str, **kwargs: Any) -> Sequence:
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
        if not isinstance(obj, str):
            raise TypeError(
                "The serialized sequence must be given as a string. "
                f"Instead, got object of type {type(obj)}."
            )
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
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized sequence must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )
        return deserialize_abstract_sequence(obj_str)

    @seq_decorators.screen
    def draw(
        self,
        mode: str = "input+output",
        as_phase_modulated: bool = False,
        draw_phase_area: bool = False,
        draw_interp_pts: bool = True,
        draw_phase_shifts: bool = False,
        draw_register: bool = False,
        draw_phase_curve: bool = False,
        draw_detuning_maps: bool = False,
        draw_qubit_amp: bool = False,
        draw_qubit_det: bool = False,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
        show: bool = True,
    ) -> None:
        """Draws the sequence in its current state.

        Args:
            mode: The curves to draw. 'input'
                draws only the programmed curves, 'output' the excepted curves
                after modulation. 'input+output' will draw both curves except
                for channels without a defined modulation bandwidth, in which
                case only the input is drawn.
            as_phase_modulated: Instead of displaying the detuning and phase
                offsets, displays the equivalent phase modulation.
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
            draw_detuning_maps: Whether to draw the detuning maps applied on
                the qubits of the register of the sequence. Shown before the
                pulse sequence, defaults to False.
            draw_qubit_amp: Draws the amplitude seen by the qubits locally
                after the drawing of the sequence.
            draw_qubit_det: Draws the detuning seen by the qubits locally after
                the drawing of the sequence.
            fig_name: The name on which to save the figures. Figures are saved
                if `fig_name` is not None. If `draw_register`, `draw_qubit_amp`
                and `draw_qubit_det` are False, only the pulses are saved, with
                no suffix. If one of them is True, the pulses will be saved
                with a suffix ``_pulses``. If draw_register is True, the
                register is saved in another figure, with a suffix
                ``_register`` in the file name. If `draw_qubit_amp` or
                `draw_qubit_det` is True, the evolution of the quantities along
                time for group of qubits is saved in another figure with the
                prefix '_per_qubit', and the group of qubits having same
                evolution of quantities along time are saved in a figure with
                suffix '_per_qubit_legend'.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.
            show: Whether or not to call `plt.show()` before returning. When
                combining this plot with other ones in a single figure, one may
                need to set this flag to False.

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
        fig_reg, fig, fig_qubit, fig_legend = self._plot(
            draw_phase_area=draw_phase_area,
            draw_interp_pts=draw_interp_pts,
            draw_phase_shifts=draw_phase_shifts,
            draw_register=draw_register,
            draw_input="input" in mode,
            draw_modulation="output" in mode,
            draw_phase_curve=draw_phase_curve,
            draw_detuning_maps=draw_detuning_maps,
            draw_qubit_amp=draw_qubit_amp,
            draw_qubit_det=draw_qubit_det,
            phase_modulated=as_phase_modulated,
        )
        if fig_name is not None:
            name, ext = os.path.splitext(fig_name)
            suffix = (
                "_pulses"
                if all(fig is None for fig in (fig_reg, fig_qubit, fig_legend))
                else ""
            )
            fig.savefig(name + suffix + ext, **kwargs_savefig)
            if fig_reg is not None:
                fig_reg.savefig(name + "_register" + ext, **kwargs_savefig)
            if fig_qubit is not None:
                fig_qubit.savefig(name + "_per_qubit" + ext, **kwargs_savefig)
                if fig_legend is not None:
                    fig_qubit.savefig(
                        name + "_per_qubit_legend" + ext, **kwargs_savefig
                    )

        if show:
            plt.show()

    def _plot(
        self, **draw_options: bool
    ) -> tuple[Figure | None, Figure, Figure | None, Figure | None]:
        return draw_sequence(self, **draw_options)

    def _modulate_slm_mask_dmm(self, duration: int, max_amp: float) -> None:
        if self._slm_mask_dmm is not None:
            bottom_detuning = cast(
                DMM, self.declared_channels[self._slm_mask_dmm]
            ).bottom_detuning
            total_bottom_detuning = cast(
                DMM, self.declared_channels[self._slm_mask_dmm]
            ).total_bottom_detuning
            min_det = -10 * max_amp
            if bottom_detuning and min_det < bottom_detuning:
                min_det = bottom_detuning
            if (
                total_bottom_detuning
                and min_det * len(set(self._slm_mask_targets))
                < total_bottom_detuning
            ):
                min_det = total_bottom_detuning / len(
                    set(self._slm_mask_targets)
                )
            cast(
                _DMMSchedule, self._schedule[self._slm_mask_dmm]
            )._waiting_for_first_pulse = False
            self._add(
                Pulse.ConstantPulse(duration, 0, min_det, 0),
                self._slm_mask_dmm,
                "no-delay",
            )

    def _add(
        self,
        pulse: Union[Pulse, Parametrized],
        channel: str,
        protocol: PROTOCOLS,
        phase_drift_params: _PhaseDriftParams | None = None,
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
        if isinstance(channel_obj, DMM):
            phase_ref = None
        elif len(ph_refs) != 1:
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

        self._schedule.add_pulse(
            pulse,
            channel,
            phase_barriers,
            protocol,
            phase_drift_params=phase_drift_params,
        )

        new_pulse_slot = self._last(channel)
        for qubit in last.targets:
            self._basis_ref[basis][qubit].update_last_used(new_pulse_slot.tf)

        total_phase_shift = pulse.post_phase_shift
        if phase_drift_params:
            # The phase correction done to the EOM pulse's phase must
            # also be done to the phase shift, as the phase reference is
            # effectively changed by -drift
            total_phase_shift -= float(
                phase_drift_params.calc_phase_drift(new_pulse_slot.ti)
            )
        if total_phase_shift != 0.0:
            self._phase_shift(total_phase_shift, *last.targets, basis=basis)
        if (
            self._in_ising
            and self._slm_mask_dmm
            and cast(
                _DMMSchedule, self._schedule[self._slm_mask_dmm]
            )._waiting_for_first_pulse
            and channel_obj.addressing == "Global"
            and not _ChannelSchedule.is_detuned_delay(pulse)
            and not isinstance(channel_obj, DMM)
        ):
            self._modulate_slm_mask_dmm(
                self._schedule[channel].get_duration(),
                np.max(pulse.amplitude.samples),
            )

    @seq_decorators.block_if_measured
    def _target(
        self,
        qubits: Union[Collection[QubitId], QubitId, Parametrized],
        channel: str,
        _index: bool = False,
    ) -> None:
        self._validate_channel(channel, block_eom_mode=True)
        channel_obj = self._schedule[channel].channel_obj
        if isinstance(qubits, pm.AbstractArray):
            qubits = qubits.tolist()
        try:
            qubits_set = (
                set(cast(Collection, qubits))
                if not isinstance(qubits, str)
                else {qubits}
            )
        except TypeError:
            qubits_set = {qubits}

        if not qubits_set:
            raise ValueError(
                "Need at least one qubit to target but none were given."
            )

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
                float(self._basis_ref[basis][q].phase.last_phase)
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
                try:
                    return {
                        self._register.qubit_ids[
                            int(index)  # type: ignore[arg-type]
                        ]
                        for index in qubits
                    }
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
    def _delay(
        self,
        duration: Union[int, Parametrized],
        channel: str,
        at_rest: bool = False,
    ) -> None:
        self._validate_channel(channel, block_if_slm=True)
        if self.is_parametrized():
            return
        if at_rest:
            self._schedule.wait_for_fall(channel)
        if not duration:
            return
        self._schedule.add_delay(cast(int, duration), channel)

    def _phase_shift(
        self,
        phi: float | Parametrized,
        *targets: QubitId | Parametrized,
        basis: str,
        _index: bool = False,
    ) -> None:
        if basis not in self._basis_ref:
            raise ValueError(
                f"No declared channel targets the given 'basis' ('{basis}')."
            )
        target_ids = self._check_qubits_give_ids(*targets, _index=_index)

        if not self.is_parametrized():
            phi = float(cast(float, phi))
            for qubit in target_ids:
                self._basis_ref[basis][qubit].increment_phase(phi)

    def _get_last_eom_pulse_phase_drift(
        self, channel: str
    ) -> _PhaseDriftParams:
        eom_settings = self._schedule[channel].eom_blocks[-1]
        try:
            last_pulse_tf = (
                self._schedule[channel]
                .last_pulse_slot(ignore_detuned_delay=True)
                .tf
            )
        except RuntimeError:
            # There is no previous pulse
            last_pulse_tf = 0
        return _PhaseDriftParams(
            drift_rate=-eom_settings.detuning_off,
            ti=max(eom_settings.ti, last_pulse_tf),
        )

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
        self,
        channel: str,
        block_eom_mode: bool = False,
        block_if_slm: bool = False,
    ) -> None:
        if isinstance(channel, Parametrized):
            raise NotImplementedError(
                "Using parametrized objects or variables to refer to channels "
                "is not supported."
            )
        if channel not in self.declared_channels:
            raise ValueError("Use the name of a declared channel.")
        if block_eom_mode and self.is_in_eom_mode(channel):
            raise RuntimeError("The chosen channel is in EOM mode.")
        if (
            block_if_slm
            and channel == self._slm_mask_dmm
            and cast(
                _DMMSchedule, self._schedule[self._slm_mask_dmm]
            )._waiting_for_first_pulse
        ):
            raise ValueError(
                "You should add a Pulse to a Global Channel prior to"
                " modulating the DMM used for the SLM Mask."
            )

    def _validate_and_adjust_pulse(
        self,
        pulse: Pulse,
        channel: str,
        phase_ref: float | None = None,
    ) -> Pulse:
        # Get the channel object and its detuning map if the channel is a DMM
        channel_obj: Channel
        # Detuning map is None if channel is not DMM
        detuning_map: DetuningMap | None = None
        if channel in self._schedule:
            # channel name can refer to a Channel or a DMM object
            # Get channel object
            channel_obj = self._schedule[channel].channel_obj
            # Get its associated detuning map if channel is a DMM
            if isinstance(channel_obj, DMM):
                # stored in _DMMSchedule with channel object
                detuning_map = cast(
                    _DMMSchedule, self._schedule[channel]
                ).detuning_map
                # Ignore the phase reference for DMM
                assert phase_ref is None
        else:
            # If channel name can't be found among _schedule keys, the
            # Sequence is parametrized and channel is a dmm_name
            dmm_id = _dmm_id_from_name(channel)
            # Get channel object
            channel_obj = self.device.dmm_channels[dmm_id]
            # Go over the calls to find the associated detuning map
            declared_dmms: list[str] = []
            for call in self._calls[1:] + self._to_build_calls:
                if (
                    call.name == "config_detuning_map"
                    or call.name == "config_slm_mask"
                ):
                    # Extract dmm_id, detuning map of call
                    call_id, call_det_map = self._get_dmm_id_detuning_map(call)
                    # Quit if dmm_name of call matches with channel
                    call_name = _get_dmm_name(call_id, declared_dmms)
                    declared_dmms.append(call_name)
                    if call_name == channel:
                        detuning_map = call_det_map
                        break
            assert detuning_map is not None
        if detuning_map is None:
            # channel points to a Channel object
            channel_obj.validate_pulse(pulse)
        else:
            # channel points to a DMM object
            cast(DMM, channel_obj).validate_pulse(pulse, detuning_map)
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

    def _process_eom_parameters(
        self,
        channel_obj: Channel,
        amp_on: Union[float, pm.TensorLike, Parametrized],
        detuning_on: Union[float, pm.TensorLike, Parametrized],
        optimal_detuning_off: Union[float, Parametrized],
    ) -> tuple[
        float | pm.AbstractArray | Parametrized, tuple[RydbergBeam, ...]
    ]:
        on_pulse = Pulse.ConstantPulse(
            channel_obj.min_duration, amp_on, detuning_on, 0.0
        )
        stored_opt_detuning_off: float | pm.AbstractArray | Parametrized = (
            optimal_detuning_off
        )
        switching_beams: tuple[RydbergBeam, ...] = ()
        if not isinstance(on_pulse, Parametrized):
            channel_obj.validate_pulse(on_pulse)
            assert not isinstance(amp_on, Parametrized)
            assert not isinstance(detuning_on, Parametrized)
            eom_config = cast(RydbergEOM, channel_obj.eom_config)
            if not isinstance(optimal_detuning_off, Parametrized):
                (
                    detuning_off,
                    switching_beams,
                ) = eom_config.calculate_detuning_off(
                    amp_on,
                    detuning_on,
                    float(optimal_detuning_off),
                    return_switching_beams=True,
                )
                off_pulse = Pulse.ConstantPulse(
                    channel_obj.min_duration, 0.0, detuning_off, 0.0
                )
                channel_obj.validate_pulse(off_pulse)
                # Update optimal_detuning_off to match the chosen detuning_off
                # This minimizes the changes to the sequence when the device
                # is switched
                stored_opt_detuning_off = detuning_off
        return stored_opt_detuning_off, switching_beams

    def _reset_parametrized(self) -> None:
        """Resets all attributes related to parametrization."""
        # Signals the sequence as actively "building" ie not parametrized
        self._building = True
        self._param_measurement = ""
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
