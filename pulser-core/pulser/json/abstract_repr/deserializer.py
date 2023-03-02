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
"""Deserializer from JSON in the abstract representation."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type, Union, cast, overload

import jsonschema

import pulser
import pulser.devices as devices
from pulser.channels import Microwave, Raman, Rydberg
from pulser.channels.base_channel import Channel
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices import Device, VirtualDevice
from pulser.devices._device_datacls import BaseDevice
from pulser.json.abstract_repr.signatures import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
)
from pulser.json.exceptions import AbstractReprError
from pulser.parametrized import ParamObj, Variable
from pulser.pulse import Pulse
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register import Register
from pulser.register.register_layout import RegisterLayout
from pulser.waveforms import (
    BlackmanWaveform,
    CompositeWaveform,
    ConstantWaveform,
    CustomWaveform,
    InterpolatedWaveform,
    KaiserWaveform,
    RampWaveform,
    Waveform,
)

if TYPE_CHECKING:
    from pulser.register.base_register import BaseRegister
    from pulser.sequence import Sequence


VARIABLE_TYPE_MAP = {"int": int, "float": float}

ExpReturnType = Union[int, float, ParamObj]

schemas_path = Path(__file__).parent / "schemas"
schemas = {}
for obj_type in ("device", "sequence"):
    with open(schemas_path / f"{obj_type}-schema.json") as f:
        schemas[obj_type] = json.load(f)

resolver = jsonschema.validators.RefResolver(
    base_uri=f"{schemas_path.resolve().as_uri()}/",
    referrer=schemas["sequence"],
)


@overload
def _deserialize_parameter(param: int, vars: dict[str, Variable]) -> int:
    pass


@overload
def _deserialize_parameter(param: float, vars: dict[str, Variable]) -> float:
    pass


@overload
def _deserialize_parameter(
    param: dict[str, str], vars: dict[str, Variable]
) -> Variable:
    pass


def _deserialize_parameter(
    param: Union[int, float, dict[str, Any]],
    vars: dict[str, Variable],
) -> Union[ExpReturnType, Variable]:
    """Deserialize a parameterized object.

    A parameter can be either a literal, a variable or an expression.
    In the first case, return the literal. Otherwise, return a reference
    to the variable, or build an expression referencing variables.

    Args:
        param: The JSON parametrized object to deserialize
        vars: The references to the sequence variables

    Returns:
        A literal (int | float), a ``Variable``, or a ``ParamObj``.
    """
    if not isinstance(param, dict):
        # This is a literal
        return param

    if "variable" in param:
        # This is a reference to a variable.
        if param["variable"] not in vars:
            raise AbstractReprError(
                f"Variable '{param['variable']}' used in operations "
                "but not found in declared variables."
            )
        return vars[param["variable"]]

    if "expression" not in param:
        # Can't deserialize param if it is a dict without a
        # `variable` or an `expression` key
        raise AbstractReprError(
            f"Parameter '{param}' is neither a literal nor "
            "a variable or an expression."
        )

    # This is a unary or a binary expression
    expression = (
        param["expression"] if param["expression"] != "div" else "truediv"
    )

    if expression in UNARY_OPERATORS:
        return cast(
            ExpReturnType,
            UNARY_OPERATORS[expression](
                _deserialize_parameter(param["lhs"], vars)
            ),
        )
    elif expression in BINARY_OPERATORS:
        return cast(
            ExpReturnType,
            BINARY_OPERATORS[expression](
                _deserialize_parameter(param["lhs"], vars),
                _deserialize_parameter(param["rhs"], vars),
            ),
        )
    else:
        raise AbstractReprError(f"Expression '{param['expression']}' invalid.")


def _deserialize_waveform(obj: dict, vars: dict) -> Waveform:
    if obj["kind"] == "constant":
        return ConstantWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            value=_deserialize_parameter(obj["value"], vars),
        )
    if obj["kind"] == "ramp":
        return RampWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            start=_deserialize_parameter(obj["start"], vars),
            stop=_deserialize_parameter(obj["stop"], vars),
        )
    if obj["kind"] == "blackman":
        return BlackmanWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            area=_deserialize_parameter(obj["area"], vars),
        )
    if obj["kind"] == "blackman_max":
        return BlackmanWaveform.from_max_val(
            max_val=_deserialize_parameter(obj["max_val"], vars),
            area=_deserialize_parameter(obj["area"], vars),
        )
    if obj["kind"] == "interpolated":
        return InterpolatedWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            values=_deserialize_parameter(obj["values"], vars),
            times=_deserialize_parameter(obj["times"], vars),
        )
    if obj["kind"] == "kaiser":
        return KaiserWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            area=_deserialize_parameter(obj["area"], vars),
            beta=_deserialize_parameter(obj["beta"], vars),
        )
    if obj["kind"] == "kaiser_max":
        return KaiserWaveform.from_max_val(
            max_val=_deserialize_parameter(obj["max_val"], vars),
            area=_deserialize_parameter(obj["area"], vars),
            beta=_deserialize_parameter(obj["beta"], vars),
        )
    if obj["kind"] == "composite":
        wfs = [_deserialize_waveform(wf, vars) for wf in obj["waveforms"]]
        return CompositeWaveform(*wfs)
    if obj["kind"] == "custom":
        return CustomWaveform(
            samples=_deserialize_parameter(obj["samples"], vars)
        )

    raise AbstractReprError("The object does not encode a known waveform.")


def _deserialize_operation(seq: Sequence, op: dict, vars: dict) -> None:
    if op["op"] == "target":
        seq.target_index(
            qubits=_deserialize_parameter(op["target"], vars),
            channel=op["channel"],
        )
    elif op["op"] == "align":
        seq.align(*op["channels"])
    elif op["op"] == "delay":
        seq.delay(
            duration=_deserialize_parameter(op["time"], vars),
            channel=op["channel"],
        )
    elif op["op"] == "phase_shift":
        seq.phase_shift_index(
            _deserialize_parameter(op["phi"], vars),
            *[_deserialize_parameter(t, vars) for t in op["targets"]],
        )
    elif op["op"] == "pulse":
        pulse = Pulse(
            amplitude=_deserialize_waveform(op["amplitude"], vars),
            detuning=_deserialize_waveform(op["detuning"], vars),
            phase=_deserialize_parameter(op["phase"], vars),
            post_phase_shift=_deserialize_parameter(
                op["post_phase_shift"], vars
            ),
        )
        seq.add(
            pulse=pulse,
            channel=op["channel"],
            protocol=op["protocol"],
        )
    elif op["op"] == "enable_eom_mode":
        seq.enable_eom_mode(
            channel=op["channel"],
            amp_on=_deserialize_parameter(op["amp_on"], vars),
            detuning_on=_deserialize_parameter(op["detuning_on"], vars),
            optimal_detuning_off=_deserialize_parameter(
                op["optimal_detuning_off"], vars
            ),
        )
    elif op["op"] == "add_eom_pulse":
        seq.add_eom_pulse(
            channel=op["channel"],
            duration=_deserialize_parameter(op["duration"], vars),
            phase=_deserialize_parameter(op["phase"], vars),
            post_phase_shift=_deserialize_parameter(
                op["post_phase_shift"], vars
            ),
            protocol=op["protocol"],
        )
    elif op["op"] == "disable_eom_mode":
        seq.disable_eom_mode(channel=op["channel"])


def _deserialize_channel(obj: dict[str, Any]) -> Channel:
    params: dict[str, Any] = {}
    channel_cls: Type[Channel]
    if obj["basis"] == "ground-rydberg":
        channel_cls = Rydberg
        params["eom_config"] = None
        if obj["eom_config"] is not None:
            data = obj["eom_config"]
            params["eom_config"] = RydbergEOM(
                mod_bandwidth=data["mod_bandwidth"],
                limiting_beam=RydbergBeam[data["limiting_beam"]],
                max_limiting_amp=data["max_limiting_amp"],
                intermediate_detuning=data["intermediate_detuning"],
                controlled_beams=tuple(
                    RydbergBeam[beam] for beam in data["controlled_beams"]
                ),
            )
    elif obj["basis"] == "digital":
        channel_cls = Raman
    elif obj["basis"] == "XY":
        channel_cls = Microwave

    for param in dataclasses.fields(channel_cls):
        if param.init and param.name != "eom_config":
            params[param.name] = obj[param.name]
    return channel_cls(**params)


def _deserialize_layout(layout_obj: dict[str, Any]) -> RegisterLayout:
    return RegisterLayout(
        layout_obj["coordinates"], slug=layout_obj.get("slug")
    )


def _deserialize_device_object(obj: dict[str, Any]) -> Device | VirtualDevice:
    device_cls: Type[Device] | Type[VirtualDevice] = (
        VirtualDevice if obj["is_virtual"] else Device
    )
    ch_ids = []
    ch_objs = []
    for ch in obj["channels"]:
        ch_ids.append(ch["id"])
        ch_objs.append(_deserialize_channel(ch))
    params: dict[str, Any] = dict(
        channel_ids=tuple(ch_ids), channel_objects=tuple(ch_objs)
    )
    ex_params = ("channel_objects", "channel_ids")
    for param in dataclasses.fields(device_cls):
        if not param.init or param.name in ex_params:
            continue
        if param.name == "pre_calibrated_layouts":
            key = "pre_calibrated_layouts"
            params[key] = tuple(
                _deserialize_layout(layout) for layout in obj[key]
            )
        else:
            params[param.name] = obj[param.name]
    return device_cls(**params)


def deserialize_abstract_sequence(obj_str: str) -> Sequence:
    """Deserialize a sequence from an abstract JSON object.

    Args:
        obj_str: the JSON string representing the sequence encoded
            in the abstract JSON format.

    Returns:
        Sequence: The Pulser sequence.
    """
    obj = json.loads(obj_str)

    # Validate the format of the data against the JSON schema.
    jsonschema.validate(
        instance=obj, schema=schemas["sequence"], resolver=resolver
    )

    # Device
    if isinstance(obj["device"], str):
        device_name = obj["device"]
        device = getattr(devices, device_name)
    else:
        device = _deserialize_device_object(obj["device"])

    # Register Layout
    layout = _deserialize_layout(obj["layout"]) if "layout" in obj else None

    # Register
    reg: Union[BaseRegister, MappableRegister]
    qubits = obj["register"]
    if {"name", "x", "y"} == qubits[0].keys():
        # Regular register
        coords = [(q["x"], q["y"]) for q in qubits]
        qubit_ids = [q["name"] for q in qubits]
        if layout:
            trap_ids = layout.get_traps_from_coordinates(*coords)
            reg = layout.define_register(*trap_ids, qubit_ids=qubit_ids)
        else:
            reg = Register(dict(zip(qubit_ids, coords)))
    else:
        # Mappable register
        assert (
            layout is not None
        ), "Layout must be defined in a MappableRegister."
        reg = MappableRegister(layout, *(d["qid"] for d in qubits))

    seq = pulser.Sequence(reg, device)

    # Channels
    for name, channel_id in obj["channels"].items():
        seq.declare_channel(name, channel_id)

    # Magnetic field
    if "magnetic_field" in obj:
        seq.set_magnetic_field(*obj["magnetic_field"])

    # SLM Mask
    if "slm_mask_targets" in obj:
        seq.config_slm_mask(obj["slm_mask_targets"])

    # Variables
    vars = {}
    for name, desc in obj["variables"].items():
        v = seq.declare_variable(
            cast(str, name),
            size=len(desc["value"]),
            dtype=VARIABLE_TYPE_MAP[desc["type"]],
        )
        vars[name] = v

    # Operations
    for op in obj["operations"]:
        _deserialize_operation(seq, op, vars)

    # Measurement
    if obj["measurement"] is not None:
        seq.measure(obj["measurement"])

    return seq


def deserialize_device(obj_str: str) -> BaseDevice:
    """Deserialize a device from an abstract JSON object.

    Args:
        obj_str: the JSON string representing the device encoded
            in the abstract JSON format.

    Returns:
        BaseDevice: The Pulser device.
    """
    obj = json.loads(obj_str)
    # Validate the format of the data against the JSON schema.
    jsonschema.validate(instance=obj, schema=schemas["device"])
    return _deserialize_device_object(obj)
