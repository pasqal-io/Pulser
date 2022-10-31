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

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast, overload

import jsonschema

import pulser
import pulser.devices as devices
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

if TYPE_CHECKING:  # pragma: no cover
    from pulser.register.base_register import BaseRegister
    from pulser.sequence import Sequence

with open(Path(__file__).parent / "schema.json") as f:
    schema = json.load(f)

VARIABLE_TYPE_MAP = {"int": int, "float": float}

ExpReturnType = Union[int, float, ParamObj]


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
    jsonschema.validate(instance=obj, schema=schema)

    # Device
    device_name = obj["device"]
    device = getattr(devices, device_name)

    # Register Layout
    layout = (
        RegisterLayout(obj["layout"]["coordinates"])
        if "layout" in obj
        else None
    )

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
