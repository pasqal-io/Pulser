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
from typing import TYPE_CHECKING, cast

import jsonschema

import pulser
import pulser.devices as devices
from pulser.json.exceptions import AbstractReprError
from pulser.pulse import Pulse
from pulser.register.register import Register
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
    from pulser.sequence import Sequence

with open("pulser-core/pulser/json/abstract_repr/schema.json") as f:
    schema = json.load(f)


VARIABLE_TYPE_MAP = {"int": int, "float": float}


def _deserialize_abstract_waveform(obj: dict) -> Waveform:
    if obj["kind"] == "constant":
        return ConstantWaveform(obj["duration"], obj["value"])
    if obj["kind"] == "ramp":
        return RampWaveform(obj["duration"], obj["start"], obj["stop"])
    if obj["kind"] == "blackman":
        return BlackmanWaveform(obj["duration"], obj["area"])
    if obj["kind"] == "blackman_max":
        return BlackmanWaveform.from_max_val(obj["max_val"], obj["area"])
    if obj["kind"] == "interpolated":
        return InterpolatedWaveform(
            obj["duration"], obj["values"], obj["times"]
        )
    if obj["kind"] == "kaiser":
        return KaiserWaveform(obj["duration"], obj["area"], obj["beta"])
    if obj["kind"] == "kaiser_max":
        return KaiserWaveform.from_max_val(
            obj["max_val"], obj["area"], obj["beta"]
        )
    if obj["kind"] == "composite":
        wfs = [_deserialize_abstract_waveform(wf) for wf in obj["waveforms"]]
        return CompositeWaveform(*wfs)
    if obj["kind"] == "custom":
        return CustomWaveform(obj["samples"])

    raise AbstractReprError("The object does not encode a known waveform.")


def _deserialize_abstract_operation(seq: Sequence, op: dict) -> None:
    if op["op"] == "target":
        seq.target_index(
            qubits=op["target"],
            channel=op["channel"],
        )
    elif op["op"] == "align":
        seq.align(*op["channels"])
    elif op["op"] == "delay":
        seq.delay(
            duration=op["time"],
            channel=op["channel"],
        )
    elif op["op"] == "phase_shift":
        seq.phase_shift_index(
            op["phi"],
            *op["targets"],
        )
    elif op["op"] == "pulse":
        pulse = Pulse(
            amplitude=_deserialize_abstract_waveform(op["amplitude"]),
            detuning=_deserialize_abstract_waveform(op["detuning"]),
            phase=op["phase"],
            post_phase_shift=op["post_phase_shift"],
        )
        seq.add(
            pulse=pulse,
            channel=op["channel"],
            protocol=op["protocol"],
        )


def deserialize_abstract_sequence(obj_str: str) -> Sequence:
    """Deserialize a sequence from an abstract JSON object.

    Args:
        obj_str (str): the JSON string representing the sequence encoded
            in the abstract JSON format.

    Returns:
        Sequence: The Pulser sequence.
    """
    pass

    obj = json.loads(obj_str)

    # Validate the format of the data against the JSON schema.
    jsonschema.validate(instance=obj, schema=schema)

    # Device
    device_name = obj["device"]
    device = getattr(devices, device_name)

    # Register
    qubits = obj["register"]
    reg = Register({q["name"]: (q["x"], q["y"]) for q in qubits})

    seq = pulser.Sequence(reg, device)

    # Channels
    for name, channel_id in obj["channels"].items():
        seq.declare_channel(name, channel_id)

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
        _deserialize_abstract_operation(seq, op)

    # Measurement
    if obj["measurement"] is not None:
        seq.measure(obj["measurement"])

    return seq
