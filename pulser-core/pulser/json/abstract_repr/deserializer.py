import json
from typing import Dict

import jsonschema

import pulser.devices as devices
from pulser.pulse import Pulse
from pulser.register.register import Register
from pulser.sequence import Sequence
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

with open("pulser-core/pulser/json/abstract_repr/schema.json") as f:
    schema = json.load(f)


VARIABLE_TYPE_MAP = {"int": int, "float": float}


def _deserialize_abstract_waveform(obj: Dict) -> Waveform:

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


def _deserialize_abstract_operation(seq, op: Dict):
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

    seq = Sequence(reg, device)

    # Channels
    for name, channel_id in obj["channels"].items():
        seq.declare_channel(name, channel_id)

    # Variables
    vars = {}
    for name, desc in obj["variables"].items():
        v = seq.declare_variable(
            name, len(desc["value"]), VARIABLE_TYPE_MAP[desc["type"]]
        )
        vars[name] = v

    # Operations
    for op in obj["operations"]:
        _deserialize_abstract_operation(seq, op)

    # Measurement
    if obj["measurement"] is not None:
        seq.measure(obj["measurement"])

    return seq
