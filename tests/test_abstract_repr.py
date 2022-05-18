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
import json
from typing import Dict, List

import jsonschema
import numpy as np
import pytest

from pulser import Pulse, Register, Register3D, Sequence
from pulser.devices import Chadoq2, MockDevice
from pulser.json.abstract_repr.deserializer import VARIABLE_TYPE_MAP
from pulser.json.abstract_repr.serializer import abstract_repr
from pulser.json.exceptions import AbstractReprError
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

# from tutorials/advanced_features/Serialization.ipynb
qubits = {"control": (-2, 0), "target": (2, 0)}
reg = Register(qubits)

seq = Sequence(reg, Chadoq2)
seq.declare_channel("digital", "raman_local", initial_target="control")
seq.declare_channel("rydberg", "rydberg_local", initial_target="control")

target_atom = seq.declare_variable("target_atom", dtype=int)
duration = seq.declare_variable("duration", dtype=int)
amps = seq.declare_variable("amps", dtype=float, size=2)

half_pi_wf = BlackmanWaveform(200, np.pi / 2)

ry = Pulse.ConstantDetuning(amplitude=half_pi_wf, detuning=0, phase=-np.pi / 2)

seq.add(ry, "digital")
seq.target_index(target_atom, "digital")

pi_2_wf = BlackmanWaveform(duration, amps[0] / 2)
pi_pulse = Pulse.ConstantDetuning(CompositeWaveform(pi_2_wf, pi_2_wf), 0, 0)

max_val = Chadoq2.rabi_from_blockade(8)
two_pi_wf = BlackmanWaveform.from_max_val(max_val, amps[1])
two_pi_pulse = Pulse.ConstantDetuning(two_pi_wf, 0, 0)

seq.align("digital", "rydberg")
seq.add(pi_pulse, "rydberg")
seq.target("target", "rydberg")
seq.add(two_pi_pulse, "rydberg")
seq.target("control", "rydberg")
seq.add(pi_pulse, "rydberg")

seq.align("digital", "rydberg")
seq.delay(100, "digital")
seq.measure("digital")

abstract = json.loads(
    seq.to_abstract_repr(
        target_atom=1,
        amps=[np.pi, 2 * np.pi],
        duration=200,
    )
)


def test_schema():
    with open("pulser-core/pulser/json/abstract_repr/schema.json") as f:
        schema = json.load(f)
    jsonschema.validate(instance=abstract, schema=schema)


def test_values():
    assert set(abstract.keys()) == set(
        [
            "name",
            "version",
            "device",
            "register",
            "variables",
            "channels",
            "operations",
            "measurement",
        ]
    )
    assert abstract["device"] == "Chadoq2"
    assert abstract["register"] == [
        {"name": "control", "x": -2.0, "y": 0.0},
        {"name": "target", "x": 2.0, "y": 0.0},
    ]
    assert abstract["channels"] == {
        "digital": "raman_local",
        "rydberg": "rydberg_local",
    }
    assert abstract["variables"] == {
        "target_atom": {"type": "int", "value": [1]},
        "amps": {"type": "float", "value": [np.pi, 2 * np.pi]},
        "duration": {"type": "int", "value": [200]},
    }
    assert len(abstract["operations"]) == 12
    assert abstract["operations"][0] == {
        "op": "target",
        "channel": "digital",
        "target": 0,
    }

    assert abstract["operations"][2] == {
        "op": "pulse",
        "channel": "digital",
        "protocol": "min-delay",
        "amplitude": {
            "area": 1.5707963267948966,
            "duration": 200,
            "kind": "blackman",
        },
        "detuning": {
            "kind": "constant",
            "duration": 200,
            "value": 0.0,
        },
        "phase": 4.71238898038469,
        "post_phase_shift": 0.0,
    }

    assert abstract["operations"][3] == {
        "op": "target",
        "channel": "digital",
        "target": {
            "expression": "index",
            "lhs": {"variable": "target_atom"},
            "rhs": 0,
        },
    }

    assert abstract["operations"][4] == {
        "op": "align",
        "channels": ["digital", "rydberg"],
    }

    duration_ref = {
        "expression": "index",
        "lhs": {"variable": "duration"},
        "rhs": 0,
    }
    amp0_ref = {
        "expression": "index",
        "lhs": {"variable": "amps"},
        "rhs": 0,
    }
    blackman_wf_dict = {
        "kind": "blackman",
        "duration": duration_ref,
        "area": {"expression": "div", "lhs": amp0_ref, "rhs": 2},
    }
    composite_wf_dict = {
        "kind": "composite",
        "waveforms": [blackman_wf_dict, blackman_wf_dict],
    }

    assert abstract["operations"][5] == {
        "op": "pulse",
        "channel": "rydberg",
        "protocol": "min-delay",
        "amplitude": composite_wf_dict,
        "detuning": {"kind": "constant", "duration": 0, "value": 0.0},
        "phase": 0.0,
        "post_phase_shift": 0.0,
    }

    assert abstract["operations"][11] == {
        "op": "delay",
        "channel": "digital",
        "time": 100,
    }

    assert abstract["measurement"] == "digital"


def test_exceptions():
    with pytest.raises(TypeError, match="not JSON serializable"):
        Sequence(Register3D.cubic(2), MockDevice).to_abstract_repr()

    with pytest.raises(
        ValueError, match="No signature found for 'FakeWaveform'"
    ):
        abstract_repr("FakeWaveform", 100, 1)

    with pytest.raises(ValueError, match="Not enough positional arguments"):
        abstract_repr("ConstantWaveform", 1000)

    with pytest.raises(ValueError, match="Too many positional arguments"):
        abstract_repr("ConstantWaveform", 1000, 1, 4)

    with pytest.raises(ValueError, match="'foo' is not in the signature"):
        abstract_repr("ConstantWaveform", 1000, 1, foo=0)

    with pytest.raises(AbstractReprError, match="Name collisions encountered"):
        Register({"0": (0, 0), 0: (20, 20)})._to_abstract_repr()


def _get_serialized_seq(
    operations: List[Dict] = None,
    variables: Dict[str, Dict] = None,
):
    return {
        "version": "1",
        "name": "John Doe",
        "device": "Chadoq2",
        "register": [
            {"name": "q0", "x": 0, "y": 2},
            {"name": "q42", "x": -2, "y": 9},
            {"name": "q666", "x": 12, "y": 0},
        ],
        "channels": {"digital": "raman_local", "global": "rydberg_global"},
        "operations": operations or [],
        "variables": variables or {},
        "measurement": None,
    }


class TestDeserializarion:
    def test_deserialize_device_and_channels(self):
        s = _get_serialized_seq()

        seq = Sequence.from_abstract_repr(json.dumps(s))

        # Check device name
        assert seq._device.name == s["device"]

        # Check channels
        assert len(seq.declared_channels) == len(s["channels"])
        for name, chan_id in s["channels"].items():
            seq.declared_channels[name] == chan_id

    def test_deserialize_register(self):
        s = _get_serialized_seq()

        seq = Sequence.from_abstract_repr(json.dumps(s))

        # Check register
        assert len(seq.register.qubits) == len(s["register"])
        for q in s["register"]:
            assert q["name"] in seq.qubit_info
            assert seq.qubit_info[q["name"]][0] == q["x"]
            assert seq.qubit_info[q["name"]][1] == q["y"]

    def test_deserialize_variables(self):
        s = _get_serialized_seq(
            variables={
                "yolo": {"type": "int", "value": [42, 43, 44]},
                "zou": {"type": "float", "value": [3.14]},
            }
        )

        seq = Sequence.from_abstract_repr(json.dumps(s))

        # Check variables
        assert len(seq.declared_variables) == len(s["variables"])
        for k, v in s["variables"].items():
            assert k in seq.declared_variables
            assert seq.declared_variables[k].name == k
            assert (
                seq.declared_variables[k].dtype == VARIABLE_TYPE_MAP[v["type"]]
            )
            assert seq.declared_variables[k].size == len(v["value"])

    @pytest.mark.parametrize(
        "op",
        [
            {"op": "target", "target": 2, "channel": "digital"},
            {"op": "delay", "time": 500, "channel": "global"},
            {"op": "align", "channels": ["digital", "global"]},
            {
                "op": "phase_shift",
                "phi": 42,
                "targets": [0, 2],
                "basis": "digital",
            },
            {
                "op": "pulse",
                "channel": "global",
                "phase": 1,
                "post_phase_shift": 2,
                "protocol": "min-delay",
                "amplitude": {
                    "kind": "constant",
                    "duration": 1000,
                    "value": 3.14,
                },
                "detuning": {
                    "kind": "ramp",
                    "duration": 1000,
                    "start": 1,
                    "stop": 5,
                },
            },
        ],
        ids=lambda op: op["op"],
    )
    def test_deserialize_non_parametrized_op(self, op):
        s = _get_serialized_seq(operations=[op])

        seq = Sequence.from_abstract_repr(json.dumps(s))

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset + 1
        # No parametrized call
        assert len(seq._to_build_calls) == 0

        c = seq._calls[offset]
        if op["op"] == "target":
            assert c.name == "_target_index"
            assert c.args == (op["target"], op["channel"])
        elif op["op"] == "align":
            assert c.name == "align"
            assert c.args == tuple(op["channels"])
        elif op["op"] == "delay":
            assert c.name == "delay"
            assert c.kwargs["duration"] == op["time"]
            assert c.kwargs["channel"] == op["channel"]
        elif op["op"] == "phase_shift":
            assert c.name == "_phase_shift_index"
            assert c.args == tuple([op["phi"], *op["targets"]])
        elif op["op"] == "pulse":
            assert c.name == "add"
            assert c.kwargs["channel"] == op["channel"]
            assert c.kwargs["protocol"] == op["protocol"]
            pulse = c.kwargs["pulse"]
            assert isinstance(pulse, Pulse)
            assert pulse.phase == op["phase"]
            assert pulse.post_phase_shift == op["post_phase_shift"]
            assert isinstance(pulse.amplitude, Waveform)
            assert isinstance(pulse.detuning, Waveform)
        else:
            assert False, f"operation type \"{op['op']}\" is not valid"

    @pytest.mark.parametrize(
        "wf_obj",
        [
            {"kind": "constant", "duration": 1200, "value": 3.14},
            {"kind": "ramp", "duration": 1200, "start": 1.14, "stop": 3},
            {"kind": "blackman", "duration": 1200, "area": 2 * 3.14},
            {"kind": "blackman_max", "max_val": 5, "area": 2 * 3.14},
            {
                "kind": "interpolated",
                "duration": 2000,
                "values": [1, 1.5, 1.7, 1.3],
                "times": [0, 0.4, 0.8, 0.9],
            },
            {"kind": "kaiser", "duration": 2000, "area": 12, "beta": 1.1},
            {"kind": "kaiser_max", "max_val": 6, "area": 12, "beta": 1.1},
            {
                "kind": "composite",
                "waveforms": [
                    {"kind": "constant", "duration": 104, "value": 1},
                    {"kind": "constant", "duration": 208, "value": 2},
                    {"kind": "constant", "duration": 312, "value": 3},
                ],
            },
            {"kind": "custom", "samples": [i / 10 for i in range(0, 20)]},
        ],
        ids=lambda op: op["kind"],
    )
    def test_deserialize_non_parametrized_waveform(self, wf_obj):
        s = _get_serialized_seq(
            operations=[
                {
                    "op": "pulse",
                    "op": "pulse",
                    "channel": "global",
                    "phase": 1,
                    "post_phase_shift": 2,
                    "protocol": "min-delay",
                    "amplitude": wf_obj,
                    "detuning": wf_obj,
                }
            ]
        )

        seq = Sequence.from_abstract_repr(json.dumps(s))

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset + 1
        # No parametrized call
        assert len(seq._to_build_calls) == 0

        c = seq._calls[offset]
        pulse: Pulse = c.kwargs["pulse"]
        wf = pulse.amplitude

        if wf_obj["kind"] == "constant":
            assert isinstance(wf, ConstantWaveform)
            assert wf.duration == wf_obj["duration"]
            assert wf._value == wf_obj["value"]

        elif wf_obj["kind"] == "ramp":
            assert isinstance(wf, RampWaveform)
            assert wf.duration == wf_obj["duration"]
            assert wf._start == wf_obj["start"]
            assert wf._stop == wf_obj["stop"]

        elif wf_obj["kind"] == "blackman":
            assert isinstance(wf, BlackmanWaveform)
            assert wf.duration == wf_obj["duration"]
            assert wf._area == wf_obj["area"]

        elif wf_obj["kind"] == "blackman_max":
            assert isinstance(wf, BlackmanWaveform)
            assert wf._area == wf_obj["area"]
            expected_duration = BlackmanWaveform.from_max_val(
                wf_obj["max_val"], wf_obj["area"]
            ).duration
            assert wf.duration == expected_duration

        elif wf_obj["kind"] == "interpolated":
            assert isinstance(wf, InterpolatedWaveform)
            assert np.array_equal(wf._values, wf_obj["values"])
            assert np.array_equal(wf._times, wf_obj["times"])

        elif wf_obj["kind"] == "kaiser":
            assert isinstance(wf, KaiserWaveform)
            assert wf.duration == wf_obj["duration"]
            assert wf._area == wf_obj["area"]
            assert wf._beta == wf_obj["beta"]

        elif wf_obj["kind"] == "kaiser_max":
            assert isinstance(wf, KaiserWaveform)
            assert wf._area == wf_obj["area"]
            assert wf._beta == wf_obj["beta"]
            expected_duration = KaiserWaveform.from_max_val(
                wf_obj["max_val"], wf_obj["area"], wf_obj["beta"]
            ).duration
            assert wf.duration == expected_duration

        elif wf_obj["kind"] == "composite":
            assert isinstance(wf, CompositeWaveform)
            assert all(isinstance(w, Waveform) for w in wf._waveforms)

        elif wf_obj["kind"] == "custom":
            assert isinstance(wf, CustomWaveform)
            assert np.array_equal(wf._samples, wf_obj["samples"])

    def test_deserialize_measurement(self):
        s = _get_serialized_seq()
        s["measurement"] = "ground-rydberg"

        seq = Sequence.from_abstract_repr(json.dumps(s))

        assert seq._measurement == s["measurement"]
