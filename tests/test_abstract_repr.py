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

import jsonschema
import numpy as np
import pytest

from pulser import Pulse, Register, Register3D, Sequence
from pulser.devices import Chadoq2, MockDevice
from pulser.json.exceptions import AbstractReprError
from pulser.json.utils import abstract_repr
from pulser.waveforms import BlackmanWaveform, CompositeWaveform

# from tutorials/advanced_features/Serialization.ipynb
qubits = {"control": (-2, 0), "target": (2, 0)}
reg = Register(qubits)

seq = Sequence(reg, Chadoq2)
seq.declare_channel("digital", "raman_local", initial_target="control")
seq.declare_channel("rydberg", "rydberg_local", initial_target="control")

target_atom = seq.declare_variable("target_atom", dtype=str)
duration = seq.declare_variable("duration", dtype=int)
amps = seq.declare_variable("amps", dtype=float, size=2)

half_pi_wf = BlackmanWaveform(200, np.pi / 2)

ry = Pulse.ConstantDetuning(amplitude=half_pi_wf, detuning=0, phase=-np.pi / 2)

seq.add(ry, "digital")
seq.target(target_atom, "digital")

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
    seq.abstract_repr(
        target_atom="target",
        amps=[np.pi, 2 * np.pi],
        duration=200,
    )
)


def test_schema():
    with open("pulser-core/pulser/json/abstract_repr_schema.json") as f:
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
    assert abstract["register"] == {
        "control": {"x": -2.0, "y": 0.0},
        "target": {"x": 2.0, "y": 0.0},
    }
    assert abstract["channels"] == {
        "digital": "raman_local",
        "rydberg": "rydberg_local",
    }
    assert abstract["variables"] == {
        "target_atom": {"type": "atom_name", "value": ["target"]},
        "amps": {"type": "float", "value": [np.pi, 2 * np.pi]},
        "duration": {"type": "int", "value": [200]},
    }
    assert len(abstract["operations"]) == 12
    assert abstract["operations"][0] == {
        "op": "target",
        "channel": "digital",
        "target": "control",
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
        Sequence(Register3D.cubic(2), MockDevice).abstract_repr()

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
