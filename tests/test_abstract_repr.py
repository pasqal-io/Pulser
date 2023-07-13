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
from __future__ import annotations

import json
import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Type
from unittest.mock import patch

import jsonschema
import numpy as np
import pytest

from pulser import Pulse, Register, Register3D, Sequence, devices
from pulser.devices import Chadoq2, Device, IroiseMVP, MockDevice
from pulser.json.abstract_repr.deserializer import (
    VARIABLE_TYPE_MAP,
    deserialize_device,
)
from pulser.json.abstract_repr.serializer import (
    AbstractReprEncoder,
    abstract_repr,
)
from pulser.json.abstract_repr.validation import RESOLVER
from pulser.json.exceptions import AbstractReprError, DeserializeDeviceError
from pulser.parametrized.decorators import parametrize
from pulser.parametrized.paramobj import ParamObj
from pulser.parametrized.variable import VariableItem
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import TriangularLatticeLayout
from pulser.sequence._call import _Call
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

SPECIAL_WFS: dict[str, tuple[Callable, tuple[str, ...]]] = {
    "kaiser_max": (KaiserWaveform.from_max_val, ("max_val", "area", "beta")),
    "blackman_max": (BlackmanWaveform.from_max_val, ("max_val", "area")),
}


class TestDevice:
    @pytest.fixture(params=[Chadoq2, IroiseMVP, MockDevice])
    def abstract_device(self, request):
        device = request.param
        return json.loads(device.to_abstract_repr())

    @pytest.fixture
    def device_schema(self):
        with open(
            "pulser-core/pulser/json/abstract_repr/schemas/device-schema.json"
        ) as f:
            dev_schema = json.load(f)
        return dev_schema

    def test_device_schema(self, abstract_device, device_schema):
        jsonschema.validate(instance=abstract_device, schema=device_schema)

    def test_roundtrip(self, abstract_device):
        device = deserialize_device(json.dumps(abstract_device))
        assert json.loads(device.to_abstract_repr()) == abstract_device

    def test_exceptions(self, abstract_device, device_schema):
        def check_error_raised(
            obj_str: str, original_err: Type[Exception], err_msg: str = ""
        ) -> Exception:
            with pytest.raises(DeserializeDeviceError) as exc_info:
                deserialize_device(obj_str)

            cause = exc_info.value.__cause__
            assert isinstance(cause, original_err)
            assert re.search(re.escape(err_msg), str(cause)) is not None
            return cause

        good_device = deserialize_device(json.dumps(abstract_device))

        check_error_raised(
            abstract_device, TypeError, "'obj_str' must be a string"
        )

        # JSONDecodeError from json.loads()
        bad_str = "\ufeff"
        with pytest.raises(
            json.JSONDecodeError, match="Unexpected UTF-8 BOM"
        ) as err:
            json.loads(bad_str)
        err_msg = str(err.value)
        check_error_raised(bad_str, json.JSONDecodeError, err_msg)

        # jsonschema.exceptions.ValidationError from jsonschema
        invalid_dev = abstract_device.copy()
        invalid_dev["rydberg_level"] = "70"
        with pytest.raises(jsonschema.exceptions.ValidationError) as err:
            jsonschema.validate(instance=invalid_dev, schema=device_schema)
        check_error_raised(
            json.dumps(invalid_dev),
            jsonschema.exceptions.ValidationError,
            str(err.value),
        )

        # AbstractReprError from invalid RydbergEOM configuration
        if good_device.channels["rydberg_global"].eom_config:
            bad_eom_dev = deepcopy(abstract_device)
            for ch_dict in bad_eom_dev["channels"]:
                if ch_dict["eom_config"]:
                    assert "max_limiting_amp" in ch_dict["eom_config"]
                    ch_dict["eom_config"]["max_limiting_amp"] = 0.0
                    break
            prev_err = check_error_raised(
                json.dumps(bad_eom_dev),
                AbstractReprError,
                "RydbergEOM deserialization failed.",
            )
            assert isinstance(prev_err.__cause__, ValueError)

        # AbstractReprError from ValueError in channel creation
        bad_ch_dev1 = deepcopy(abstract_device)
        bad_ch_dev1["channels"][0]["min_duration"] = -1
        prev_err = check_error_raised(
            json.dumps(bad_ch_dev1),
            AbstractReprError,
            "Channel deserialization failed.",
        )
        assert isinstance(prev_err.__cause__, ValueError)

        # AbstractReprError from NotImplementedError in channel creation
        bad_ch_dev2 = deepcopy(abstract_device)
        bad_ch_dev2["channels"][0]["mod_bandwidth"] = 1000
        prev_err = check_error_raised(
            json.dumps(bad_ch_dev2),
            AbstractReprError,
            "Channel deserialization failed.",
        )
        assert isinstance(prev_err.__cause__, NotImplementedError)

        # AbstractReprError from bad layout (only in physical devices)
        if isinstance(good_device, Device):
            bad_layout_dev = abstract_device.copy()
            # Identical coords fail
            bad_layout_obj = {"coordinates": [[0, 0], [0.0, 0.0]]}
            bad_layout_dev["pre_calibrated_layouts"] = [bad_layout_obj]
            prev_err = check_error_raised(
                json.dumps(bad_layout_dev),
                AbstractReprError,
                "Register layout deserialization failed.",
            )
            assert isinstance(prev_err.__cause__, ValueError)

        # AbstractReprError from TypeError in device init
        if "XY" in good_device.supported_bases:
            bad_xy_coeff_dev = abstract_device.copy()
            bad_xy_coeff_dev["interaction_coeff_xy"] = None
            prev_err = check_error_raised(
                json.dumps(bad_xy_coeff_dev),
                AbstractReprError,
                "Device deserialization failed.",
            )
            assert isinstance(prev_err.__cause__, TypeError)

        # AbstractReprError from ValueError in device init
        bad_dev = abstract_device.copy()
        bad_dev["min_atom_distance"] = -1
        prev_err = check_error_raised(
            json.dumps(bad_dev),
            AbstractReprError,
            "Device deserialization failed.",
        )
        assert isinstance(prev_err.__cause__, ValueError)


def validate_schema(instance):
    with open(
        "pulser-core/pulser/json/abstract_repr/schemas/" "sequence-schema.json"
    ) as f:
        schema = json.load(f)
    jsonschema.validate(instance=instance, schema=schema, resolver=RESOLVER)


class TestSerialization:
    @pytest.fixture
    def triangular_lattice(self):
        return TriangularLatticeLayout(50, 6)

    @pytest.fixture(params=[Chadoq2, MockDevice])
    def sequence(self, request):
        qubits = {"control": (-2, 0), "target": (2, 0)}
        reg = Register(qubits)
        device = request.param
        seq = Sequence(reg, device)
        seq.declare_channel("digital", "raman_local", initial_target="control")
        seq.declare_channel(
            "rydberg", "rydberg_local", initial_target="control"
        )

        target_atom = seq.declare_variable("target_atom", dtype=int)
        duration = seq.declare_variable("duration", dtype=int)
        amps = seq.declare_variable("amps", dtype=float, size=2)

        half_pi_wf = BlackmanWaveform(200, np.pi / 2)

        ry = Pulse.ConstantDetuning(
            amplitude=half_pi_wf, detuning=0, phase=-np.pi / 2
        )

        seq.add(ry, "digital")
        seq.target_index(target_atom, "digital")
        seq.phase_shift_index(-1.0, target_atom)

        pi_2_wf = BlackmanWaveform(duration, amps[0] / 2)
        pi_pulse = Pulse.ConstantDetuning(
            CompositeWaveform(pi_2_wf, pi_2_wf), 0, 0
        )

        max_val = Chadoq2.rabi_from_blockade(8)
        two_pi_wf = BlackmanWaveform.from_max_val(max_val, amps[1])
        two_pi_pulse = Pulse.ConstantDetuning(two_pi_wf, 0, 0)

        seq.align("digital", "rydberg")
        seq.add(pi_pulse, "rydberg")
        seq.phase_shift(1.0, "control", "target", basis="ground-rydberg")
        seq.target("target", "rydberg")
        seq.add(two_pi_pulse, "rydberg")

        seq.delay(100, "digital")
        seq.measure("digital")
        return seq

    @pytest.fixture
    def abstract(self, sequence):
        return json.loads(
            sequence.to_abstract_repr(
                target_atom=1,
                amps=[np.pi, 2 * np.pi],
                duration=200,
            )
        )

    def test_schema(self, abstract):
        validate_schema(abstract)

    def test_values(self, abstract):
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
        device_name = abstract["device"]["name"]
        assert abstract["device"]["name"] in [
            d.name for d in [*devices._valid_devices, *devices._mock_devices]
        ]
        assert abstract["device"] == json.loads(
            getattr(devices, device_name).to_abstract_repr()
        )
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
        assert len(abstract["operations"]) == 11
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

        assert abstract["operations"][5] == {
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

        assert abstract["operations"][6] == {
            "op": "pulse",
            "channel": "rydberg",
            "protocol": "min-delay",
            "amplitude": composite_wf_dict,
            "detuning": {"kind": "constant", "duration": 0, "value": 0.0},
            "phase": 0.0,
            "post_phase_shift": 0.0,
        }

        assert abstract["operations"][10] == {
            "op": "delay",
            "channel": "digital",
            "time": 100,
        }

        assert abstract["measurement"] == "digital"

    def test_exceptions(self, sequence):
        with pytest.raises(TypeError, match="not JSON serializable"):
            Sequence(
                Register3D.cubic(2, prefix="q"), MockDevice
            ).to_abstract_repr()

        with pytest.raises(
            ValueError, match="No signature found for 'FakeWaveform'"
        ):
            abstract_repr("FakeWaveform", 100, 1)

        with pytest.raises(ValueError, match="Not enough arguments"):
            abstract_repr("ConstantWaveform", 1000)

        with pytest.raises(ValueError, match="Too many positional arguments"):
            abstract_repr("ConstantWaveform", 1000, 1, 4)

        with pytest.raises(ValueError, match="'foo' is not in the signature"):
            abstract_repr("ConstantWaveform", 1000, 1, foo=0)

        with pytest.warns(
            UserWarning, match="converts all qubit ID's to strings"
        ), pytest.raises(
            AbstractReprError, match="Name collisions encountered"
        ):
            Register({"0": (0, 0), 0: (20, 20)})._to_abstract_repr()

        with pytest.raises(
            AbstractReprError,
            match="Export of an InterpolatedWaveform is only supported for the"
            " 'PchipInterpolator'",
        ):
            InterpolatedWaveform(
                1000, [0, 1, 0], interpolator="interp1d"
            )._to_abstract_repr()

        with pytest.raises(
            AbstractReprError, match="without any 'interpolator_kwargs'"
        ):
            InterpolatedWaveform(
                1000, [0, 1, 0], extrapolate=False
            )._to_abstract_repr()

        with pytest.raises(
            ValueError,
            match="The given 'defaults' produce an invalid sequence.",
        ):
            sequence.to_abstract_repr(
                target_atom=1,
                amps=[-np.pi, 2 * np.pi],
                duration=200,
            )

    @pytest.mark.parametrize(
        "call",
        [
            _Call("targets", ({"q0", "q1"}, "ch0"), {}),
            _Call(
                "phase_shifts", (1.0, "q2", "q3"), dict(basis="ground-rydberg")
            ),
            _Call("wait", (100,), {}),
        ],
    )
    def test_unknown_calls(self, call):
        seq = Sequence(Register.square(2, prefix="q"), Chadoq2)
        seq.declare_channel("ch0", "rydberg_global")
        seq._calls.append(call)
        with pytest.raises(
            AbstractReprError, match=f"Unknown call '{call.name}'."
        ):
            seq.to_abstract_repr()

    @pytest.mark.parametrize(
        "obj,serialized_obj",
        [
            (Register({"q0": (0.0, 0.0)}), [dict(name="q0", x=0.0, y=0.0)]),
            (np.arange(3), [0, 1, 2]),
            ({"a"}, ["a"]),
        ],
        ids=["register", "np.array", "set"],
    )
    def test_abstract_repr_encoder(self, obj, serialized_obj):
        assert json.dumps(obj, cls=AbstractReprEncoder) == json.dumps(
            serialized_obj
        )

    @pytest.mark.filterwarnings(
        "ignore:Serialization of 'getattr':UserWarning"
    )
    def test_paramobj_serialization(self, sequence):
        var = sequence._variables["duration"][0]
        ser_var = {
            "expression": "index",
            "lhs": {"variable": "duration"},
            "rhs": 0,
        }
        wf = BlackmanWaveform(1000, 1.0)
        ser_wf = wf._to_abstract_repr()
        warn_msg = (
            "Calls to methods of parametrized objects are only "
            "executed if they serve as arguments of other parametrized"
            " objects that are themselves built"
        )
        with pytest.warns(UserWarning, match=warn_msg):
            param_obj_call = BlackmanWaveform(var, 1)()

        err_msg = "Serialization of calls to parametrized objects"
        with pytest.raises(ValueError, match=err_msg):
            json.dumps(param_obj_call, cls=AbstractReprEncoder)

        s = json.dumps(
            Pulse.ConstantAmplitude(var, wf, 1.0, 1.0), cls=AbstractReprEncoder
        )
        assert json.loads(s) == dict(
            amplitude={"kind": "constant", "duration": 0, "value": ser_var},
            detuning=ser_wf,
            phase=1.0,
            post_phase_shift=1.0,
        )

        s = json.dumps(
            Pulse.ConstantDetuning(wf, 0.0, var),
            cls=AbstractReprEncoder,
        )
        assert json.loads(s) == dict(
            amplitude=ser_wf,
            detuning={"kind": "constant", "duration": 0, "value": 0.0},
            phase=ser_var,
            post_phase_shift=0.0,  # The default is added
        )

        s = json.dumps(
            Pulse.ConstantPulse(var, 2.0, 0.0, 1.0, post_phase_shift=1.0),
            cls=AbstractReprEncoder,
        )
        assert json.loads(s) == dict(
            amplitude={"kind": "constant", "duration": ser_var, "value": 2.0},
            detuning={"kind": "constant", "duration": ser_var, "value": 0.0},
            phase=1.0,
            post_phase_shift=1.0,
        )

        method_call = parametrize(BlackmanWaveform.change_duration)(wf, var)
        with pytest.raises(
            NotImplementedError,
            match="Instance or static method serialization is not supported.",
        ):
            method_call._to_abstract_repr()

        # Check the defaults are added when not specified
        s = json.dumps(
            KaiserWaveform.from_max_val(1.0, var), cls=AbstractReprEncoder
        )
        assert json.loads(s) == dict(
            kind="kaiser_max",
            max_val=1.0,
            area=ser_var,
            beta=14.0,  # The default beta parameter
        )

        s = json.dumps(KaiserWaveform(var, var, var), cls=AbstractReprEncoder)
        assert json.loads(s) == dict(
            kind="kaiser",
            duration=ser_var,
            area=ser_var,
            beta=ser_var,  # The given beta parameter
        )

        s = json.dumps(
            InterpolatedWaveform(var, [1, 2, -3]), cls=AbstractReprEncoder
        )
        assert json.loads(s) == dict(
            kind="interpolated",
            duration=ser_var,
            values=[1, 2, -3],
            times=[0.0, 0.5, 1.0],
        )

        list_var = sequence.declare_variable("list_var", size=3)
        ser_list_var = {"variable": "list_var"}
        s = json.dumps(
            InterpolatedWaveform(var, list_var), cls=AbstractReprEncoder
        )
        assert json.loads(s) == dict(
            kind="interpolated",
            duration=ser_var,
            values=ser_list_var,
            times=[0.0, 0.5, 1.0],
        )

        err_msg = (
            "An InterpolatedWaveform with 'values' of unknown length "
            "and unspecified 'times' can't be serialized to the abstract"
            " representation."
        )
        with pytest.raises(AbstractReprError, match=err_msg):
            json.dumps(
                InterpolatedWaveform(1000, np.cos(list_var)),
                cls=AbstractReprEncoder,
            )

        with pytest.raises(
            AbstractReprError, match="No abstract representation for 'Foo'"
        ):

            class Foo:
                def __init__(self, bar: str):
                    pass

            ParamObj(Foo, "bar")._to_abstract_repr()

    def test_mw_sequence(self, triangular_lattice):
        mag_field = [-10, 40, 0]
        mask = {"q0", "q2", "q4"}
        reg = triangular_lattice.hexagonal_register(5)
        seq = Sequence(reg, MockDevice)
        seq.declare_channel("mw_ch", "mw_global")
        seq.set_magnetic_field(*mag_field)
        seq.config_slm_mask(mask)
        seq.add(Pulse.ConstantPulse(100, 1, 0, 2), "mw_ch")
        seq.measure("XY")

        abstract = json.loads(seq.to_abstract_repr())
        validate_schema(abstract)
        assert abstract["register"] == [
            {"name": str(qid), "x": c[0], "y": c[1]}
            for qid, c in reg.qubits.items()
        ]
        assert abstract["layout"] == {
            "coordinates": triangular_lattice.coords.tolist(),
            "slug": triangular_lattice.slug,
        }
        assert abstract["magnetic_field"] == mag_field
        assert abstract["slm_mask_targets"] == list(mask)
        assert abstract["measurement"] == "XY"

    def test_mappable_register(self, triangular_lattice):
        reg = triangular_lattice.make_mappable_register(2)
        seq = Sequence(reg, MockDevice)
        _ = seq.declare_variable("var", dtype=int)

        abstract = json.loads(seq.to_abstract_repr())
        validate_schema(abstract)
        assert abstract["layout"] == {
            "coordinates": triangular_lattice.coords.tolist(),
            "slug": triangular_lattice.slug,
        }
        assert abstract["register"] == [{"qid": qid} for qid in reg.qubit_ids]
        assert abstract["variables"]["var"] == dict(type="int", value=[0])

        with pytest.raises(
            ValueError,
            match="The given 'defaults' produce an invalid sequence.",
        ):
            seq.to_abstract_repr(var=0)

        with pytest.raises(
            ValueError,
            match="The given 'defaults' produce an invalid sequence.",
        ):
            seq.to_abstract_repr(var=0, qubits={"q1": 0})

        with pytest.raises(TypeError, match="Did not receive values"):
            seq.to_abstract_repr(qubits={"q0": 0})

        assert not seq.is_parametrized()

        abstract = json.loads(seq.to_abstract_repr(var=0, qubits={"q0": 0}))
        assert abstract["register"] == [
            {"qid": "q0", "default_trap": 0},
            {"qid": "q1"},
        ]
        assert abstract["variables"]["var"] == dict(type="int", value=[0])

    def test_eom_mode(self, triangular_lattice):
        reg = triangular_lattice.hexagonal_register(7)
        seq = Sequence(reg, IroiseMVP)
        seq.declare_channel("ryd", "rydberg_global")
        det_off = seq.declare_variable("det_off", dtype=float)
        duration = seq.declare_variable("duration", dtype=int)
        seq.enable_eom_mode(
            "ryd", amp_on=3.0, detuning_on=0.0, optimal_detuning_off=det_off
        )
        seq.add_eom_pulse("ryd", duration, 0.0)
        seq.delay(duration, "ryd")
        seq.disable_eom_mode("ryd")

        abstract = json.loads(seq.to_abstract_repr())
        validate_schema(abstract)

        assert abstract["operations"][0] == {
            "op": "enable_eom_mode",
            "channel": "ryd",
            "amp_on": 3.0,
            "detuning_on": 0.0,
            "optimal_detuning_off": {
                "expression": "index",
                "lhs": {"variable": "det_off"},
                "rhs": 0,
            },
        }

        ser_duration = {
            "expression": "index",
            "lhs": {"variable": "duration"},
            "rhs": 0,
        }
        assert abstract["operations"][1] == {
            "op": "add_eom_pulse",
            "channel": "ryd",
            "duration": ser_duration,
            "phase": 0.0,
            "post_phase_shift": 0.0,
            "protocol": "min-delay",
        }

        assert abstract["operations"][3] == {
            "op": "disable_eom_mode",
            "channel": "ryd",
        }

    @pytest.mark.parametrize("use_default", [True, False])
    def test_default_basis(
        self, triangular_lattice: TriangularLatticeLayout, use_default
    ):
        phase_kwargs = {} if use_default else dict(basis="ground-rydberg")
        measure_kwargs = {} if use_default else dict(basis="digital")

        seq = Sequence(triangular_lattice.hexagonal_register(5), Chadoq2)
        seq.declare_channel("ryd", "rydberg_global")
        seq.declare_channel("raman", "raman_local", initial_target="q0")
        seq.phase_shift(1, "q0", **phase_kwargs)
        seq.phase_shift_index(2, 1, **phase_kwargs)
        seq.measure(**measure_kwargs)

        abstract = json.loads(seq.to_abstract_repr())
        validate_schema(abstract)
        assert len(abstract["operations"]) == 3

        assert abstract["operations"][0] == {
            "op": "target",
            "channel": "raman",
            "target": 0,
        }

        assert abstract["operations"][1] == {
            "op": "phase_shift",
            "basis": phase_kwargs.get("basis", "digital"),
            "targets": [0],
            "phi": 1,
        }
        assert abstract["operations"][2] == {
            "op": "phase_shift",
            "basis": phase_kwargs.get("basis", "digital"),
            "targets": [1],
            "phi": 2,
        }
        assert abstract["measurement"] == measure_kwargs.get(
            "basis", "ground-rydberg"
        )

    @pytest.mark.parametrize(
        "op,args",
        [
            ("target", ("q0", "raman_local")),
            ("phase_shift", (1, "q0", "q1")),
        ],
    )
    def test_mappable_reg_with_local_ops(
        self, op, args, triangular_lattice: TriangularLatticeLayout
    ):
        mappable_reg = triangular_lattice.make_mappable_register(5)
        seq = Sequence(mappable_reg, MockDevice)

        seq.declare_channel("raman_local", "raman_local")
        getattr(seq, op)(*args)
        seq.to_abstract_repr()


def _get_serialized_seq(
    operations: list[dict] = [],
    variables: dict[str, dict] = {},
    **override_kwargs: Any,
) -> dict[str, Any]:
    seq_dict = {
        "version": "1",
        "name": "John Doe",
        "device": json.loads(Chadoq2.to_abstract_repr()),
        "register": [
            {"name": "q0", "x": 0.0, "y": 2.0},
            {"name": "q42", "x": -2.0, "y": 9.0},
            {"name": "q666", "x": 12.0, "y": 0.0},
        ],
        "channels": {"digital": "raman_local", "global": "rydberg_global"},
        "operations": operations,
        "variables": variables,
        "measurement": None,
    }
    seq_dict.update(override_kwargs)
    return seq_dict


def _check_roundtrip(serialized_seq: dict[str, Any]):
    s = serialized_seq.copy()
    # Replaces the special wfs when they are not parametrized
    for op in s["operations"]:
        if op["op"] == "pulse":
            for wf in ("amplitude", "detuning"):
                if op[wf]["kind"] in SPECIAL_WFS:
                    wf_cls, wf_args = SPECIAL_WFS[op[wf]["kind"]]
                    for val in op[wf].values():
                        if isinstance(val, dict):
                            # Parametrized
                            break
                    else:
                        reconstructed_wf = wf_cls(
                            *(op[wf][qty] for qty in wf_args)
                        )
                        op[wf] = reconstructed_wf._to_abstract_repr()

    seq = Sequence.from_abstract_repr(json.dumps(s))
    defaults = {
        name: var["value"]
        for name, var in s["variables"].items()
        if "value" in var
    }
    qubits_default = {
        q["qid"]: q["default_trap"]
        for q in s["register"]
        if "default_trap" in q
    }
    rs = seq.to_abstract_repr(
        seq_name=serialized_seq["name"],
        qubits=qubits_default or None,
        json_dumps_options={"indent": None},
        **defaults,
    )
    assert s == json.loads(rs)

    # Remove the defaults and check it still works
    for var in seq.declared_variables.values():
        s["variables"][var.name]["value"] = [var.dtype()] * var.size
    for q in s["register"]:
        q.pop("default_trap", None)
    s["name"] = "pulser-exported"

    seq2 = Sequence.from_abstract_repr(json.dumps(s))
    rs_no_defaults = seq2.to_abstract_repr()
    assert s == json.loads(rs_no_defaults)


# Needed to replace lambdas in the pytest.mark.parametrize calls (due to mypy)
def _get_op(op: dict) -> Any:
    return op["op"]


def _get_kind(op: dict) -> Any:
    return op["kind"]


def _get_expression(op: dict) -> Any:
    return op["expression"]


class TestDeserialization:
    def test_deserialize_device_and_channels(self) -> None:
        s = _get_serialized_seq()
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        # Check device
        assert seq._device == deserialize_device(json.dumps(s["device"]))

        # Check channels
        assert len(seq.declared_channels) == len(s["channels"])
        for name, chan_id in s["channels"].items():
            seq.declared_channels[name] == chan_id

    _coords = np.array([[0.0, 2.0], [-2.0, 9.0], [12.0, 0.0]])
    _coords = np.concatenate((_coords, -_coords))

    @pytest.mark.parametrize("layout_coords", [None, _coords])
    def test_deserialize_register(self, layout_coords):
        if layout_coords is not None:
            reg_layout = RegisterLayout(layout_coords)
            s = _get_serialized_seq(
                layout={"coordinates": reg_layout.coords.tolist()}
            )
        else:
            s = _get_serialized_seq()
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        # Check register
        assert len(seq.register.qubits) == len(s["register"])
        for q in s["register"]:
            assert q["name"] in seq.qubit_info
            assert seq.qubit_info[q["name"]][0] == q["x"]
            assert seq.qubit_info[q["name"]][1] == q["y"]

        # Check layout
        if layout_coords is not None:
            assert seq.register.layout == reg_layout
            q_coords = list(seq.qubit_info.values())
            assert seq.register._layout_info.trap_ids == tuple(
                reg_layout.get_traps_from_coordinates(*q_coords)
            )
        else:
            assert "layout" not in s
            assert seq.register.layout is None

    def test_deserialize_mappable_register(self):
        layout_coords = (5 * np.arange(8)).reshape((4, 2))
        s = _get_serialized_seq(
            register=[{"qid": "q0", "default_trap": 2}, {"qid": "q1"}],
            layout={
                "coordinates": layout_coords.tolist(),
                "slug": "test_layout",
            },
        )
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        assert seq.is_register_mappable()
        qids = [q["qid"] for q in s["register"]]
        assert seq._register.qubit_ids == tuple(qids)
        assert seq._register.layout == RegisterLayout(layout_coords)

    def test_deserialize_seq_with_slm_mask(self):
        s = _get_serialized_seq(slm_mask_targets=["q0"])
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))
        assert seq._slm_mask_targets == {"q0"}

    def test_deserialize_seq_with_mag_field(self):
        mag_field = [10.0, -43.2, 0.0]
        s = _get_serialized_seq(
            magnetic_field=mag_field,
            device=json.loads(MockDevice.to_abstract_repr()),
            channels={"mw": "mw_global"},
        )
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))
        assert np.all(seq.magnetic_field == mag_field)

    @pytest.mark.parametrize("without_default", [True, False])
    def test_deserialize_variables(self, without_default):
        s = _get_serialized_seq(
            variables={
                "yolo": {"type": "int", "value": [42, 43, 44]},
                "zou": {"type": "float", "value": [3.14]},
            }
        )
        with pytest.warns(
            UserWarning, match="Building a non-parametrized sequence"
        ):
            _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))
        if without_default:
            # Serialize and deserialize again, without the defaults
            seq = Sequence.from_abstract_repr(seq.to_abstract_repr())

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
        ids=_get_op,
    )
    def test_deserialize_non_parametrized_op(self, op):
        s = _get_serialized_seq(operations=[op])
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset + 1
        # No parametrized call
        assert len(seq._to_build_calls) == 0

        c = seq._calls[offset]
        if op["op"] == "target":
            assert c.name == "target_index"
            assert c.kwargs["qubits"] == op["target"]
            assert c.kwargs["channel"] == op["channel"]
        elif op["op"] == "align":
            assert c.name == "align"
            assert c.args == tuple(op["channels"])
        elif op["op"] == "delay":
            assert c.name == "delay"
            assert c.kwargs["duration"] == op["time"]
            assert c.kwargs["channel"] == op["channel"]
        elif op["op"] == "phase_shift":
            assert c.name == "phase_shift_index"
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
        ids=_get_kind,
    )
    def test_deserialize_non_parametrized_waveform(self, wf_obj):
        s = _get_serialized_seq(
            operations=[
                {
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
        _check_roundtrip(s)
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
        _check_roundtrip(s)
        s["measurement"] = "ground-rydberg"

        seq = Sequence.from_abstract_repr(json.dumps(s))

        assert seq._measurement == s["measurement"]

    var1 = {
        "expression": "index",
        "lhs": {"variable": "var1"},
        "rhs": 0,
    }

    var2 = {
        "expression": "index",
        "lhs": {"variable": "var2"},
        "rhs": 0,
    }

    var3 = {
        "expression": "index",
        "lhs": {"variable": "var3"},
        "rhs": 0,
    }

    @pytest.mark.parametrize(
        "op",
        [
            {"op": "target", "target": var1, "channel": "digital"},
            {"op": "delay", "time": var2, "channel": "global"},
            {
                "op": "phase_shift",
                "phi": var1,
                "targets": [2, var1],
                "basis": "digital",
            },
            {
                "op": "pulse",
                "channel": "global",
                "phase": var1,
                "post_phase_shift": var2,
                "protocol": "min-delay",
                "amplitude": {
                    "kind": "constant",
                    "duration": var2,
                    "value": 3.14,
                },
                "detuning": {
                    "kind": "ramp",
                    "duration": var2,
                    "start": 1,
                    "stop": 5,
                },
            },
        ],
        ids=_get_op,
    )
    def test_deserialize_parametrized_op(self, op):
        s = _get_serialized_seq(
            operations=[op],
            variables={
                "var1": {"type": "int", "value": [0]},
                "var2": {"type": "int", "value": [42]},
            },
        )
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset
        # No parametrized call
        assert len(seq._to_build_calls) == 1

        c = seq._to_build_calls[0]
        if op["op"] == "target":
            assert c.name == "target_index"
            assert isinstance(c.kwargs["qubits"], VariableItem)
            assert c.kwargs["channel"] == op["channel"]
        elif op["op"] == "delay":
            assert c.name == "delay"
            assert c.kwargs["channel"] == op["channel"]
            assert isinstance(c.kwargs["duration"], VariableItem)
        elif op["op"] == "phase_shift":
            assert c.name == "phase_shift_index"
            # phi is variable
            assert isinstance(c.args[0], VariableItem)
            # qubit 1 is fixed
            assert c.args[1] == 2
            # qubit 2 is variable
            assert isinstance(c.args[2], VariableItem)
        elif op["op"] == "pulse":
            assert c.name == "add"
            assert c.kwargs["channel"] == op["channel"]
            assert c.kwargs["protocol"] == op["protocol"]
            pulse = c.kwargs["pulse"]
            assert isinstance(pulse, ParamObj)
            assert pulse.cls == Pulse
            assert isinstance(pulse.kwargs["phase"], VariableItem)
            assert isinstance(pulse.kwargs["post_phase_shift"], VariableItem)

            assert isinstance(pulse.kwargs["amplitude"], ParamObj)
            assert issubclass(pulse.kwargs["amplitude"].cls, Waveform)
            assert isinstance(pulse.kwargs["detuning"], ParamObj)
            assert issubclass(pulse.kwargs["detuning"].cls, Waveform)
        else:
            assert False, f"operation type \"{op['op']}\" is not valid"

    @pytest.mark.parametrize(
        "op, pulse_cls",
        [
            (
                {
                    "op": "pulse",
                    "channel": "global",
                    "phase": var1,
                    "post_phase_shift": var2,
                    "protocol": "min-delay",
                    "amplitude": {
                        "kind": "constant",
                        "duration": var2,
                        "value": 3.14,
                    },
                    "detuning": {
                        "kind": "ramp",
                        "duration": var2,
                        "start": 1,
                        "stop": 5,
                    },
                },
                "Pulse",
            ),
            (
                {
                    "op": "pulse",
                    "channel": "global",
                    "phase": var1,
                    "post_phase_shift": var2,
                    "protocol": "min-delay",
                    "amplitude": {
                        "kind": "constant",
                        "duration": 0,
                        "value": 3.14,
                    },
                    "detuning": {
                        "kind": "ramp",
                        "duration": var2,
                        "start": 1,
                        "stop": 5,
                    },
                },
                "ConstantAmplitude",
            ),
            (
                {
                    "op": "pulse",
                    "channel": "global",
                    "phase": var1,
                    "post_phase_shift": var2,
                    "protocol": "min-delay",
                    "amplitude": {
                        "kind": "constant",
                        "duration": var2,
                        "value": 3.14,
                    },
                    "detuning": {
                        "kind": "constant",
                        "duration": 0,
                        "value": 1,
                    },
                },
                "ConstantDetuning",
            ),
        ],
    )
    def test_deserialize_parametrized_pulse(self, op, pulse_cls):
        s = _get_serialized_seq(
            operations=[op],
            variables={
                "var1": {"type": "int", "value": [0]},
                "var2": {"type": "int", "value": [42]},
            },
        )
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset
        # No parametrized call
        assert len(seq._to_build_calls) == 1

        c = seq._to_build_calls[0]

        assert c.name == "add"
        assert c.kwargs["channel"] == op["channel"]
        assert c.kwargs["protocol"] == op["protocol"]
        pulse = c.kwargs["pulse"]
        assert isinstance(pulse, ParamObj)
        assert pulse.cls.__name__ == pulse_cls
        assert isinstance(pulse.kwargs["phase"], VariableItem)
        assert isinstance(pulse.kwargs["post_phase_shift"], VariableItem)

        if pulse_cls != "ConstantAmplitude":
            assert isinstance(pulse.kwargs["amplitude"], ParamObj)
            assert issubclass(pulse.kwargs["amplitude"].cls, Waveform)
        else:
            assert pulse.kwargs["amplitude"] == 3.14

        if pulse_cls != "ConstantDetuning":
            assert isinstance(pulse.kwargs["detuning"], ParamObj)
            assert issubclass(pulse.kwargs["detuning"].cls, Waveform)
        else:
            assert pulse.kwargs["detuning"] == 1

    def test_deserialize_eom_ops(self):
        s = _get_serialized_seq(
            operations=[
                {
                    "op": "enable_eom_mode",
                    "channel": "global",
                    "amp_on": 3.0,
                    "detuning_on": 0.0,
                    "optimal_detuning_off": -1.0,
                },
                {
                    "op": "add_eom_pulse",
                    "channel": "global",
                    "duration": {
                        "expression": "index",
                        "lhs": {"variable": "duration"},
                        "rhs": 0,
                    },
                    "phase": 0.0,
                    "post_phase_shift": 0.0,
                    "protocol": "no-delay",
                },
                {
                    "op": "disable_eom_mode",
                    "channel": "global",
                },
            ],
            variables={"duration": {"type": "int", "value": [100]}},
            device=json.loads(IroiseMVP.to_abstract_repr()),
            channels={"global": "rydberg_global"},
        )
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))
        # init + declare_channel + enable_eom_mode
        assert len(seq._calls) == 3
        # add_eom_pulse + disable_eom
        assert len(seq._to_build_calls) == 2

        enable_eom_call = seq._calls[-1]
        assert enable_eom_call.name == "enable_eom_mode"
        assert enable_eom_call.kwargs == {
            "channel": "global",
            "amp_on": 3.0,
            "detuning_on": 0.0,
            "optimal_detuning_off": -1.0,
        }

        disable_eom_call = seq._to_build_calls[-1]
        assert disable_eom_call.name == "disable_eom_mode"
        assert disable_eom_call.kwargs == {"channel": "global"}

        eom_pulse_call = seq._to_build_calls[0]
        assert eom_pulse_call.name == "add_eom_pulse"
        assert eom_pulse_call.kwargs["channel"] == "global"
        assert isinstance(eom_pulse_call.kwargs["duration"], VariableItem)
        assert eom_pulse_call.kwargs["phase"] == 0.0
        assert eom_pulse_call.kwargs["post_phase_shift"] == 0.0
        assert eom_pulse_call.kwargs["protocol"] == "no-delay"

    @pytest.mark.parametrize(
        "wf_obj",
        [
            {"kind": "constant", "duration": var1, "value": var2},
            {
                "kind": "ramp",
                "duration": var1,
                "start": var2,
                "stop": var3,
            },
            {"kind": "blackman", "duration": var1, "area": var2},
            {"kind": "blackman_max", "max_val": var3, "area": var2},
            {
                "kind": "interpolated",
                "duration": var1,
                "values": {"variable": "var_values"},
                "times": {"variable": "var_times"},
            },
            {
                "kind": "kaiser",
                "duration": var1,
                "area": var3,
                "beta": var2,
            },
            {
                "kind": "kaiser_max",
                "max_val": var2,
                "area": var2,
                "beta": var2,
            },
            {
                "kind": "composite",
                "waveforms": [
                    {
                        "kind": "constant",
                        "duration": var1,
                        "value": var2,
                    },
                    {
                        "kind": "constant",
                        "duration": var1,
                        "value": var2,
                    },
                    {
                        "kind": "constant",
                        "duration": var1,
                        "value": var2,
                    },
                ],
            },
        ],
        ids=_get_kind,
    )
    def test_deserialize_parametrized_waveform(self, wf_obj):
        # var1,2 = duration 1000, 2000
        # var2,4 = value - 2, 5
        s = _get_serialized_seq(
            operations=[
                {
                    "op": "pulse",
                    "channel": "global",
                    "phase": 1,
                    "post_phase_shift": 2,
                    "protocol": "min-delay",
                    "amplitude": wf_obj,
                    "detuning": wf_obj,
                }
            ],
            variables={
                "var1": {"type": "int", "value": [1000]},
                "var2": {"type": "int", "value": [2]},
                "var3": {"type": "int", "value": [5]},
                "var_values": {"type": "float", "value": [1, 1.5, 1.7, 1.3]},
                "var_times": {"type": "float", "value": [0, 0.4, 0.8, 0.9]},
            },
        )
        _check_roundtrip(s)
        seq = Sequence.from_abstract_repr(json.dumps(s))

        seq_var1 = seq._variables["var1"]
        seq_var2 = seq._variables["var2"]
        seq_var3 = seq._variables["var3"]
        seq_var_values = seq._variables["var_values"]
        seq_var_times = seq._variables["var_times"]

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset
        # No parametrized call
        assert len(seq._to_build_calls) == 1

        c = seq._to_build_calls[0]
        pulse: Pulse = c.kwargs["pulse"]
        wf = pulse.kwargs["amplitude"]

        if wf_obj["kind"] == "constant":
            assert wf.cls == ConstantWaveform
            assert wf.kwargs["duration"] == seq_var1[0]
            assert wf.kwargs["value"] == seq_var2[0]

        elif wf_obj["kind"] == "ramp":
            assert wf.cls == RampWaveform
            assert wf.kwargs["duration"] == seq_var1[0]
            assert wf.kwargs["start"] == seq_var2[0]
            assert wf.kwargs["stop"] == seq_var3[0]

        elif wf_obj["kind"] == "blackman":
            assert wf.cls == BlackmanWaveform
            assert wf.kwargs["duration"] == seq_var1[0]
            assert wf.kwargs["area"] == seq_var2[0]

        elif wf_obj["kind"] == "blackman_max":
            assert wf.cls == BlackmanWaveform.from_max_val.__wrapped__
            assert wf.kwargs["area"] == seq_var2[0]
            assert wf.kwargs["max_val"] == seq_var3[0]

        elif wf_obj["kind"] == "interpolated":
            assert wf.cls == InterpolatedWaveform
            assert wf.kwargs["duration"] == seq_var1[0]
            assert wf.kwargs["values"] == seq_var_values
            assert wf.kwargs["times"] == seq_var_times

        elif wf_obj["kind"] == "kaiser":
            assert wf.cls == KaiserWaveform
            assert wf.kwargs["duration"] == seq_var1[0]
            assert wf.kwargs["area"] == seq_var3[0]
            assert wf.kwargs["beta"] == seq_var2[0]

        elif wf_obj["kind"] == "kaiser_max":
            assert wf.cls == KaiserWaveform.from_max_val.__wrapped__
            assert wf.kwargs["area"] == seq_var2[0]
            assert wf.kwargs["beta"] == seq_var2[0]
            assert wf.kwargs["max_val"] == seq_var2[0]

        elif wf_obj["kind"] == "composite":
            assert wf.cls == CompositeWaveform
            assert all(isinstance(w, ParamObj) for w in wf.args)
            assert all(issubclass(w.cls, Waveform) for w in wf.args)

    @pytest.mark.parametrize(
        "json_param",
        [
            {"expression": "neg", "lhs": {"variable": "var1"}},
            {"expression": "abs", "lhs": var1},
            {"expression": "ceil", "lhs": {"variable": "var1"}},
            {"expression": "floor", "lhs": var1},
            {"expression": "sqrt", "lhs": var1},
            {"expression": "exp", "lhs": var1},
            {"expression": "log", "lhs": var1},
            {"expression": "log2", "lhs": {"variable": "var1"}},
            {"expression": "sin", "lhs": {"variable": "var1"}},
            {"expression": "cos", "lhs": var1},
            {"expression": "tan", "lhs": {"variable": "var1"}},
            {"expression": "index", "lhs": {"variable": "var1"}, "rhs": 0},
            {"expression": "add", "lhs": var1, "rhs": 0.5},
            {"expression": "sub", "lhs": {"variable": "var1"}, "rhs": 0.5},
            {"expression": "mul", "lhs": {"variable": "var1"}, "rhs": 0.5},
            {"expression": "div", "lhs": var1, "rhs": 0.5},
            {"expression": "pow", "lhs": {"variable": "var1"}, "rhs": 0.5},
            {"expression": "mod", "lhs": {"variable": "var1"}, "rhs": 2},
        ],
        ids=_get_expression,
    )
    def test_deserialize_param(self, json_param):
        s = _get_serialized_seq(
            operations=[
                {
                    "op": "pulse",
                    "channel": "global",
                    "phase": 1,
                    "post_phase_shift": 2,
                    "protocol": "min-delay",
                    "amplitude": {
                        "kind": "constant",
                        "duration": 1000,
                        "value": 2.0,
                    },
                    "detuning": {
                        "kind": "constant",
                        "duration": 1000,
                        "value": json_param,
                    },
                }
            ],
            variables={
                "var1": {"type": "float", "value": [1.5]},
            },
        )
        # Note: If built, some of these sequences will be invalid
        # since they are giving an array of size 1 to a parameter
        # where a single value is expected. Still, all we want to
        # see is whether the parametrization of the operations
        # works as expected
        if (
            json_param["lhs"] != {"variable": "var1"}
            and json_param["expression"] != "index"
        ):
            _check_roundtrip(s)

        seq = Sequence.from_abstract_repr(json.dumps(s))
        seq_var1 = seq._variables["var1"]

        # init + declare channels + 1 operation
        offset = 1 + len(s["channels"])
        assert len(seq._calls) == offset
        # No parametrized call
        assert len(seq._to_build_calls) == 1

        c = seq._to_build_calls[0]
        pulse: ParamObj = c.kwargs["pulse"]
        wf = pulse.kwargs["detuning"]
        param = wf.kwargs["value"]

        expression = json_param["expression"]
        rhs = json_param.get("rhs")

        if expression == "neg":
            assert param == -seq_var1
        if expression == "abs":
            assert param == abs(seq_var1[0])
        if expression == "ceil":
            assert param == np.ceil(seq_var1)
        if expression == "floor":
            assert param == np.floor(seq_var1[0])
        if expression == "sqrt":
            assert param == np.sqrt(seq_var1[0])
        if expression == "exp":
            assert param == np.exp(seq_var1[0])
        if expression == "log":
            assert param == np.log(seq_var1[0])
        if expression == "log2":
            assert param == np.log2(seq_var1)
        if expression == "sin":
            assert param == np.sin(seq_var1)
        if expression == "cos":
            assert param == np.cos(seq_var1[0])
        if expression == "tan":
            assert param == np.tan(seq_var1)

        if expression == "index":
            assert param == seq_var1[rhs]
        if expression == "add":
            assert param == seq_var1[0] + rhs
        if expression == "sub":
            assert param == seq_var1 - rhs
        if expression == "mul":
            assert param == seq_var1 * rhs
        if expression == "div":
            assert param == seq_var1[0] / rhs
        if expression == "pow":
            assert param == seq_var1**rhs
        if expression == "mod":
            assert param == seq_var1 % rhs

    @pytest.mark.parametrize(
        "param,msg,patch_jsonschema",
        [
            (
                var1,
                "Variable 'var1' used in operations but not found in declared "
                "variables.",
                False,
            ),
            (
                {"abs": 1},
                f"Parameter '{dict(abs=1)}' is neither a literal nor a "
                "variable or an expression.",
                True,
            ),
            (
                {"expression": "floordiv", "lhs": 0, "rhs": 0},
                "Expression 'floordiv' invalid.",
                True,
            ),
        ],
        ids=["bad_var", "bad_param", "bad_exp"],
    )
    def test_param_exceptions(self, param, msg, patch_jsonschema):
        s = _get_serialized_seq(
            [
                {
                    "op": "delay",
                    "time": param,
                    "channel": "global",
                }
            ]
        )
        extra_params = {}
        if patch_jsonschema:
            std_error = jsonschema.exceptions.ValidationError
            with patch("jsonschema.validate"):
                with pytest.raises(AbstractReprError, match=msg):
                    Sequence.from_abstract_repr(json.dumps(s))
        else:
            std_error = AbstractReprError
            extra_params["match"] = msg
        with pytest.raises(std_error, **extra_params):
            Sequence.from_abstract_repr(json.dumps(s))

    def test_unknow_waveform(self):
        s = _get_serialized_seq(
            [
                {
                    "op": "pulse",
                    "channel": "global",
                    "phase": 1,
                    "post_phase_shift": 2,
                    "protocol": "min-delay",
                    "amplitude": {
                        "kind": "constant",
                        "duration": 1000,
                        "value": 2.0,
                    },
                    "detuning": {
                        "kind": "gaussian",
                        "duration": 1000,
                        "value": -1,
                    },
                }
            ]
        )
        with pytest.raises(jsonschema.exceptions.ValidationError):
            Sequence.from_abstract_repr(json.dumps(s))

        with pytest.raises(
            AbstractReprError,
            match="The object does not encode a known waveform.",
        ):
            with patch("jsonschema.validate"):
                Sequence.from_abstract_repr(json.dumps(s))

    @pytest.mark.parametrize("device", [Chadoq2, IroiseMVP, MockDevice])
    def test_legacy_device(self, device):
        s = _get_serialized_seq(
            device=device.name, channels={"global": "rydberg_global"}
        )
        seq = Sequence.from_abstract_repr(json.dumps(s))
        assert seq.device == device
