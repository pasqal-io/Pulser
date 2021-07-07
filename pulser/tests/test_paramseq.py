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

import copy

import numpy as np
import pytest

from pulser import Sequence, Register, Pulse
from pulser.devices import Chadoq2, MockDevice
from pulser.parametrized import Variable
from pulser.waveforms import BlackmanWaveform

reg = Register.rectangle(4, 3)
device = Chadoq2


def test_var_declarations():
    sb = Sequence(reg, device)
    assert sb.declared_variables == {}
    var = sb.declare_variable("var")
    assert sb.declared_variables == {"var": var}
    assert isinstance(var, Variable)
    assert var.dtype == float
    assert len(var) == 1
    with pytest.raises(ValueError, match="already being used"):
        sb.declare_variable("var", dtype=int, size=10)
    var2 = sb.declare_variable("var2", 4, str)
    assert var2.dtype == str
    assert len(var2) == 4


def test_stored_calls():
    sb = Sequence(reg, device)
    assert sb._calls[-1].name == "__init__"
    var = sb.declare_variable("var")
    assert sb._to_build_calls == []
    sb.declare_channel("ch1", "rydberg_local", initial_target=var)
    assert sb._calls[-1].name == "declare_channel"
    assert sb._to_build_calls[-1].name == "target"
    assert sb._to_build_calls[-1].args == (var, "ch1")
    with pytest.raises(ValueError, match="name of a declared channel"):
        sb.delay(1000, "rydberg_local")
    x = Variable("x", str)
    var_ = copy.deepcopy(var)
    with pytest.raises(ValueError, match="Unknown variable 'x'"):
        sb.target(x, "ch1")
    with pytest.raises(ValueError, match="come from this Sequence"):
        sb.target(var_, "ch1")

    with pytest.raises(ValueError, match="non-variable qubits must belong"):
        sb.target("q20", "ch1")

    sb.delay(var, "ch1")
    call = sb._to_build_calls[1]
    assert call.name == "delay"
    assert call.args == (var, "ch1")
    assert call.kwargs == {}

    pls = Pulse.ConstantPulse(1000, var, var, var)

    with pytest.raises(ValueError, match="Invalid protocol 'last'"):
        sb.add(pls, "ch1", protocol="last")

    with pytest.raises(ValueError, match="amplitude goes over the maximum"):
        sb.add(
            Pulse.ConstantPulse(20, 2 * np.pi * 100, -2 * np.pi * 100, 0),
            "ch1",
        )
    with pytest.raises(
        ValueError, match="detuning values go out of the range"
    ):
        sb.add(Pulse.ConstantPulse(500, 2 * np.pi, -2 * np.pi * 100, 0), "ch1")

    assert sb._to_build_calls[-1] == call
    sb.add(pls, "ch1", protocol="wait-for-all")
    call = sb._to_build_calls[2]
    assert call.name == "add"
    assert call.args == (pls, "ch1")
    assert call.kwargs == {"protocol": "wait-for-all"}

    q_var = sb.declare_variable("q_var", size=5, dtype=str)
    sb.declare_channel("ch2", "rydberg_global")
    assert len(sb._calls) == 3
    assert sb._calls[-1].name == "declare_channel"
    with pytest.raises(ValueError, match="'Local' channels"):
        sb.target(0, "ch2")
    with pytest.raises(ValueError, match="target at most 1 qubits"):
        sb.target(q_var, "ch1")

    sb2 = Sequence(reg, MockDevice)
    sb2.declare_channel("ch1", "rydberg_local", initial_target={3, 4, 5})
    q_var2 = sb2.declare_variable("q_var2", size=5, dtype=str)
    var2 = sb2.declare_variable("var2")
    assert sb2._building
    sb2.target({var2, 7, 9, 10}, "ch1")
    assert not sb2._building
    sb2.target(q_var2, "ch1")

    with pytest.raises(ValueError, match="targets the given 'basis'"):
        sb.phase_shift(var, *q_var)

    with pytest.raises(ValueError, match="non-variable targets must belong"):
        sb.phase_shift(var, *q_var, "q1", basis="ground-rydberg")

    with pytest.raises(ValueError, match="correspond to declared channels"):
        sb.align("ch1", var)
    with pytest.raises(ValueError, match="more than once"):
        sb.align("ch1", "ch2", "ch2")
    with pytest.raises(ValueError, match="at least two channels"):
        sb.align("ch1")

    with pytest.raises(ValueError, match="not supported"):
        sb.measure(basis="z")
    sb.measure()
    with pytest.raises(RuntimeError):
        sb.delay(var * 50, "ch1")


def test_build():
    reg_ = Register.rectangle(2, 1, prefix="q")
    sb = Sequence(reg_, device)
    var = sb.declare_variable("var")
    targ_var = sb.declare_variable("targ_var", size=2, dtype=str)
    sb.declare_channel("ch1", "rydberg_local")
    sb.declare_channel("ch2", "raman_local", initial_target=targ_var[0])
    sb.target(targ_var[1], "ch1")
    wf = BlackmanWaveform(var * 100, np.pi)
    pls = Pulse.ConstantDetuning(wf, var, var)
    sb.add(pls, "ch1")
    sb.delay(var * 50, "ch1")
    sb.align("ch2", "ch1")
    sb.phase_shift(var, targ_var[0])
    pls2 = Pulse.ConstantPulse(wf.duration, var, var, 0)
    sb.add(pls2, "ch2")
    sb.measure()
    with pytest.warns(UserWarning, match="No declared variables"):
        sb.build(t=100, var=2, targ_var=["q1", "q0"])
    with pytest.raises(TypeError, match="Did not receive values for"):
        sb.build(var=2)
    seq = sb.build(var=2, targ_var=["q1", "q0"])
    assert seq._schedule["ch2"][-1].tf == 500
    assert seq.current_phase_ref("q1") == 2.0
    assert seq.current_phase_ref("q0") == 0.0
    assert seq._measurement == "ground-rydberg"

    s = sb.serialize()
    sb_ = Sequence.deserialize(s)
    assert str(sb) == str(sb_)

    s2 = sb_.serialize()
    sb_2 = Sequence.deserialize(s2)
    assert str(sb) == str(sb_2)


def test_str():
    reg_ = Register.rectangle(2, 1, prefix="q")
    sb = Sequence(reg_, device)
    sb.declare_channel("ch1", "rydberg_global")
    with pytest.warns(UserWarning, match="Building a non-parametrized"):
        seq = sb.build()
    var = sb.declare_variable("var")
    pls = Pulse.ConstantPulse(var * 100, var, -1, var)
    sb.add(pls, "ch1")
    s = (
        f"Prelude\n-------\n{str(seq)}Stored calls\n------------\n\n"
        + "1. add(Pulse.ConstantPulse(mul(var, 100), var, -1, var), ch1)"
    )
    assert s == str(sb)


def test_screen():
    sb = Sequence(reg, device)
    sb.declare_channel("ch1", "rydberg_global")
    assert sb.current_phase_ref(4, basis="ground-rydberg") == 0
    var = sb.declare_variable("var")
    sb.delay(var, "ch1")
    with pytest.raises(RuntimeError, match="can't be called in parametrized"):
        sb.current_phase_ref(4, basis="ground-rydberg")
