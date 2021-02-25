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
import pytest

from pulser import SequenceBuilder, Register, Pulse
from pulser.devices import Chadoq2
from pulser.parametrized import Variable

reg = Register.rectangle(4, 3)
device = Chadoq2


def test_properties():
    sb = SequenceBuilder(reg, device)
    assert sb.qubit_info.keys() == set(range(12))
    assert sb.declared_channels == {}
    assert sb.available_channels.keys() == Chadoq2.channels.keys()
    sb.declare_channel("ch1", "rydberg_local")
    assert sb.declared_channels == {"ch1": Chadoq2.channels["rydberg_local"]}
    assert sb.available_channels.keys() == (Chadoq2.channels.keys()
                                            - {"rydberg_local"})
    assert sb.declared_variables == {}
    var = sb.declare_variable("var")
    assert sb.declared_variables == {"var": var}


def test_declarations():
    sb = SequenceBuilder(reg, device)
    var = sb.declare_variable("var")
    assert isinstance(var, Variable)
    assert var.dtype == float
    assert len(var) == 1
    with pytest.raises(ValueError, match="already being used"):
        sb.declare_variable("var", dtype=int, size=10)
    var2 = sb.declare_variable("var2", 4, str)
    assert var2.dtype == str
    assert len(var2) == 4

    with pytest.raises(ValueError, match="target has to be a fixed qubit"):
        sb.declare_channel("ch2", "rydberg_local", var)


def test_stored_calls():
    sb = SequenceBuilder(reg, device)
    var = sb.declare_variable("var")
    sb.declare_channel("ch1", "rydberg_local")
    assert sb._calls == []
    with pytest.raises(ValueError, match="name of a declared channel"):
        sb.delay(1000, "rydberg_local")
    x = Variable("x", str)
    var_ = copy.deepcopy(var)
    with pytest.raises(ValueError, match="Unknown variable 'x'"):
        sb.target(x, "ch1")
    with pytest.raises(ValueError, match="come from this SequenceBuilder"):
        sb.target(var_, "ch1")

    sb.delay(var, "ch1")
    call = sb._calls[0]
    assert call.name == "delay"
    assert call.args == (var, "ch1")
    assert call.kwargs == {}

    pls = Pulse.ConstantPulse(1000, var, var, var)

    with pytest.raises(ValueError, match="Invalid protocol 'last'"):
        sb.add(pls, "ch1", protocol="last")
    assert sb._calls[-1] == call
    sb.add(pls, "ch1", protocol="wait-for-all")
    call = sb._calls[1]
    assert call.name == "add"
    assert call.args == (pls, "ch1")
    assert call.kwargs == {"protocol": "wait-for-all"}

    q_var = sb.declare_variable("q_var", size=5, dtype=str)
    sb.declare_channel("ch2", "rydberg_global")
    with pytest.raises(ValueError, match="'Local' channels"):
        sb.target(0, "ch2")
    with pytest.raises(ValueError, match="target at most 1 qubits"):
        sb.target(q_var, "ch1")

    with pytest.raises(ValueError, match="targets the given 'basis'"):
        sb.phase_shift(var, *q_var)

    with pytest.raises(ValueError, match="correspond to declared channels"):
        sb.align("ch1", var)
    with pytest.raises(ValueError, match="more than once"):
        sb.align("ch1", "ch2", "ch2")
    with pytest.raises(ValueError, match="at least two channels"):
        sb.align("ch1")

    with pytest.raises(ValueError, match="not supported"):
        sb.measure(basis="z")
    sb.measure()
    with pytest.raises(SystemError):
        sb.delay(var*50, "ch1")


def test_build():
    reg_ = Register.rectangle(2, 1, prefix="q")
    sb = SequenceBuilder(reg_, device)
    var = sb.declare_variable("var")
    targ_var = sb.declare_variable("targ_var", size=2, dtype=str)
    sb.declare_channel("ch1", "rydberg_local")
    sb.declare_channel("ch2", "raman_local")
    sb.target(targ_var[1], "ch1")
    pls = Pulse.ConstantPulse(var*100, var, var, var)
    sb.add(pls, "ch1")
    sb.delay(var*50, "ch1")
    sb.target(targ_var[0], "ch2")
    sb.align("ch2", "ch1")
    sb.phase_shift(var, targ_var[0])
    sb.measure()
    with pytest.raises(TypeError, match="No declared variables"):
        sb.build(t=100, var=2, targ_var=["q1", "q0"])
    with pytest.raises(TypeError, match="Did not receive values for"):
        sb.build(var=2)
    seq = sb.build(var=2, targ_var=["q1", "q0"])
    assert seq._schedule["ch2"][-1].tf == 300
    assert seq.current_phase_ref("q1") == 2.0
    assert seq.current_phase_ref("q0") == 0.
    assert seq._measurement == "ground-rydberg"


def test_str():
    reg_ = Register.rectangle(2, 1, prefix="q")
    sb = SequenceBuilder(reg_, device)
    sb.declare_channel("ch1", "rydberg_global")
    seq = sb.build()
    var = sb.declare_variable("var")
    pls = Pulse.ConstantPulse(var*100, var, -1, var)
    sb.add(pls, "ch1")
    s = (f"Prelude\n-------\n{str(seq)}Stored calls\n------------\n\n"
         + "1. add(Pulse(ConstantWaveform(mul(var, 100), var), "
         + "ConstantWaveform(mul(var, 100), -1), var, 0), ch1)")
    assert s == str(sb)
