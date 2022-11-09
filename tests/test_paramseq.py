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

from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2, MockDevice
from pulser.parametrized import Variable
from pulser.parametrized.variable import VariableItem
from pulser.waveforms import BlackmanWaveform

reg = Register.rectangle(4, 3)
device = Chadoq2


def test_var_declarations():
    sb = Sequence(reg, device)
    assert sb.declared_variables == {}
    var = sb.declare_variable("var", size=1)
    assert sb.declared_variables == {"var": var}
    assert isinstance(var, Variable)
    assert var.dtype == float
    assert var.size == 1
    with pytest.raises(ValueError, match="already being used"):
        sb.declare_variable("var", dtype=int, size=10)
    var3 = sb.declare_variable("var3")
    assert sb.declared_variables["var3"] == var3.var
    assert isinstance(var3, VariableItem)
    with pytest.raises(ValueError, match="'qubits' is a protected name"):
        sb.declare_variable("qubits", size=10, dtype=int)


def test_parametrized_channel_initial_target():
    sb = Sequence(reg, device)
    var = sb.declare_variable("var")
    sb.declare_channel("ch1", "rydberg_local")
    sb.target_index(var, "ch1")
    sb.declare_channel("ch0", "raman_local", initial_target=0)
    assert sb._calls[-1].name == "declare_channel"
    assert sb._to_build_calls[-1].name == "target"
    assert sb._to_build_calls[-1].args == (0, "ch0")


def test_stored_calls():
    sb = Sequence(reg, device)
    assert sb._calls[-1].name == "__init__"
    var = sb.declare_variable("var")
    assert sb._to_build_calls == []
    with pytest.raises(
        TypeError, match="initial_target cannot be parametrized"
    ):
        sb.declare_channel("ch1", "rydberg_local", initial_target=var)
    sb.declare_channel("ch1", "rydberg_local")
    sb.target_index(var, "ch1")
    assert sb._calls[-1].name == "declare_channel"
    assert sb._to_build_calls[-1].name == "target_index"
    assert sb._to_build_calls[-1].args == (var, "ch1")
    with pytest.raises(ValueError, match="name of a declared channel"):
        sb.delay(1000, "rydberg_local")
    x = Variable("x", int)
    var_ = copy.deepcopy(var)
    with pytest.raises(ValueError, match="Unknown variable 'x'"):
        sb.target_index(x, "ch1")
    with pytest.raises(ValueError, match="come from this Sequence"):
        sb.target(var_, "ch1")

    with pytest.raises(ValueError, match="ids have to be qubit ids"):
        sb.target("q20", "ch1")

    with pytest.raises(
        NotImplementedError,
        match="Using parametrized objects or variables to refer to channels",
    ):
        sb.target("q0", var)
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

    q_var = sb.declare_variable("q_var", size=5, dtype=int)
    sb.declare_channel("ch2", "rydberg_global")
    assert len(sb._calls) == 3
    assert sb._calls[-1].name == "declare_channel"
    with pytest.raises(ValueError, match="'Local' channels"):
        sb.target(0, "ch2")
    with pytest.raises(ValueError, match="target at most 1 qubits"):
        sb.target_index(q_var, "ch1")

    sb2 = Sequence(reg, MockDevice)
    sb2.declare_channel("ch1", "rydberg_local", initial_target={3, 4, 5})
    q_var2 = sb2.declare_variable("q_var2", size=5, dtype=int)
    var2 = sb2.declare_variable("var2")
    assert sb2._building
    with pytest.raises(ValueError, match="ids have to be qubit ids"):
        sb2.target({var2, 7, 9, 10}, "ch1")
    sb2.target_index(q_var2, "ch1")
    assert not sb2._building

    with pytest.raises(ValueError, match="targets the given 'basis'"):
        sb.phase_shift_index(var, *q_var)

    with pytest.raises(
        ValueError,
        match="ids have to be qubit ids",
    ):
        sb.phase_shift(var, "wacky_id", basis="ground-rydberg")
    with pytest.raises(ValueError, match="ids have to be qubit ids"):
        sb2.phase_shift(np.pi, var2, basis="ground-rydberg")
    with pytest.raises(
        ValueError,
        match="All non-variable targets must be indices valid for the"
        " register, between 0 and 11. Wrong index: 'q1'.",
    ):
        sb.phase_shift_index(var, *q_var, "q1", basis="ground-rydberg")
    with pytest.raises(
        ValueError,
        match="All non-variable targets must be indices valid for the"
        " register, between 0 and 11. Wrong index: 'q1'.",
    ):
        sb.target_index("q1", channel="ch1")

    with pytest.raises(ValueError, match="correspond to declared channels"):
        sb.align("ch1", var)
    with pytest.raises(ValueError, match="more than once"):
        sb.align("ch1", "ch2", "ch2")
    with pytest.raises(ValueError, match="at least two channels"):
        sb.align("ch1")

    with pytest.raises(ValueError, match="not supported"):
        sb.measure(basis=var)

    sb.measure()
    with pytest.raises(RuntimeError):
        sb.delay(var * 50, "ch1")


def test_build():
    reg_ = Register.rectangle(2, 1, prefix="q")
    sb = Sequence(reg_, device)
    var = sb.declare_variable("var")
    targ_var = sb.declare_variable("targ_var", size=2, dtype=int)
    sb.declare_channel("ch1", "rydberg_local")
    sb.declare_channel("ch2", "raman_local")
    sb.target_index(targ_var[0], "ch2")
    sb.target_index(targ_var[1], "ch1")
    wf = BlackmanWaveform(var * 100, np.pi)
    pls = Pulse.ConstantDetuning(wf, var, var)
    sb.add(pls, "ch1")
    sb.delay(var * 50, "ch1")
    sb.align("ch2", "ch1")
    sb.phase_shift_index(var, targ_var[0])
    pls2 = Pulse.ConstantPulse(var * 100, var, var, 0)
    sb.add(pls2, "ch2")
    sb.measure()
    with pytest.warns(UserWarning, match="No declared variables"):
        sb.build(t=100, var=2, targ_var=reg_.find_indices(["q1", "q0"]))
    with pytest.raises(TypeError, match="Did not receive values for"):
        sb.build(var=2)
    seq = sb.build(var=2, targ_var=reg_.find_indices(["q1", "q0"]))
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
        + "1. add(Pulse(ConstantWaveform(mul(var[0], 100), var[0]), "
        + "ConstantWaveform(mul(var[0], 100), -1), var[0], 0.0), ch1)"
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


def test_parametrized_in_eom_mode(mod_device):
    # Case 1: Sequence becomes parametrized while in EOM mode
    seq = Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_local", initial_target=0)

    assert not seq.is_in_eom_mode("ch0")
    seq.enable_eom_mode("ch0", amp_on=2.0, detuning_on=0.0)
    assert seq.is_in_eom_mode("ch0")
    assert not seq.is_parametrized()

    dt = seq.declare_variable("dt", dtype=int)
    seq.add_eom_pulse("ch0", dt, 0.0)

    assert seq.is_in_eom_mode("ch0")
    assert seq.is_parametrized()

    with pytest.raises(
        RuntimeError, match="The 'ch0' channel is already in EOM mode"
    ):
        seq.enable_eom_mode("ch0", amp_on=2.0, detuning_on=0.0)

    with pytest.raises(
        RuntimeError, match="The chosen channel is in EOM mode"
    ):
        seq.target_index(1, "ch0")

    seq.disable_eom_mode("ch0")
    assert not seq.is_in_eom_mode("ch0")

    with pytest.raises(
        RuntimeError, match="The 'ch0' channel is not in EOM mode"
    ):
        seq.disable_eom_mode("ch0")

    # Just check that building works
    seq.build(dt=100)


def test_parametrized_before_eom_mode(mod_device):
    # Case 2: Sequence is parametrized before entering EOM mode
    seq = Sequence(reg, mod_device)

    seq.declare_channel("ch0", "rydberg_local", initial_target=0)
    seq.declare_channel("raman", "raman_local", initial_target=2)
    amp = seq.declare_variable("amp", dtype=float)
    seq.add(Pulse.ConstantPulse(200, amp, -1, 0), "ch0")

    assert not seq.is_in_eom_mode("ch0")
    assert seq.is_parametrized()

    # Validation is still done whenever possible
    with pytest.raises(
        RuntimeError, match="Channel 'ch0' must be in EOM mode."
    ):
        seq.add_eom_pulse("ch0", 100, 0.0)

    with pytest.raises(
        ValueError, match="Invalid option for the detuning choice"
    ):
        seq.enable_eom_mode("ch0", amp, 0.0, detuning_off_choice="zero")

    with pytest.raises(
        TypeError, match="Channel 'raman' does not have an EOM"
    ):
        seq.enable_eom_mode("raman", 1.0, 0.0)

    with pytest.raises(
        ValueError,
        match="The pulse's amplitude goes over the maximum "
        "value allowed for the chosen channel.",
    ):
        seq.enable_eom_mode("ch0", 10000, 0.0)

    seq.enable_eom_mode("ch0", amp_on=amp, detuning_on=0.0)
    assert seq.is_in_eom_mode("ch0")

    # Validation still works
    with pytest.raises(ValueError, match="Invalid protocol 'smallest'"):
        seq.add_eom_pulse("ch0", 1000, 0.0, protocol="smallest")

    with pytest.raises(
        TypeError, match="Phase values must be a numeric value."
    ):
        seq.add_eom_pulse("ch0", 200, "0.")

    with pytest.raises(ValueError, match="duration has to be at least"):
        seq.add_eom_pulse("ch0", 0, 0.0)

    seq.add_eom_pulse("ch0", 100, 0.0)

    seq.disable_eom_mode("ch0")
    assert not seq.is_in_eom_mode("ch0")

    # Just check that building works
    seq.build(amp=3.0)
