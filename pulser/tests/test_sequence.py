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
from unittest.mock import patch

import numpy as np
import pytest

import pulser
from pulser import Sequence, Pulse, Register, Register3D
from pulser.devices import Chadoq2, MockDevice
from pulser.devices._device_datacls import Device
from pulser.sequence import _TimeSlot
from pulser.waveforms import (
    BlackmanWaveform,
    CompositeWaveform,
    RampWaveform,
    InterpolatedWaveform,
)

reg = Register.triangular_lattice(4, 7, spacing=5, prefix="q")
device = Chadoq2


def test_init():
    with pytest.raises(TypeError, match="must be of type 'Device'"):
        Sequence(reg, Device)

    fake_device = Device("fake", 2, 70, 100, 100, 1, Chadoq2._channels)
    with pytest.warns(UserWarning, match="imported from 'pulser.devices'"):
        Sequence(reg, fake_device)

    seq = Sequence(reg, device)
    assert seq.qubit_info == reg.qubits
    assert seq.declared_channels == {}
    assert seq.available_channels.keys() == device.channels.keys()


def test_channel_declaration():
    seq = Sequence(reg, device)
    available_channels = set(seq.available_channels)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "raman_local")
    with pytest.raises(ValueError, match="No channel"):
        seq.declare_channel("ch2", "raman")
    with pytest.raises(ValueError, match="not available"):
        seq.declare_channel("ch2", "rydberg_global")
    with pytest.raises(ValueError, match="name is already in use"):
        seq.declare_channel("ch0", "raman_local")

    chs = {"rydberg_global", "raman_local"}
    assert seq._schedule["ch0"][-1] == _TimeSlot(
        "target", -1, 0, set(seq.qubit_info.keys())
    )
    assert set(seq.available_channels) == available_channels - chs

    seq2 = Sequence(reg, MockDevice)
    available_channels = set(seq2.available_channels)
    seq2.declare_channel("ch0", "raman_local", initial_target="q1")
    seq2.declare_channel("ch1", "rydberg_global")
    seq2.declare_channel("ch2", "rydberg_global")
    assert set(seq2.available_channels) == (available_channels - {"mw_global"})
    assert seq2._taken_channels == {
        "ch0": "raman_local",
        "ch1": "rydberg_global",
        "ch2": "rydberg_global",
    }
    assert seq2._taken_channels.keys() == seq2._channels.keys()
    with pytest.raises(ValueError, match="type 'Microwave' cannot work "):
        seq2.declare_channel("ch3", "mw_global")

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "mw_global")
    assert set(seq2.available_channels) == {"mw_global"}
    with pytest.raises(
        ValueError,
        match="cannot work simultaneously with the declared 'Microwave'",
    ):
        seq2.declare_channel("ch3", "rydberg_global")


def test_magnetic_field():
    seq = Sequence(reg, MockDevice)
    with pytest.raises(
        AttributeError,
        match="only defined when the sequence " "is in 'XY Mode'.",
    ):
        seq.magnetic_field
    seq.declare_channel("ch0", "mw_global")  # seq in XY mode
    # mag field is the default
    assert np.all(seq.magnetic_field == np.array((0.0, 0.0, 30.0)))
    seq.set_magnetic_field(bx=1.0, by=-1.0, bz=0.5)
    assert np.all(seq.magnetic_field == np.array((1.0, -1.0, 0.5)))
    with pytest.raises(ValueError, match="magnitude greater than 0"):
        seq.set_magnetic_field(bz=0.0)
    assert seq._empty_sequence
    seq.add(Pulse.ConstantPulse(100, 1, 1, 0), "ch0")
    assert not seq._empty_sequence
    with pytest.raises(ValueError, match="can only be set on an empty seq"):
        seq.set_magnetic_field(1.0, 0.0, 0.0)

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "rydberg_global")  # not in XY mode
    with pytest.raises(ValueError, match="can only be set in 'XY Mode'."):
        seq2.set_magnetic_field(1.0, 0.0, 0.0)

    seq3 = Sequence(reg, MockDevice)
    seq3.set_magnetic_field(1.0, 0.0, 0.0)  # sets seq to XY mode
    assert set(seq3.available_channels) == {"mw_global"}
    seq3.declare_channel("ch0", "mw_global")
    # Does not change to default
    assert np.all(seq3.magnetic_field == np.array((1.0, 0.0, 0.0)))
    var = seq3.declare_variable("var")
    # Sequence is marked as non-empty when parametrized too
    seq3.add(Pulse.ConstantPulse(100, var, 1, 0), "ch0")
    assert seq3.is_parametrized()
    with pytest.raises(ValueError, match="can only be set on an empty seq"):
        seq3.set_magnetic_field()

    seq3_str = seq3.serialize()
    seq3_ = Sequence.deserialize(seq3_str)
    assert seq3_._in_xy
    assert str(seq3) == str(seq3_)
    assert np.all(seq3_.magnetic_field == np.array((1.0, 0.0, 0.0)))


def test_target():
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q1")
    seq.declare_channel("ch1", "rydberg_global")

    with pytest.raises(ValueError, match="name of a declared channel"):
        seq.target("q0", "ch2")
    with pytest.raises(ValueError, match="qubits must belong"):
        seq.target(0, "ch0")
        seq.target("0", "ch1")
    with pytest.raises(ValueError, match="Can only choose target of 'Local'"):
        seq.target("q3", "ch1")
    with pytest.raises(ValueError, match="can target at most 1 qubits"):
        seq.target(["q1", "q5"], "ch0")

    assert seq._schedule["ch0"][-1] == _TimeSlot("target", -1, 0, {"q1"})
    seq.target("q4", "ch0")
    retarget_t = seq.declared_channels["ch0"].retarget_time
    assert seq._schedule["ch0"][-1] == _TimeSlot(
        "target", 0, retarget_t, {"q4"}
    )
    seq.target("q4", "ch0")  # targets the same qubit
    seq.target("q20", "ch0")
    assert seq._schedule["ch0"][-1] == _TimeSlot(
        "target", retarget_t, 2 * retarget_t, {"q20"}
    )
    seq.delay(216, "ch0")
    seq.target("q2", "ch0")
    ti = 2 * retarget_t + 216
    tf = ti + 16
    assert seq._schedule["ch0"][-1] == _TimeSlot("target", ti, tf, {"q2"})

    seq.delay(220, "ch0")
    seq.target("q1", "ch0")
    ti = tf + 220
    tf = ti
    assert seq._schedule["ch0"][-1] == _TimeSlot("target", ti, tf, {"q1"})

    seq.delay(100, "ch0")
    seq.target("q10", "ch0")
    ti = tf + 100
    tf = ti + 120
    assert seq._schedule["ch0"][-1] == _TimeSlot("target", ti, tf, {"q10"})

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "raman_local", initial_target={"q1", "q10"})
    seq2.phase_shift(1, "q2")
    with pytest.raises(ValueError, match="qubits with different phase"):
        seq2.target({"q3", "q1", "q2"}, "ch0")


def test_delay():
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local")
    with pytest.raises(ValueError, match="Use the name of a declared channel"):
        seq.delay(1e3, "ch01")
    with pytest.raises(ValueError, match="channel has no target"):
        seq.delay(100, "ch0")
    seq.target("q19", "ch0")
    seq.delay(388, "ch0")
    assert seq._last("ch0") == _TimeSlot("delay", 0, 388, {"q19"})


def test_delay_min_duration():
    # Check that a delay shorter than a channel's minimal duration
    # is automatically extended to that minimal duration
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "rydberg_local")
    seq.target("q0", "ch1")
    pulse0 = Pulse.ConstantPulse(52, 1, 1, 0)
    pulse1 = Pulse.ConstantPulse(180, 1, 1, 0)
    seq.add(pulse1, "ch1")
    seq.add(pulse0, "ch0")
    seq.target("q1", "ch1")
    seq.add(pulse1, "ch1")
    min_duration = seq._channels["ch1"].min_duration
    assert seq._schedule["ch1"][3] == _TimeSlot(
        "delay", 220, 220 + min_duration, {"q1"}
    )


def test_min_pulse_duration():
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "rydberg_local")
    seq.target("q0", "ch1")
    pulse0 = Pulse.ConstantPulse(60, 1, 1, 0)
    pulse1 = Pulse.ConstantPulse(80, 1, 1, 0)
    seq.add(pulse1, "ch1")
    assert seq._min_pulse_duration() == 80
    seq.add(pulse0, "ch0")
    seq.delay(52, "ch0")
    seq.target("q1", "ch1")
    seq.add(pulse1, "ch1")
    assert seq._min_pulse_duration() == 60


def test_phase():
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    seq.phase_shift(-1, "q0", "q1")
    with pytest.raises(ValueError, match="id of a qubit declared"):
        seq.current_phase_ref(0, "digital")
    with pytest.raises(ValueError, match="targets the given 'basis'"):
        seq.current_phase_ref("q1", "ground-rydberg")
    with pytest.raises(ValueError, match="No declared channel targets"):
        seq.phase_shift(1, "q3", basis="hyperfine")
    assert seq.current_phase_ref("q0", "digital") == 2 * np.pi - 1

    # Phase shifts of 0
    seq.phase_shift(0, "q0")
    seq.phase_shift(-8 * np.pi, "q1")
    assert seq.current_phase_ref("q0", "digital") == 2 * np.pi - 1
    assert seq.current_phase_ref("q1", "digital") == 2 * np.pi - 1

    with pytest.raises(ValueError, match="targets have to be qubit ids"):
        seq.phase_shift(np.pi, "q1", "q4", "q100")

    seq.declare_channel("ch1", "rydberg_global")
    seq.phase_shift(1, *seq._qids, basis="ground-rydberg")
    for q in seq.qubit_info:
        assert seq.current_phase_ref(q, "ground-rydberg") == 1
    seq.phase_shift(1, *seq._qids)
    assert seq.current_phase_ref("q1", "digital") == 0
    assert seq.current_phase_ref("q10", "digital") == 1


def test_align():
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    seq.declare_channel("ch1", "rydberg_global")
    with pytest.raises(ValueError, match="names must correspond to declared"):
        seq.align("ch0", "ch1", "ch2")
    with pytest.raises(ValueError, match="more than once"):
        seq.align("ch0", "ch1", "ch0")
    with pytest.raises(ValueError, match="at least two channels"):
        seq.align()
        seq.align("ch1")


def test_measure():
    pulse = Pulse.ConstantPulse(500, 2, -10, 0, post_phase_shift=np.pi)
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    assert "XY" in MockDevice.supported_bases
    with pytest.raises(ValueError, match="not supported"):
        seq.measure(basis="XY")
    seq.measure()
    with pytest.raises(RuntimeError, match="already been measured"):
        seq.measure(basis="digital")
    with pytest.raises(RuntimeError, match="Nothing more can be added."):
        seq.add(pulse, "ch0")

    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "mw_global")
    assert "digital" in MockDevice.supported_bases
    with pytest.raises(ValueError, match="not supported"):
        seq.measure(basis="digital")
    seq.measure(basis="XY")


def test_str():
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    pulse = Pulse.ConstantPulse(500, 2, -10, 0, post_phase_shift=np.pi)
    seq.add(pulse, "ch0")
    seq.delay(200, "ch0")
    seq.target("q7", "ch0")
    seq.measure("digital")
    msg = (
        "Channel: ch0\nt: 0 | Initial targets: q0 | Phase Reference: 0.0 "
        + "\nt: 0->500 | Pulse(Amp=2 rad/µs, Detuning=-10 rad/µs, Phase=0) "
        + "| Targets: q0\nt: 500->700 | Delay \nt: 700->700 | Target: q7 | "
        + "Phase Reference: 0.0\n\nMeasured in basis: digital"
    )
    assert seq.__str__() == msg


def test_sequence():
    seq = Sequence(reg, device)
    assert seq.get_duration() == 0
    with pytest.raises(RuntimeError, match="empty sequence"):
        seq.draw()
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    seq.declare_channel("ch1", "rydberg_local", initial_target="q0")
    seq.declare_channel("ch2", "rydberg_global")
    assert seq.get_duration("ch0") == 0
    assert seq.get_duration("ch2") == 0
    seq.phase_shift(np.pi, "q0", basis="ground-rydberg")

    with patch("matplotlib.pyplot.show"):
        with patch("matplotlib.figure.Figure.savefig"):
            seq.draw(fig_name="my_sequence.pdf")
            seq.draw(draw_register=True, fig_name="both.pdf")

    pulse1 = Pulse(
        InterpolatedWaveform(500, [0, 1, 0]),
        InterpolatedWaveform(500, [-1, 1, 0]),
        phase=0,
        post_phase_shift=np.pi,
    )
    pulse2 = Pulse.ConstantDetuning(
        BlackmanWaveform(1e3, np.pi / 4), 25, np.pi, post_phase_shift=1
    )
    with pytest.raises(TypeError):
        seq.add([1, 5, 3], "ch0")
    with pytest.raises(ValueError, match="amplitude goes over the maximum"):
        seq.add(
            Pulse.ConstantPulse(20, 2 * np.pi * 10, -2 * np.pi * 100, 0), "ch2"
        )
    with pytest.raises(
        ValueError, match="detuning values go out of the range"
    ):
        seq.add(
            Pulse.ConstantPulse(500, 2 * np.pi, -2 * np.pi * 100, 0), "ch0"
        )
    with pytest.raises(ValueError, match="qubits with different phase ref"):
        seq.add(pulse2, "ch2")
    with pytest.raises(ValueError, match="Invalid protocol"):
        seq.add(pulse1, "ch0", protocol="now")

    wf_ = CompositeWaveform(BlackmanWaveform(30, 1), RampWaveform(15, 0, 2))
    with pytest.raises(TypeError, match="Failed to automatically adjust"):
        with pytest.warns(UserWarning, match="rounded up to 48 ns"):
            seq.add(Pulse.ConstantAmplitude(1, wf_, 0), "ch0")

    pulse1_ = Pulse.ConstantPulse(499, 2, -10, 0, post_phase_shift=np.pi)
    with pytest.warns(UserWarning, match="rounded up to 500 ns"):
        seq.add(pulse1_, "ch0")
    seq.add(pulse1, "ch1")
    seq.add(pulse2, "ch2")

    assert seq._last("ch0").ti == 0
    assert seq._last("ch0").tf == seq._last("ch1").ti
    assert seq._last("ch2").tf == seq._last("ch2").ti + 1000
    assert seq.current_phase_ref("q0", "digital") == np.pi

    seq.add(pulse1, "ch2")
    assert seq.get_duration("ch2") == 2500
    seq.add(pulse2, "ch1", protocol="no-delay")
    assert seq.get_duration("ch1") == 3500
    seq.add(pulse1, "ch0", protocol="no-delay")
    assert seq._last("ch0").ti == 500
    assert seq.get_duration("ch0") == 1000
    assert seq.current_phase_ref("q0", "digital") == 0
    seq.phase_shift(np.pi / 2, "q1")
    seq.target("q1", "ch0")
    assert seq._last_used["digital"]["q1"] == 0
    assert seq._last_target["ch0"] == 1000
    assert seq._last("ch0").ti == 1000
    assert seq.get_duration("ch0") == 1000
    seq.add(pulse1, "ch0")
    assert seq._last("ch0").ti == 2500
    assert seq.get_duration("ch0") == 3000
    seq.add(pulse1, "ch0", protocol="wait-for-all")
    assert seq._last("ch0").ti == 3500
    assert seq.get_duration("ch2") != seq.get_duration("ch0")
    seq.align("ch0", "ch2")
    assert seq.get_duration("ch2") == seq.get_duration("ch0")

    with patch("matplotlib.pyplot.show"):
        seq.draw(draw_phase_shifts=True)

    assert seq.get_duration() == 4000

    seq.measure(basis="digital")

    with patch("matplotlib.pyplot.show"):
        seq.draw(draw_phase_area=True)

    s = seq.serialize()
    assert json.loads(s)["__version__"] == pulser.__version__
    seq_ = Sequence.deserialize(s)
    assert str(seq) == str(seq_)


def test_config_slm_mask():
    reg_s = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    seq_s = Sequence(reg_s, device)

    with pytest.raises(TypeError, match="must be castable to set"):
        seq_s.config_slm_mask(0)
    with pytest.raises(TypeError, match="must be castable to set"):
        seq_s.config_slm_mask((0))
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask("q0")
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask(["q3"])
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask(("q3",))
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask({"q3"})
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask([0])
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask((0,))
    with pytest.raises(ValueError, match="exist in the register"):
        seq_s.config_slm_mask({0})

    targets_s = ["q0", "q2"]
    seq_s.config_slm_mask(targets_s)
    assert seq_s._slm_mask_targets == {"q0", "q2"}

    with pytest.raises(ValueError, match="configured only once"):
        seq_s.config_slm_mask(targets_s)

    reg_i = Register({0: (0, 0), 1: (10, 10), 2: (-10, -10)})
    seq_i = Sequence(reg_i, device)

    with pytest.raises(TypeError, match="must be castable to set"):
        seq_i.config_slm_mask(0)
    with pytest.raises(TypeError, match="must be castable to set"):
        seq_i.config_slm_mask((0))
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask("q0")
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask([3])
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask((3,))
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask({3})
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask(["0"])
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask(("0",))
    with pytest.raises(ValueError, match="exist in the register"):
        seq_i.config_slm_mask({"0"})

    targets_i = [0, 2]
    seq_i.config_slm_mask(targets_i)
    assert seq_i._slm_mask_targets == {0, 2}

    with pytest.raises(ValueError, match="configured only once"):
        seq_i.config_slm_mask(targets_i)


def test_slm_mask():
    reg = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    targets = ["q0", "q2"]
    pulse1 = Pulse.ConstantPulse(100, 10, 0, 0)
    pulse2 = Pulse.ConstantPulse(200, 10, 0, 0)

    # Set mask when an XY pulse is already in the schedule
    seq_xy1 = Sequence(reg, MockDevice)
    seq_xy1.declare_channel("ch_xy", "mw_global")
    seq_xy1.add(pulse1, "ch_xy")
    seq_xy1.config_slm_mask(targets)
    assert seq_xy1._slm_mask_time == [0, 100]

    # Set mask and then add an XY pulse to the schedule
    seq_xy2 = Sequence(reg, MockDevice)
    seq_xy2.config_slm_mask(targets)
    seq_xy2.declare_channel("ch_xy", "mw_global")
    seq_xy2.add(pulse1, "ch_xy")
    assert seq_xy2._slm_mask_time == [0, 100]

    # Check that adding extra pulses does not change SLM mask time
    seq_xy2.add(pulse2, "ch_xy")
    assert seq_xy2._slm_mask_time == [0, 100]

    # Check that SLM mask time is updated accordingly if a new pulse with
    # earlier start is added
    seq_xy3 = Sequence(reg, MockDevice)
    seq_xy3.declare_channel("ch_xy1", "mw_global")
    seq_xy3.config_slm_mask(targets)
    seq_xy3.delay(duration=100, channel="ch_xy1")
    seq_xy3.add(pulse1, "ch_xy1")
    assert seq_xy3._slm_mask_time == [100, 200]
    seq_xy3.declare_channel("ch_xy2", "mw_global")
    seq_xy3.add(pulse1, "ch_xy2", "no-delay")
    assert seq_xy3._slm_mask_time == [0, 100]

    # Same as previous check, but mask is added afterwards
    seq_xy4 = Sequence(reg, MockDevice)
    seq_xy4.declare_channel("ch_xy1", "mw_global")
    seq_xy4.delay(duration=100, channel="ch_xy1")
    seq_xy4.add(pulse1, "ch_xy1")
    seq_xy4.declare_channel("ch_xy2", "mw_global")
    seq_xy4.add(pulse1, "ch_xy2", "no-delay")
    seq_xy4.config_slm_mask(targets)
    assert seq_xy4._slm_mask_time == [0, 100]

    # Check that paramatrize works with SLM mask
    seq_xy5 = Sequence(reg, MockDevice)
    seq_xy5.declare_channel("ch", "mw_global")
    var = seq_xy5.declare_variable("var")
    seq_xy5.add(Pulse.ConstantPulse(200, var, 0, 0), "ch")
    assert seq_xy5.is_parametrized()
    seq_xy5.config_slm_mask(targets)
    seq_xy5_str = seq_xy5.serialize()
    seq_xy5_ = Sequence.deserialize(seq_xy5_str)
    assert str(seq_xy5) == str(seq_xy5_)

    # Check drawing method
    with patch("matplotlib.pyplot.show"):
        seq_xy2.draw()


def test_draw_register():
    # Draw 2d register from sequence
    reg = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    targets = ["q0", "q2"]
    pulse = Pulse.ConstantPulse(100, 10, 0, 0)
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ch_xy", "mw_global")
    seq.add(pulse, "ch_xy")
    seq.config_slm_mask(targets)
    with patch("matplotlib.pyplot.show"):
        seq.draw(draw_register=True)

    # Draw 3d register from sequence
    reg3d = Register3D.cubic(3, 8)
    seq3d = Sequence(reg3d, MockDevice)
    seq3d.declare_channel("ch_xy", "mw_global")
    seq3d.add(pulse, "ch_xy")
    seq3d.config_slm_mask([6, 15])
    with patch("matplotlib.pyplot.show"):
        seq3d.draw(draw_register=True)
