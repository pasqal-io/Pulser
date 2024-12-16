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
import dataclasses

import numpy as np
import pytest

from pulser.channels import Rydberg
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.parametrized import ParamObj, Variable
from pulser.pulse import PHASE_PRECISION, Pulse
from pulser.waveforms import (
    BlackmanWaveform,
    ConstantWaveform,
    CustomWaveform,
    InterpolatedWaveform,
    RampWaveform,
)

cwf = ConstantWaveform(100, -10)
bwf = BlackmanWaveform(200, 3)
rwf = RampWaveform(200, 0, 1)

pls = Pulse(bwf, bwf, 2 * np.pi)
pls2 = Pulse.ConstantPulse(100, 1, -10, -np.pi)
pls3 = Pulse.ConstantAmplitude(1, cwf, -np.pi)
pls4 = Pulse.ConstantDetuning(bwf, -10, 0)


def test_creation():
    with pytest.raises(TypeError):
        Pulse(10, 0, 0, post_phase_shift=2)
        Pulse(cwf, 1, 0)
        Pulse(0, bwf, 1)
        Pulse(bwf, cwf, bwf)
        Pulse(bwf, cwf, 0, post_phase_shift=cwf)

    with pytest.raises(ValueError, match="The duration of"):
        Pulse(bwf, cwf, 0)

    with pytest.raises(ValueError, match="All samples of an amplitude"):
        Pulse(cwf, cwf, 0)
        Pulse.ConstantAmplitude(-1, cwf, 0)
        Pulse.ConstantPulse(100, -1, 0, 0)

    with pytest.raises(TypeError, match="'phase' must be a single float"):
        Pulse(bwf, rwf, [0.0, 1.0, 2.0])

    assert pls.phase == 0
    assert pls2 == pls3
    assert pls != pls4
    assert pls4.detuning != cwf
    assert pls4.amplitude == pls.amplitude


def test_str():
    assert pls2.__str__() == (
        "Pulse(Amp=1 rad/µs, Detuning=-10 rad/µs, " + "Phase=3.14)"
    )
    pls_ = Pulse(bwf, rwf, 1)
    msg = (
        "Pulse(Amp=Blackman(Area: 3) rad/µs, Detuning=Ramp(0->1) rad/µs, "
        "Phase=1)"
    )
    assert pls_.__str__() == msg


def test_repr():
    pls_ = Pulse(bwf, rwf, 1, post_phase_shift=-np.pi)
    msg = (
        "Pulse(amp=BlackmanWaveform(200 ns, Area: 3) rad/µs, "
        + "detuning=RampWaveform(200 ns, 0->1) rad/µs, "
        + "phase=1, post_phase_shift=3.14)"
    )
    assert pls_.__repr__() == msg


def test_draw(patch_plt_show):
    pls_ = Pulse.ConstantDetuning(bwf, -10, 1, post_phase_shift=-np.pi)
    pls_.draw()


@pytest.fixture
def eom_channel():
    eom_config = RydbergEOM(
        mod_bandwidth=24,
        max_limiting_amp=100,
        limiting_beam=RydbergBeam.RED,
        intermediate_detuning=700,
        controlled_beams=tuple(RydbergBeam),
    )
    return Rydberg.Global(None, None, mod_bandwidth=4, eom_config=eom_config)


def test_fall_time(eom_channel):
    assert eom_channel.eom_config.rise_time == 20
    assert eom_channel.rise_time == 120

    pulse = Pulse.ConstantPulse(1000, 1, 0, 0)
    assert pulse.fall_time(eom_channel, in_eom_mode=False) == 240
    assert pulse.fall_time(eom_channel, in_eom_mode=True) == 40


def test_full_duration(eom_channel):
    with pytest.raises(TypeError, match="must be a channel object instance"):
        pls.get_full_duration("eom_channel")

    channel1 = Rydberg.Global(None, None)
    assert not channel1.supports_eom()
    with pytest.raises(
        ValueError, match="does not support EOM mode operation"
    ):
        pls.get_full_duration(channel1, in_eom_mode=True)

    assert pls.get_full_duration(channel1) == pls.duration
    channel2 = dataclasses.replace(channel1, mod_bandwidth=4)
    assert pls.get_full_duration(channel2) == pls.duration + pls.fall_time(
        channel2
    )

    assert pls.get_full_duration(
        eom_channel, in_eom_mode=True
    ) == pls.duration + pls.fall_time(eom_channel, in_eom_mode=True)


@pytest.mark.parametrize(
    "phase_wf, det_wf, phase_0",
    [
        (
            ConstantWaveform(200, -123),
            ConstantWaveform(200, 0),
            -123 % (2 * np.pi),
        ),
        (
            RampWaveform(200, -5, 5),
            ConstantWaveform(200, (_slope := -10 / 199) * 1e3),
            (-5 + _slope) % (2 * np.pi),
        ),
        (
            -bwf,
            CustomWaveform(
                np.pad(np.diff(bwf.samples), (1, 0), mode="edge") * 1e3
            ),
            -bwf[0] + (-bwf[0] + bwf[1]),
        ),
        (
            interp_wf := InterpolatedWaveform(200, values=[1, 3, -2, 4]),
            CustomWaveform(
                np.pad(-np.diff(interp_wf.samples), (1, 0), mode="edge") * 1e3
            ),
            interp_wf[0] + (interp_wf[0] - interp_wf[1]),
        ),
    ],
)
def test_arbitrary_phase(phase_wf, det_wf, phase_0):
    with pytest.raises(TypeError, match="must be a waveform"):
        Pulse.ArbitraryPhase(bwf, -3)

    pls_ = Pulse.ArbitraryPhase(bwf, phase_wf)
    assert pls_ == Pulse(bwf, det_wf, phase_0)

    calculated_phase = -np.cumsum(
        pls_.detuning.samples.as_array() * 1e-3
    ) + float(phase_0)
    phase_samples = phase_wf.samples.as_array()
    assert np.allclose(
        calculated_phase % (2 * np.pi),
        phase_samples % (2 * np.pi),
        atol=PHASE_PRECISION,
        # The shift makes sure we don't fail around the wrapping point
    ) or np.allclose(
        (calculated_phase + 1) % (2 * np.pi),
        (phase_samples + 1) % (2 * np.pi),
        atol=PHASE_PRECISION,
    )


def test_parametrized_pulses():
    vars = Variable("vars", float, size=2)
    vars._assign([1000, 1.0])
    param_bwf = BlackmanWaveform(vars[0], vars[1])
    const_pulse = Pulse.ConstantPulse(vars[0], vars[1], vars[1], vars[1])
    assert isinstance(const_pulse, ParamObj)
    assert const_pulse.cls is Pulse
    param_const = ConstantWaveform(vars[0], vars[1])
    assert (
        const_pulse.build() == Pulse(param_const, param_const, vars[1]).build()
    )
    const_amp = Pulse.ConstantAmplitude(vars[1], param_bwf, vars[1])
    assert const_amp.cls.__name__ == "ConstantAmplitude"
    const_det = Pulse.ConstantDetuning(param_bwf, vars[1], vars[1])
    assert const_det.cls.__name__ == "ConstantDetuning"
    arb_phase = Pulse.ArbitraryPhase(
        param_bwf, RampWaveform(vars[0], 0, vars[1])
    )
    assert arb_phase.cls.__name__ == "ArbitraryPhase"
    special_pulses = [const_amp, const_det, arb_phase]
    for p in special_pulses:
        assert isinstance(p, ParamObj)
        assert p.cls is not Pulse
        assert p.args[0] is Pulse

    assert const_amp.build() == Pulse(param_const, param_bwf, vars[1]).build()
    assert const_det.build() == Pulse(param_bwf, param_const, vars[1]).build()
    assert (
        arb_phase.build()
        == Pulse(
            param_bwf,
            ConstantWaveform(vars[0], -vars[1] * 1e3 / (vars[0] - 1)),
            -vars[1] / (vars[0] - 1),
        ).build()
    )


def test_eq():
    assert (pls_ := Pulse.ConstantPulse(100, 1, -1, 0)) == Pulse(
        ConstantWaveform(100, 1),
        ConstantWaveform(100, -1),
        1e-6,
        post_phase_shift=-1e-6,
    )
    assert pls_ != repr(pls_)


def _assert_pulse_requires_grad(pulse: Pulse, invert: bool = False) -> None:
    assert pulse.amplitude.samples.is_differentiable == (not invert)
    assert pulse.detuning.samples.is_differentiable == (not invert)
    assert pulse.phase.is_differentiable == (not invert)


@pytest.mark.parametrize("requires_grad", [True, False])
def test_pulse_diff(requires_grad, eom_channel, patch_plt_show):
    torch = pytest.importorskip("torch")

    duration = 1000
    diff_val = torch.tensor(1.0, requires_grad=requires_grad)
    constant_wf = ConstantWaveform(duration, diff_val)
    phase = torch.tensor(3.14, requires_grad=requires_grad)
    phase_wf = RampWaveform(
        duration,
        phase - diff_val * 1e-3,
        phase - diff_val * duration * 1e-3,
    )
    assert torch.isclose(torch.tensor(phase_wf.slope), -diff_val * 1e-3)

    pulses: list[Pulse] = [
        Pulse(constant_wf, constant_wf, phase),
        Pulse.ConstantDetuning(constant_wf, diff_val, phase),
        Pulse.ConstantAmplitude(diff_val, constant_wf, phase),
        Pulse.ConstantPulse(constant_wf.duration, diff_val, diff_val, phase),
        Pulse.ArbitraryPhase(constant_wf, phase_wf),
    ]
    for i, pulse in enumerate(pulses):
        _assert_pulse_requires_grad(pulse, invert=not requires_grad)
        # Check other methods still work
        assert pulse.duration == duration
        assert pulse.get_full_duration(
            eom_channel
        ) == duration + pulse.fall_time(eom_channel)

    # Check all pulses are equal (by design)
    for pulse2 in pulses[1:]:
        assert str(pulses[0]) == str(pulse2)
        assert repr(pulses[0]) == repr(pulse2)
        assert pulses[0] == pulse2

    # Extra checks for ArbitraryPhase (since it's more complex)
    bwf = BlackmanWaveform(duration, diff_val)
    phase_pulse = Pulse.ArbitraryPhase(constant_wf, bwf)
    _assert_pulse_requires_grad(phase_pulse, invert=not requires_grad)
