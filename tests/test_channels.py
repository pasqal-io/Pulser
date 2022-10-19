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

import re

import numpy as np
import pytest

import pulser
from pulser.channels import Microwave, Raman, Rydberg
from pulser.waveforms import BlackmanWaveform, ConstantWaveform


@pytest.mark.parametrize(
    "bad_param,bad_value",
    [
        ("max_amp", 0),
        ("max_abs_detuning", -0.001),
        ("clock_period", 0),
        ("min_duration", 0),
        ("max_duration", 0),
        ("mod_bandwidth", 0),
    ],
)
def test_bad_init_global_channel(bad_param, bad_value):
    kwargs = dict(max_abs_detuning=None, max_amp=None)
    kwargs[bad_param] = bad_value
    with pytest.raises(ValueError, match=f"'{bad_param}' must be"):
        Microwave.Global(**kwargs)


@pytest.mark.parametrize(
    "bad_param,bad_value",
    [
        ("max_amp", -0.0001),
        ("max_abs_detuning", -1e6),
        ("min_retarget_interval", -1),
        ("fixed_retarget_t", -1),
        ("max_targets", 0),
        ("clock_period", -4),
        ("min_duration", -2),
        ("max_duration", -1),
        ("mod_bandwidth", -1e4),
    ],
)
def test_bad_init_local_channel(bad_param, bad_value):
    kwargs = dict(max_abs_detuning=None, max_amp=None)
    kwargs[bad_param] = bad_value
    with pytest.raises(ValueError, match=f"'{bad_param}' must be"):
        Rydberg.Local(**kwargs)


def test_bad_durations():
    max_duration, min_duration = 10, 16
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"When defined, 'max_duration'({max_duration}) must be"
            f" greater than or equal to 'min_duration'({min_duration})."
        ),
    ):
        Rydberg.Global(
            None, None, min_duration=min_duration, max_duration=max_duration
        )


@pytest.mark.parametrize(
    "field",
    [
        "min_retarget_interval",
        "fixed_retarget_t",
        "max_targets",
    ],
)
def test_bad_none_fields(field):
    with pytest.raises(
        TypeError, match=f"'{field}' can't be None in a 'Local' channel."
    ):
        Raman.Local(None, None, **{field: None})


@pytest.mark.parametrize("max_amp", [1, None])
@pytest.mark.parametrize("max_abs_detuning", [0, None])
@pytest.mark.parametrize("max_duration", [1000, None])
def test_virtual_channel(max_amp, max_abs_detuning, max_duration):
    params = (max_amp, max_abs_detuning, max_duration)
    assert Raman.Global(
        max_amp=max_amp,
        max_abs_detuning=max_abs_detuning,
        max_duration=max_duration,
    ).is_virtual() == (None in params)


def test_device_channels():
    for dev in pulser.devices._valid_devices:
        for i, (id, ch) in enumerate(dev.channels.items()):
            assert id == dev._channels[i][0]
            assert isinstance(id, str)
            assert ch == dev._channels[i][1]
            assert isinstance(ch, pulser.channels.Channel)
            assert ch.name in ["Rydberg", "Raman"]
            assert ch.basis in ["digital", "ground-rydberg"]
            assert ch.addressing in ["Local", "Global"]
            assert ch.max_abs_detuning >= 0
            assert ch.max_amp > 0
            assert ch.clock_period >= 1
            assert ch.min_duration >= 1
            if ch.addressing == "Local":
                assert ch.min_retarget_interval >= 0
                assert ch.min_retarget_interval == int(
                    ch.min_retarget_interval
                )
                assert ch.max_targets >= 1
                assert ch.max_targets == int(ch.max_targets)


def test_validate_duration():
    ch = Rydberg.Local(20, 10, min_duration=16, max_duration=1000)
    with pytest.raises(TypeError, match="castable to an int"):
        ch.validate_duration("twenty")
    with pytest.raises(ValueError, match="at least 16 ns"):
        ch.validate_duration(10)
    with pytest.raises(ValueError, match="at most 1000 ns"):
        ch.validate_duration(1e5)
    with pytest.warns(UserWarning, match="not a multiple"):
        ch.validate_duration(31.4)


def test_repr():
    raman = Raman.Local(
        None,
        2,
        min_retarget_interval=1000,
        fixed_retarget_t=200,
        max_targets=4,
    )
    r1 = (
        "Raman.Local(Max Absolute Detuning: None, Max Amplitude: "
        "2 rad/µs, Phase Jump Time: 0 ns, Minimum retarget time: 1000 ns, "
        "Fixed retarget time: 200 ns, Max targets: 4, Clock period: 4 ns, "
        "Minimum pulse duration: 16 ns, Maximum pulse duration: 1000000 ns, "
        "Basis: 'digital')"
    )
    assert raman.__str__() == r1

    ryd = Rydberg.Global(50, None, phase_jump_time=300, mod_bandwidth=4)
    r2 = (
        "Rydberg.Global(Max Absolute Detuning: 50 rad/µs, "
        "Max Amplitude: None, Phase Jump Time: 300 ns, "
        "Clock period: 4 ns, Minimum pulse duration: 16 ns, "
        "Maximum pulse duration: 1000000 ns, "
        "Modulation Bandwidth: 4 MHz, Basis: 'ground-rydberg')"
    )
    assert ryd.__str__() == r2


def test_modulation():
    rydberg_global = Rydberg.Global(2 * np.pi * 20, 2 * np.pi * 2.5)

    raman_local = Raman.Local(
        2 * np.pi * 20,
        2 * np.pi * 10,
        mod_bandwidth=4,  # MHz
    )

    wf = ConstantWaveform(100, 1)
    assert rydberg_global.mod_bandwidth is None
    with pytest.warns(UserWarning, match="No modulation bandwidth defined"):
        out_samples = rydberg_global.modulate(wf.samples)
    assert np.all(out_samples == wf.samples)

    with pytest.raises(TypeError, match="doesn't have a modulation bandwidth"):
        rydberg_global.calc_modulation_buffer(wf.samples, out_samples)

    out_ = raman_local.modulate(wf.samples)
    tr = raman_local.rise_time
    assert len(out_) == wf.duration + 2 * tr
    assert raman_local.calc_modulation_buffer(wf.samples, out_) == (tr, tr)

    wf2 = BlackmanWaveform(800, np.pi)
    side_buffer_len = 45
    out_ = raman_local.modulate(wf2.samples)
    assert len(out_) == wf2.duration + 2 * tr  # modulate() does not truncate
    assert raman_local.calc_modulation_buffer(wf2.samples, out_) == (
        side_buffer_len,
        side_buffer_len,
    )
