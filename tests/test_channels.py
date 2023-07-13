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
from dataclasses import replace
from operator import gt

import numpy as np
import pytest

import pulser
from pulser.channels import Microwave, Raman, Rydberg
from pulser.channels.comparison_tools import (
    _compare_with_None,
    validate_channel_from_best,
    validate_eom_from_best,
)
from pulser.channels.eom import MODBW_TO_TR, BaseEOM, RydbergBeam, RydbergEOM
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
        ("mod_bandwidth", MODBW_TO_TR * 1e3 + 1),
    ],
)
def test_bad_init_global_channel(bad_param, bad_value):
    kwargs = dict(max_abs_detuning=None, max_amp=None)
    kwargs[bad_param] = bad_value
    if bad_param == "mod_bandwidth" and bad_value > 1:
        error_type = NotImplementedError
    else:
        error_type = ValueError
    with pytest.raises(error_type, match=f"'{bad_param}' must be"):
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
        ("mod_bandwidth", MODBW_TO_TR * 1e3 + 1),
    ],
)
def test_bad_init_local_channel(bad_param, bad_value):
    kwargs = dict(max_abs_detuning=None, max_amp=None)
    kwargs[bad_param] = bad_value
    if bad_param == "mod_bandwidth" and bad_value > 1:
        error_type = NotImplementedError
    else:
        error_type = ValueError
    with pytest.raises(error_type, match=f"'{bad_param}' must be"):
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
@pytest.mark.parametrize("max_targets", [1, None])
def test_virtual_channel(max_amp, max_abs_detuning, max_duration, max_targets):
    params = (max_amp, max_abs_detuning, max_duration, max_targets)
    assert Raman.Local(
        max_amp=max_amp,
        max_abs_detuning=max_abs_detuning,
        max_duration=max_duration,
        max_targets=max_targets,
    ).is_virtual() == (None in params)


def test_device_channels():
    for dev in pulser.devices._valid_devices:
        for i, (id, ch) in enumerate(dev.channels.items()):
            assert id == dev.channel_ids[i]
            assert isinstance(id, str)
            assert ch == dev.channel_objects[i]
            assert isinstance(ch, pulser.channels.channels.Channel)
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
        min_duration=16,
        clock_period=4,
        max_duration=None,
    )
    r1 = (
        "Raman.Local(Max Absolute Detuning: None, Max Amplitude: "
        "2 rad/µs, Minimum retarget time: 1000 ns, "
        "Fixed retarget time: 200 ns, Max targets: 4, Clock period: 4 ns, "
        "Minimum pulse duration: 16 ns, Basis: 'digital')"
    )
    assert raman.__str__() == r1

    ryd = Rydberg.Global(50, None, mod_bandwidth=4)
    r2 = (
        "Rydberg.Global(Max Absolute Detuning: 50 rad/µs, "
        "Max Amplitude: None, Clock period: 1 ns, "
        "Minimum pulse duration: 1 ns, "
        "Maximum pulse duration: 100000000 ns, "
        "Modulation Bandwidth: 4 MHz, Basis: 'ground-rydberg')"
    )
    assert ryd.__str__() == r2


_eom_config = RydbergEOM(
    mod_bandwidth=20,
    limiting_beam=RydbergBeam.RED,
    max_limiting_amp=100 * 2 * np.pi,
    intermediate_detuning=500 * 2 * np.pi,
    controlled_beams=tuple(RydbergBeam),
)


def test_eom_channel():
    with pytest.raises(
        ValueError,
        match="'eom_config' can't be defined in a Channel without a"
        " modulation bandwidth",
    ):
        Rydberg.Global(None, None, eom_config=_eom_config)

    with pytest.raises(
        TypeError,
        match="When defined, 'eom_config' must be a valid 'RydbergEOM'",
    ):
        Rydberg.Global(None, None, mod_bandwidth=3, eom_config=BaseEOM(50))

    assert not Rydberg.Global(None, None).supports_eom()
    assert Rydberg.Global(
        None, None, mod_bandwidth=3, eom_config=_eom_config
    ).supports_eom()


def test_modulation_errors():
    wf = ConstantWaveform(100, 1)
    no_eom_msg = "The channel Rydberg.Global(.*) does not have an EOM."
    with pytest.raises(TypeError, match=no_eom_msg):
        Rydberg.Global(None, None, mod_bandwidth=10).modulate(
            wf.samples, eom=True
        )

    with pytest.raises(TypeError, match=no_eom_msg):
        Rydberg.Global(None, None, mod_bandwidth=10).calc_modulation_buffer(
            wf.samples, wf.samples, eom=True
        )

    rydberg_global = Rydberg.Global(2 * np.pi * 20, 2 * np.pi * 2.5)
    assert rydberg_global.mod_bandwidth is None
    with pytest.warns(UserWarning, match="No modulation bandwidth defined"):
        out_samples = rydberg_global.modulate(wf.samples)
    assert np.all(out_samples == wf.samples)

    with pytest.raises(TypeError, match="doesn't have a modulation bandwidth"):
        rydberg_global.calc_modulation_buffer(wf.samples, out_samples)


_raman_local = Raman.Local(
    2 * np.pi * 20,
    2 * np.pi * 10,
    mod_bandwidth=4,  # MHz
)
_eom_rydberg = Rydberg.Global(
    max_amp=2 * np.pi * 10,
    max_abs_detuning=2 * np.pi * 5,
    mod_bandwidth=10,
    eom_config=_eom_config,
)


@pytest.mark.parametrize(
    "channel, tr, eom, side_buffer_len",
    [
        (_raman_local, _raman_local.rise_time, False, 45),
        (_eom_rydberg, _eom_config.rise_time, True, 0),
    ],
)
def test_modulation(channel, tr, eom, side_buffer_len):
    wf = ConstantWaveform(100, 1)
    out_ = channel.modulate(wf.samples, eom=eom)
    assert len(out_) == wf.duration + 2 * tr
    assert channel.calc_modulation_buffer(wf.samples, out_, eom=eom) == (
        tr,
        tr,
    )

    wf2 = BlackmanWaveform(800, np.pi)
    out_ = channel.modulate(wf2.samples, eom=eom)
    assert len(out_) == wf2.duration + 2 * tr  # modulate() does not truncate
    assert channel.calc_modulation_buffer(wf2.samples, out_, eom=eom) == (
        side_buffer_len,
        side_buffer_len,
    )


def test_compare_with_None():
    comp_ops = {"max_amp": gt, "max_duration": gt}
    left_values = {"max_amp": 1, "clock_period": 1, "max_duration": None}
    right_values = {"max_amp": None, "clock_period": 2}
    with pytest.raises(
        ValueError,
        match="Keys in comparison_ops should be in left values and right",
    ):
        _compare_with_None(comp_ops, left_values, right_values)
    right_values["max_duration"] = None
    comp_ops["clock_period"] = gt
    assert _compare_with_None(comp_ops, left_values, right_values) == {
        "max_amp": True,
        "clock_period": False,
        "max_duration": None,
    }
    assert _compare_with_None(comp_ops, right_values, left_values) == {
        "max_amp": False,
        "clock_period": True,
        "max_duration": None,
    }


@pytest.mark.parametrize(
    "param, value, comparison",
    [
        ("mod_bandwidth", 10, None),
        ("mod_bandwidth", 20, None),
        ("mod_bandwidth", 40, "above"),
        ("limiting_beam", RydbergBeam.RED, None),
        ("limiting_beam", RydbergBeam.BLUE, "different"),
        ("max_limiting_amp", 2 * np.pi * 100, None),
        ("max_limiting_amp", 2 * np.pi * 150, "above"),
        ("controlled_beams", (RydbergBeam.RED,), None),
        ("controlled_beams", tuple(RydbergBeam), "above"),
        ("intermediate_detuning", 500 * 2 * np.pi, None),
        ("intermediate_detuning", 200 * 2 * np.pi, "different"),
    ],
)
def test_compare_eom(param, value, comparison):
    print(param, value)
    best_eom_config = replace(_eom_config, controlled_beams=(RydbergBeam.RED,))

    with pytest.raises(
        ValueError,
        match="EOM config is RydbergEOM whereas best EOM config is not.",
    ):
        validate_eom_from_best(best_eom_config, BaseEOM(20))

    eom_config = replace(best_eom_config, **{param: value})
    if comparison is None:
        assert validate_eom_from_best(eom_config, best_eom_config)
        if param == "mod_bandwidth":
            # Works because BaseEOM can be implemented by RydbergEOM
            validate_eom_from_best(BaseEOM(value), best_eom_config)
    else:
        with pytest.raises(
            ValueError, match=f"{param} cannot be {comparison} "
        ):
            validate_eom_from_best(eom_config, best_eom_config)
        if param == "mod_bandwidth":
            with pytest.raises(
                ValueError, match=f"{param} cannot be {comparison} "
            ):
                validate_eom_from_best(BaseEOM(value), best_eom_config)


@pytest.mark.parametrize(
    "param, value, comparison",
    [
        ("max_abs_detuning", 300, "above"),
        ("max_amp", 100, "above"),
        ("mod_bandwidth", 10, "above"),
        ("clock_period", 2, "below"),
        ("max_duration", 2e8, "above"),
        ("min_duration", 1, None),
        ("max_targets", 4, "above"),
        ("fixed_retarget_t", 0, None),
        ("min_retarget_interval", 180, "below"),
    ],
)
def test_compare_channels(mod_device, param, value, comparison):
    # Error if channel types are different
    with pytest.raises(
        ValueError,
        match=f"Channels do not have the same types, {Raman} and {Rydberg}",
    ):
        validate_channel_from_best(_raman_local, _eom_rydberg)
    # Error if the best channel does not have an EOM
    rydberg_no_eom = Rydberg.Global(
        max_amp=2 * np.pi * 10,
        max_abs_detuning=2 * np.pi * 5,
        mod_bandwidth=10,
    )
    with pytest.raises(
        ValueError, match="eom_config cannot be defined in channel"
    ):
        validate_channel_from_best(_eom_rydberg, rydberg_no_eom)
    assert validate_eom_from_best(rydberg_no_eom, _eom_rydberg)
    # Error if addressing is not the same
    best_channel = mod_device.channels["rydberg_local"]
    with pytest.raises(
        ValueError, match="addressing cannot be different than Local."
    ):
        validate_channel_from_best(rydberg_no_eom, best_channel)
    # Changing parameters of a Local channel.
    channel = replace(best_channel, **{param: value})
    if comparison is None:
        assert validate_channel_from_best(channel, best_channel)
    else:
        with pytest.raises(
            ValueError, match=f"{param} cannot be {comparison} "
        ):
            validate_channel_from_best(channel, best_channel)
