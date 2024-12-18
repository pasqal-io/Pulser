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
import re

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator, interp1d

from pulser.channels import Rydberg
from pulser.json.coders import PulserDecoder, PulserEncoder
from pulser.parametrized import ParamObj, Variable
from pulser.waveforms import (
    BlackmanWaveform,
    CompositeWaveform,
    ConstantWaveform,
    CustomWaveform,
    InterpolatedWaveform,
    KaiserWaveform,
    RampWaveform,
)

np.random.seed(20201105)

constant = ConstantWaveform(100, -3)
ramp = RampWaveform(2000, 5, 19)
arb_samples = np.random.random(52)
custom = CustomWaveform(arb_samples)
blackman = BlackmanWaveform(40, np.pi)
composite = CompositeWaveform(blackman, constant, custom)
interp_values = [0, 1, 4.4, 2, 3, 1, 0]
interp = InterpolatedWaveform(1000, interp_values)
kaiser = KaiserWaveform(40, np.pi)


def test_duration():
    with pytest.raises(TypeError, match="needs to be castable to int"):
        ConstantWaveform("s", -1)
        RampWaveform([0, 1, 3], 1, 0)

    with pytest.raises(ValueError, match="positive duration"):
        ConstantWaveform(15, -10)
        RampWaveform(-20, 3, 4)

    with pytest.warns(UserWarning):
        wf = BlackmanWaveform(np.pi * 10, 1)

    assert wf.duration == 31
    assert custom.duration == 52
    assert composite.duration == 192


def test_change_duration():
    with pytest.raises(NotImplementedError):
        custom.change_duration(53)

    new_cte = constant.change_duration(103)
    assert constant.duration == 100
    assert new_cte.duration == 103

    new_blackman = blackman.change_duration(30)
    assert np.isclose(new_blackman.integral, blackman.integral)
    assert new_blackman != blackman

    new_ramp = ramp.change_duration(100)
    assert new_ramp.duration == 100
    assert new_ramp != ramp

    assert interp.duration == 1000
    new_interp = interp.change_duration(100)
    assert new_interp.duration == 100


def test_samples():
    assert np.all(constant.samples.as_array() == -3)
    bm_samples = np.clip(np.blackman(40), 0, np.inf)
    bm_samples *= np.pi / np.sum(bm_samples) / 1e-3
    comp_samples = np.concatenate([bm_samples, np.full(100, -3), arb_samples])
    assert np.all(np.isclose(composite.samples.as_array(), comp_samples))


def test_integral():
    assert np.isclose(blackman.integral, np.pi)
    assert constant.integral == -0.3
    assert ramp.integral == 24


def test_draw(patch_plt_show):
    rydberg_global = Rydberg.Global(
        2 * np.pi * 20,
        2 * np.pi * 2.5,
        mod_bandwidth=4,  # MHz
    )

    composite.draw()
    blackman.draw(output_channel=rydberg_global)
    interp.draw(output_channel=rydberg_global)


def test_eq():
    assert constant == CustomWaveform(np.full(100, -3))
    assert constant != -3
    assert constant != CustomWaveform(np.full(48, -3))


def test_first_last():
    assert constant.first_value == constant.last_value
    assert ramp.first_value == 5
    assert ramp.last_value == 19
    assert blackman.first_value == 0
    assert blackman.last_value == 0
    assert composite.first_value == 0
    assert composite.last_value == arb_samples[-1]
    assert custom.first_value == arb_samples[0]
    assert np.isclose(interp.first_value, interp_values[0])
    assert np.isclose(interp.last_value, interp_values[-1])


def test_hash():
    assert hash(constant) == hash(tuple(np.full(100, -3)))
    assert hash(ramp) == hash(tuple(np.linspace(5, 19, num=2000)))


def test_composite():
    with pytest.raises(ValueError, match="Needs at least two waveforms"):
        CompositeWaveform()
        CompositeWaveform(composite)
        CompositeWaveform([blackman, custom])
        CompositeWaveform(10)

    with pytest.raises(TypeError, match="not a valid waveform"):
        CompositeWaveform(composite, "constant")

    assert composite.waveforms == [blackman, constant, custom]

    wf = CompositeWaveform(blackman, constant)
    msg = (
        "BlackmanWaveform(40 ns, Area: 3.14), "
        + "ConstantWaveform(100 ns, -3)"
    )
    assert wf.__str__() == f"Composite({msg})"
    assert wf.__repr__() == f"CompositeWaveform(140 ns, [{msg}])"


def test_custom():
    data = np.arange(16, dtype=float)
    wf = CustomWaveform(data)
    assert wf.__str__() == "Custom"
    assert wf.__repr__() == f"CustomWaveform(16 ns, {data!r})"


def test_ramp():
    assert np.isclose(ramp.slope, 7e-3, atol=1e-5)

    ramp_samples = RampWaveform(
        3000, top := 25.757450291031688, 0
    ).samples.as_array()
    assert np.all(np.logical_and(ramp_samples <= top, ramp_samples >= 0))


def test_blackman():
    with pytest.raises(TypeError):
        BlackmanWaveform(100, np.array([1, 2]))
    wf = BlackmanWaveform(100, -2)
    assert np.isclose(wf.integral, -2)
    assert np.all(wf.samples <= 0)
    assert wf == BlackmanWaveform(100, np.array(-2))

    with pytest.raises(ValueError, match="matching signs"):
        BlackmanWaveform.from_max_val(-10, np.pi)

    wf = BlackmanWaveform.from_max_val(10, 2 * np.pi)
    assert np.isclose(wf.integral, 2 * np.pi)
    assert np.max(wf.samples) < 10

    wf = BlackmanWaveform.from_max_val(-10, -np.pi)
    assert np.isclose(wf.integral, -np.pi)
    assert np.min(wf.samples) > -10

    var = Variable("var", float)
    wf_var = BlackmanWaveform.from_max_val(-10, var[0])
    assert isinstance(wf_var, ParamObj)
    var._assign(-np.pi)
    assert wf_var.build() == wf

    # Check moving back to previous even duration
    area: float = np.pi / 6
    max_val: float = 46
    wf: BlackmanWaveform = BlackmanWaveform.from_max_val(max_val, area)
    duration = wf.duration
    assert duration % 2 == 0
    wf2 = BlackmanWaveform(duration + 1, area)
    assert np.max(wf2.samples) < np.max(wf.samples) <= max_val

    # Same but with a negative max value
    wf: BlackmanWaveform = BlackmanWaveform.from_max_val(-max_val, -area)
    duration = wf.duration
    assert duration % 2 == 0
    wf2 = BlackmanWaveform(duration + 1, -area)
    assert np.min(wf2.samples) > np.min(wf.samples) >= -max_val


def test_interpolated():
    assert isinstance(interp.interp_function, PchipInterpolator)

    times = np.linspace(0.2, 0.8, num=len(interp_values))
    with pytest.raises(ValueError, match="must match the number of `values`"):
        InterpolatedWaveform(1000, interp_values, times=times[:-1])
    with pytest.raises(ValueError, match="must be greater than or equal to 0"):
        InterpolatedWaveform(1000, interp_values, times=times - 0.21)
    with pytest.raises(ValueError, match="must be less than or equal to 1"):
        InterpolatedWaveform(1000, interp_values, times=times + 0.21)
    with pytest.raises(ValueError, match="array of non-repeating values"):
        InterpolatedWaveform(
            1000, interp_values, times=[0.2] + times[:-1].tolist()
        )

    with pytest.raises(ValueError, match="Invalid interpolator 'fake'"):
        InterpolatedWaveform(
            1000, interp_values, times=times, interpolator="fake"
        )

    dt = 1000
    interp_wf = InterpolatedWaveform(
        dt, [0, 1], interpolator="interp1d", kind="linear"
    )
    assert isinstance(interp_wf.interp_function, interp1d)
    np.testing.assert_allclose(
        interp_wf.samples.as_array(), np.linspace(0, 1.0, num=dt)
    )

    interp_wf *= 2
    np.testing.assert_allclose(
        interp_wf.samples.as_array(), np.linspace(0, 2.0, num=dt)
    )

    wf_str = "InterpolatedWaveform(Points: (0, 0), (999, 2)"
    assert str(interp_wf) == wf_str + ")"
    assert repr(interp_wf) == wf_str + ", Interpolator=interp1d)"

    vals = np.linspace(0, 1, num=5) ** 2
    interp_wf2 = InterpolatedWaveform(
        dt, vals, interpolator="interp1d", kind="quadratic"
    )
    np.testing.assert_allclose(
        interp_wf2.samples.as_array(),
        np.linspace(0, 1, num=dt) ** 2,
        atol=1e-3,
    )

    # Test rounding when range of values is large
    wf = InterpolatedWaveform(
        1000, times=[0.0, 0.5, 1.0], values=[0, 2.6e7, 0]
    )
    assert np.all((wf.samples >= 0).as_array())


def test_kaiser():
    duration: int = 40
    area: float = np.pi
    beta: float = 14.0

    wf: KaiserWaveform = KaiserWaveform(duration, area, beta)
    wf_samples = wf.samples.as_array()

    # Check type error on area
    with pytest.raises(TypeError):
        KaiserWaveform(duration, np.array([1, 2]))

    # Check type error on beta
    with pytest.raises(TypeError):
        KaiserWaveform(duration, area, beta=np.array([1, 2]))

    # Check beta must not be negative
    with pytest.raises(ValueError, match="must be greater than 0"):
        KaiserWaveform(duration, area, -1.0)

    # Check duration
    assert wf.duration == duration
    assert wf.samples.size == duration

    # Check default beta
    wf_default_beta = KaiserWaveform(duration, area)
    kaiser_beta_14: np.ndarray = np.kaiser(duration, 14.0)
    kaiser_beta_14 *= area / float(np.sum(kaiser_beta_14)) / 1e-3
    np.testing.assert_allclose(
        wf_default_beta.samples.as_array(), kaiser_beta_14, atol=1e-3
    )

    # Check area
    assert np.isclose(np.sum(wf_samples), area * 1000.0)

    # Check duration change
    new_duration = duration * 2
    wf_change_duration = wf.change_duration(new_duration)
    assert wf_change_duration.samples.size == new_duration
    assert np.isclose(
        np.sum(wf_samples), np.sum(wf_change_duration.samples.as_array())
    )

    # Check __str__
    assert str(wf) == (
        f"Kaiser({duration} ns, Area: {area:.3g}, Beta: {beta:.3g})"
    )

    # Check __repr__
    assert repr(wf) == (
        f"KaiserWaveform(duration: {duration}, "
        f"area: {area:.3g}, beta: {beta:.3g})"
    )

    # Check multiplication
    wf_multiplication = wf * 2
    assert np.all(wf_multiplication.samples == wf_samples * 2)

    # Check area and max_val must have matching signs
    with pytest.raises(ValueError, match="must have matching signs"):
        KaiserWaveform.from_max_val(1, -1)

    # Test from_max_val
    for max_val in range(1, 501, 50):
        for beta in range(1, 20):
            wf = KaiserWaveform.from_max_val(max_val, area, beta)
            assert np.isclose(np.sum(wf.samples.as_array()), area * 1000.0)
            assert np.max(wf.samples.as_array()) <= max_val
            wf = KaiserWaveform.from_max_val(-max_val, -area, beta)
            assert np.isclose(np.sum(wf.samples.as_array()), -area * 1000.0)
            assert np.min(wf.samples.as_array()) >= -max_val


def test_ops():
    assert -constant == ConstantWaveform(100, 3)
    assert ramp * 2 == RampWaveform(2e3, 10, 38)
    assert --custom == custom
    assert blackman / 2 == BlackmanWaveform(40, np.pi / 2)
    assert composite * 1 == composite
    with pytest.raises(ZeroDivisionError):
        constant / 0


def test_serialization():
    for wf in [blackman, composite, constant, custom, interp, kaiser, ramp]:
        s = json.dumps(wf, cls=PulserEncoder)
        assert wf == json.loads(s, cls=PulserDecoder)


def test_get_item():
    # Check errors raised

    duration = constant.duration
    with pytest.raises(
        IndexError,
        match=re.escape(
            "Index ('index_or_slice' = "
            f"{duration}) must be in the range "
            f"0~{duration-1}, or "
            f"{-duration}~-1 from the end."
        ),
    ):
        constant[duration]
    with pytest.raises(
        IndexError,
        match=re.escape(
            "Index ('index_or_slice' = "
            f"{-duration-1}) must be in the range "
            f"0~{duration-1}, or "
            f"{-duration}~-1 from the end."
        ),
    ):
        constant[-duration - 1]

    with pytest.raises(
        IndexError, match="The step of the slice must be None or 1."
    ):
        constant[0:1:2]

    # Check nominal operations

    for wf in [blackman, composite, constant, custom, kaiser, ramp, interp]:
        duration = wf.duration
        duration14 = duration // 4
        duration34 = duration * 3 // 4
        samples = wf.samples

        # Check with int index
        for i in range(-duration, duration):
            assert wf[i] == samples[i]

        # Check with slices

        assert np.all(wf[0:duration] == samples)
        assert np.all(wf[0:-1] == samples[0:-1])
        assert np.all(wf[0:] == samples)
        assert np.all(wf[-1:] == samples[-1:])
        assert np.all(wf[:duration] == samples)
        assert np.all(wf[:] == samples)
        assert np.all(
            wf[duration14:duration34] == samples[duration14:duration34]
        )
        assert np.all(
            wf[-duration34:-duration14] == samples[-duration34:-duration14]
        )

        # Check with out of bounds slices
        assert np.all(wf[: duration * 2] == samples)
        assert np.all(wf[-duration * 2 :] == samples)
        assert np.all(wf[-duration * 2 : duration * 2] == samples)
        assert np.all(
            wf[duration // 2 : duration * 2]
            == samples[duration // 2 : duration * 2]
        )
        assert np.all(
            wf[-duration * 2 : duration // 2]
            == samples[-duration * 2 : duration // 2]
        )
        assert wf[2:1].size == 0
        assert wf[duration * 2 :].size == 0
        assert wf[duration * 2 : duration * 3].size == 0
        assert wf[-duration * 3 : -duration * 2].size == 0


@pytest.fixture
def rydberg_global():
    return Rydberg.Global(
        2 * np.pi * 20,
        2 * np.pi * 2.5,
        mod_bandwidth=4,  # MHz
    )


def test_modulation(rydberg_global):
    mod_samples = constant.modulated_samples(rydberg_global).as_array()
    assert np.all(mod_samples == rydberg_global.modulate(constant.samples))
    assert constant.modulation_buffers(rydberg_global) == (
        rydberg_global.rise_time,
        rydberg_global.rise_time,
    )
    assert len(mod_samples) == constant.duration + 2 * rydberg_global.rise_time
    assert np.isclose(np.sum(mod_samples) * 1e-3, constant.integral)
    assert max(np.abs(mod_samples)) < np.abs(constant[0])


@pytest.mark.parametrize(
    "wf_type, diff_param_name, diff_param_value, extra_params",
    [
        (CustomWaveform, "samples", np.arange(-10.0, 10.0), {}),
        (ConstantWaveform, "value", -3.14, {"duration": 20}),
        (RampWaveform, "start", -10.0, {"duration": 10, "stop": 10}),
        (RampWaveform, "stop", -10.0, {"duration": 10, "start": 10}),
        (BlackmanWaveform, "area", 2.0, {"duration": 200}),
        (BlackmanWaveform.from_max_val, "area", -2.0, {"max_val": -1}),
        (KaiserWaveform, "area", -2.0, {"duration": 200}),
        (KaiserWaveform.from_max_val, "area", 2.0, {"max_val": 1}),
    ],
)
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("composite", [True, False])
def test_waveform_diff(
    wf_type,
    diff_param_name,
    diff_param_value,
    extra_params,
    requires_grad,
    composite,
    rydberg_global,
    patch_plt_show,
):
    torch = pytest.importorskip("torch")
    kwargs = {
        diff_param_name: torch.tensor(
            diff_param_value, requires_grad=requires_grad
        ),
        **extra_params,
    }
    wf = wf_type(**kwargs)
    if composite:
        wf = CompositeWaveform(wf, ConstantWaveform(100, 1.0))

    samples_tensor = wf.samples.as_tensor()
    assert samples_tensor.requires_grad == requires_grad
    assert wf.modulated_samples(rydberg_global).requires_grad == requires_grad
    wfx2_tensor = (-wf * 2).samples.as_tensor()
    assert torch.equal(wfx2_tensor, samples_tensor * -2.0)
    assert wfx2_tensor.requires_grad == requires_grad

    wfdiv2 = wf / torch.tensor(2.0, requires_grad=True)
    assert torch.equal(wfdiv2.samples.as_tensor(), samples_tensor / 2.0)
    # Should always be true because it was divided by diff tensor
    assert wfdiv2.samples.requires_grad

    assert wf[-1].requires_grad == requires_grad

    try:
        assert wf.change_duration(1000).samples.requires_grad == requires_grad
    except NotImplementedError:
        pass

    # Check that all non-related methods still work
    wf.draw(output_channel=rydberg_global)
    repr(wf)
    str(wf)
    hash(wf)
    wf._to_dict()
    wf._to_abstract_repr()
    assert isinstance(wf.integral, float)
