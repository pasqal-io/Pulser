# Copyright 2022 Pulser Development Team
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

import pytest

from pulser.channels.eom import RydbergBeam, RydbergEOM


@pytest.fixture
def params():
    return dict(
        mod_bandwidth=1,
        limiting_beam=RydbergBeam.RED,
        max_limiting_amp=60,
        intermediate_detuning=700,
        controlled_beams=tuple(RydbergBeam),
    )


@pytest.mark.parametrize(
    "bad_param,bad_value",
    [
        ("mod_bandwidth", 0),
        ("mod_bandwidth", -3),
        ("max_limiting_amp", 0),
        ("intermediate_detuning", -500),
        ("intermediate_detuning", 0),
    ],
)
def test_bad_value_init_eom(bad_param, bad_value, params):
    params[bad_param] = bad_value
    with pytest.raises(
        ValueError, match=f"'{bad_param}' must be greater than zero"
    ):
        RydbergEOM(**params)


@pytest.mark.parametrize(
    "bad_param,bad_value",
    [
        ("limiting_beam", "red"),
        ("limiting_beam", RydbergBeam),
        ("limiting_beam", RydbergBeam.RED | RydbergBeam.BLUE),
        ("controlled_beams", (RydbergBeam.RED | RydbergBeam.BLUE,)),
        ("controlled_beams", (RydbergBeam,)),
    ],
)
def test_bad_init_eom_beam(bad_param, bad_value, params):
    params[bad_param] = bad_value
    with pytest.raises(
        TypeError,
        match="Every beam must be one of options of the `RydbergBeam`",
    ):
        RydbergEOM(**params)


def test_bad_controlled_beam(params):
    params["controlled_beams"] = set(RydbergBeam)
    with pytest.raises(
        TypeError,
        match="The 'controlled_beams' must be provided as a tuple or list.",
    ):
        RydbergEOM(**params)

    params["controlled_beams"] = tuple()
    with pytest.raises(
        ValueError,
        match="There must be at least one beam in 'controlled_beams'",
    ):
        RydbergEOM(**params)

    params["controlled_beams"] = list(RydbergBeam)
    assert RydbergEOM(**params).controlled_beams == tuple(RydbergBeam)


@pytest.mark.parametrize("limit_amp_fraction", [0.5, 2])
def test_detuning_off(limit_amp_fraction, params):
    eom = RydbergEOM(**params)
    limit_amp = params["max_limiting_amp"] ** 2 / (
        2 * params["intermediate_detuning"]
    )
    amp = limit_amp_fraction * limit_amp

    def calc_offset(amp):
        if amp <= limit_amp:
            return 0.0
        assert params["limiting_beam"] == RydbergBeam.RED
        red_amp = params["max_limiting_amp"]
        blue_amp = 2 * params["intermediate_detuning"] * amp / red_amp
        return -(blue_amp**2 - red_amp**2) / (
            4 * params["intermediate_detuning"]
        )

    zero_det = calc_offset(amp)
    assert eom._lightshift(amp, *RydbergBeam) == -zero_det
    assert eom._lightshift(amp) == 0.0
    det_off_options = eom.detuning_off_options(amp, 0.0)
    det_off_options.sort()
    assert det_off_options[0] < zero_det  # RED on
    assert det_off_options[1] == zero_det  # All off
    assert det_off_options[2] > zero_det  # BLUE on

    detuning_on = 1.0
    for beam, ind in [(RydbergBeam.RED, 2), (RydbergBeam.BLUE, 0)]:
        params["controlled_beams"] = (beam,)
        eom_ = RydbergEOM(**params)
        off_options = eom_.detuning_off_options(amp, detuning_on)
        assert len(off_options) == 1
        assert off_options[0] == det_off_options[ind] + detuning_on
