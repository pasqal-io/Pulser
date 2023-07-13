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

from pulser.channels.eom import MODBW_TO_TR, RydbergBeam, RydbergEOM


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
        ("mod_bandwidth", MODBW_TO_TR * 1e3 + 1),
        ("max_limiting_amp", 0),
        ("intermediate_detuning", -500),
        ("intermediate_detuning", 0),
        ("custom_buffer_time", 0.1),
        ("custom_buffer_time", 0),
    ],
)
def test_bad_value_init_eom(bad_param, bad_value, params):
    params[bad_param] = bad_value
    if bad_param == "mod_bandwidth" and bad_value > 0:
        error_type = NotImplementedError
        error_message = (
            f"'mod_bandwidth' must be lower than {MODBW_TO_TR*1e3} MHz"
        )
    else:
        error_type = ValueError
        error_message = f"'{bad_param}' must be greater than zero"
    with pytest.raises(error_type, match=error_message):
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


@pytest.mark.parametrize("multiple_beam_control", [True, False])
@pytest.mark.parametrize("limit_amp_fraction", [0.5, 2])
def test_detuning_off(multiple_beam_control, limit_amp_fraction, params):
    params["multiple_beam_control"] = multiple_beam_control
    eom = RydbergEOM(**params)
    limit_amp = params["max_limiting_amp"] ** 2 / (
        2 * params["intermediate_detuning"]
    )
    amp = limit_amp_fraction * limit_amp

    def calc_offset(amp):
        # Manually calculates the offset needed to correct the lightshift
        # coming from a difference in power between the beams
        if amp <= limit_amp:
            # Below limit_amp, red_amp=blue_amp so there is no lightshift
            return 0.0
        assert params["limiting_beam"] == RydbergBeam.RED
        red_amp = params["max_limiting_amp"]
        blue_amp = 2 * params["intermediate_detuning"] * amp / red_amp
        # The offset to have resonance when the pulse is on is -lightshift
        return -(blue_amp**2 - red_amp**2) / (
            4 * params["intermediate_detuning"]
        )

    # Case where the EOM pulses are resonant
    detuning_on = 0.0
    zero_det = calc_offset(amp)  # detuning when both beams are off = offset
    assert eom._lightshift(amp, *RydbergBeam) == -zero_det
    assert eom._lightshift(amp) == 0.0
    det_off_options = eom.detuning_off_options(amp, detuning_on)
    assert len(det_off_options) == 2 + multiple_beam_control
    det_off_options.sort()
    assert det_off_options[0] < zero_det  # RED on
    next_ = 1
    if multiple_beam_control:
        assert det_off_options[next_] == zero_det  # All off
        next_ += 1
    assert det_off_options[next_] > zero_det  # BLUE on

    # Case where the EOM pulses are off-resonant
    detuning_on = 1.0
    for beam, ind in [(RydbergBeam.RED, next_), (RydbergBeam.BLUE, 0)]:
        # When only one beam is controlled, there is a single
        # detuning_off option
        params["controlled_beams"] = (beam,)
        eom_ = RydbergEOM(**params)
        off_options = eom_.detuning_off_options(amp, detuning_on)
        assert len(off_options) == 1
        # The new detuning_off is shifted by the new detuning_on,
        # since that changes the offset compared the resonant case
        assert off_options[0] == det_off_options[ind] + detuning_on
