# Copyright 2024 Pulser Development Team
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
from __future__ import annotations

import re
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulser.noise_model import (
    _NOISE_TYPE_PARAMS,
    _PARAM_TO_NOISE_TYPE,
    NoiseModel,
    _noisy_register,
)


def test_constants():
    # Recreate _PARAM_TO_NOISE_TYPE and check it matches
    params_dict = {}
    for noise_type, params in _NOISE_TYPE_PARAMS.items():
        for p in params:
            assert p not in params_dict
            params_dict[p] = noise_type
    assert params_dict == _PARAM_TO_NOISE_TYPE


class TestNoiseModel:

    @pytest.mark.parametrize(
        "params, noise_types",
        [
            ({"p_false_pos", "dephasing_rate"}, {"SPAM", "dephasing"}),
            (
                {
                    "state_prep_error",
                    "relaxation_rate",
                    "runs",
                    "samples_per_run",
                },
                {"SPAM", "relaxation"},
            ),
            (
                {
                    "temperature",
                    "depolarizing_rate",
                    "runs",
                    "samples_per_run",
                },
                {"doppler", "depolarizing"},
            ),
            (
                {"amp_sigma", "runs", "samples_per_run"},
                {"amplitude"},
            ),
            (
                {"laser_waist", "hyperfine_dephasing_rate"},
                {"amplitude", "dephasing"},
            ),
            ({"detuning_sigma", "runs", "samples_per_run"}, {"detuning"}),
            (
                {
                    "temperature",
                    "trap_waist",
                    "trap_depth",
                    "runs",
                    "samples_per_run",
                },
                {"doppler", "register"},
            ),
        ],
    )
    def test_init(self, params, noise_types):
        noise_model = NoiseModel(**{p: 1.0 for p in params})
        assert set(noise_model.noise_types) == noise_types
        relevant_params = NoiseModel._find_relevant_params(
            noise_types,
            noise_model.state_prep_error,
            noise_model.amp_sigma,
            noise_model.laser_waist,
        )
        assert all(getattr(noise_model, p) == 1.0 for p in params)
        assert all(
            not getattr(noise_model, p) for p in relevant_params - params
        )

    @pytest.mark.parametrize(
        "noise_param", ["relaxation_rate", "p_false_neg", "laser_waist"]
    )
    @pytest.mark.parametrize("unused_param", ["runs", "samples_per_run"])
    def test_unused_params(self, unused_param, noise_param):
        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"'{unused_param}' is not used by any active noise type in"
                f" {(_PARAM_TO_NOISE_TYPE[noise_param],)} when the only "
                f"defined parameters are {[noise_param]}"
            ),
        ):
            if unused_param == "samples_per_run":
                with pytest.deprecated_call(
                    match="Setting samples_per_run different to 1 is"
                ):
                    NoiseModel(**{unused_param: 100, noise_param: 1.0})
            else:
                NoiseModel(**{unused_param: 100, noise_param: 1.0})

    @pytest.mark.parametrize(
        "param",
        ["runs", "samples_per_run", "laser_waist"],
    )
    def test_init_strict_pos(self, param):
        with pytest.raises(
            ValueError, match=f"'{param}' must be greater than zero, not 0"
        ):
            NoiseModel(**{param: 0})

    @pytest.mark.parametrize("value", [None, -1e-9, 0.0, 0.2, 1.0001])
    @pytest.mark.parametrize(
        "param, noise",
        [
            ("dephasing_rate", "dephasing"),
            ("hyperfine_dephasing_rate", "dephasing"),
            ("relaxation_rate", "relaxation"),
            ("depolarizing_rate", "depolarizing"),
            ("temperature", "doppler"),
            ("detuning_sigma", "detuning"),
        ],
    )
    def test_init_rate_like(self, param, noise, value):
        kwargs = {param: value}
        if (
            param == "temperature" or param == "detuning_sigma"
        ) and value != 0:
            kwargs.update(dict(runs=1, samples_per_run=1))
        if value is None:
            with pytest.raises(
                TypeError,
                match=f"{param} should be castable to float, not",
            ):
                NoiseModel(**kwargs)
        elif value < 0:
            with pytest.raises(
                ValueError,
                match=f"'{param}' must be greater than "
                f"or equal to zero, not {value}.",
            ):
                NoiseModel(**kwargs)
        else:
            noise_model = NoiseModel(**kwargs)
            assert getattr(noise_model, param) == value
            if value > 0:
                assert noise_model.noise_types == (noise,)
            else:
                assert noise_model.noise_types == ()

    @pytest.mark.parametrize("value", [-1e-9, 0.0, 0.5, 1.0, 1.0001])
    @pytest.mark.parametrize(
        "param, noise",
        [
            ("state_prep_error", "SPAM"),
            ("p_false_pos", "SPAM"),
            ("p_false_neg", "SPAM"),
            ("amp_sigma", "amplitude"),
        ],
    )
    def test_init_prob_like(self, param, noise, value):
        if 0 <= value <= 1:
            kwargs = {param: value}
            if value > 0 and param in ("amp_sigma", "state_prep_error"):
                kwargs.update(dict(runs=1, samples_per_run=1))
            noise_model = NoiseModel(**kwargs)
            assert getattr(noise_model, param) == value
            if value > 0:
                assert noise_model.noise_types == (noise,)
            else:
                assert noise_model.noise_types == ()
            return
        with pytest.raises(
            ValueError,
            match=f"'{param}' must be greater than or equal to zero and "
            f"smaller than or equal to one, not {value}",
        ):
            NoiseModel(
                # Define the strict positive quantities first so that their
                # absence doesn't trigger their own errors
                runs=1,
                samples_per_run=1,
                **{param: value},
            )

    @pytest.fixture
    def matrices(self):
        matrices = {}
        matrices["I"] = np.eye(2)
        matrices["X"] = np.ones((2, 2)) - np.eye(2)
        matrices["Y"] = np.array([[0, -1j], [1j, 0]])
        matrices["Zh"] = 0.5 * np.array([[1, 0], [0, -1]])
        matrices["ket"] = np.array([[1.0], [2.0]])
        matrices["I3"] = np.eye(3)
        matrices["I4"] = np.eye(4)
        return matrices

    @pytest.mark.parametrize("value", [False, True])
    def test_init_bool_like(self, value, matrices):
        noise_model = NoiseModel(
            eff_noise_rates=[0.1],
            eff_noise_opers=[matrices["I3"] if value else matrices["I"]],
            with_leakage=value,
        )
        assert noise_model.with_leakage == value

    @pytest.mark.parametrize("value", [0, 1, 0.1])
    def test_wrong_init_bool_like(self, value, matrices):
        with pytest.raises(
            ValueError, match=f"'with_leakage' must be a boolean, not {value}"
        ):
            NoiseModel(
                eff_noise_rates=[0.1],
                eff_noise_opers=[matrices["I3"] if value else matrices["I"]],
                with_leakage=value,
            )

    def test_eff_noise_rates(self, matrices):
        with pytest.raises(
            ValueError, match="The provided rates must be greater than 0."
        ):
            NoiseModel(
                eff_noise_opers=[matrices["I"], matrices["X"]],
                eff_noise_rates=[-1.0, 0.5],
            )

    def test_eff_noise_opers(self, matrices):
        with pytest.raises(ValueError, match="The operators list length"):
            NoiseModel(eff_noise_rates=[1.0])
        with pytest.raises(
            TypeError, match="eff_noise_rates is a list of floats"
        ):
            NoiseModel(
                eff_noise_rates=["0.1"],
                eff_noise_opers=[np.eye(2)],
            )
        with pytest.raises(TypeError, match="not castable to a Numpy array"):
            NoiseModel(
                eff_noise_rates=[2.0],
                eff_noise_opers=[{(1.0, 0), (0.0, -1)}],
            )
        with pytest.raises(ValueError, match="is not a 2D array."):
            NoiseModel(
                eff_noise_opers=[2.0],
                eff_noise_rates=[1.0],
            )
        with pytest.raises(ValueError, match="With leakage, operator's shape"):
            NoiseModel(
                eff_noise_opers=[matrices["I"]],
                eff_noise_rates=[1.0],
                with_leakage=True,
            )
        with pytest.raises(ValueError, match="With leakage, operator's shape"):
            NoiseModel(
                eff_noise_opers=[np.eye(5)],
                eff_noise_rates=[1.0],
                with_leakage=True,
            )
        with pytest.raises(
            ValueError, match="Without leakage, operator's shape"
        ):
            NoiseModel(
                eff_noise_opers=[matrices["I4"]],
                eff_noise_rates=[1.0],
            )

    @pytest.mark.parametrize("param", ["dephasing_rate", "depolarizing_rate"])
    def test_leakage(self, param):
        with pytest.raises(
            ValueError, match="At least one effective noise operator"
        ):
            NoiseModel(with_leakage=True)

    def test_eq(self, matrices):
        final_fields = dict(
            p_false_pos=0.1,
            eff_noise_rates=(0.1, 0.4),
            eff_noise_opers=(((0, 1), (1, 0)), ((0, -1j), (1j, 0))),
        )
        noise_model = NoiseModel(
            p_false_pos=0.1,
            eff_noise_rates=[0.1, 0.4],
            eff_noise_opers=[matrices["X"], matrices["Y"]],
        )
        assert noise_model == NoiseModel(**final_fields)
        assert set(noise_model.noise_types) == {"SPAM", "eff_noise"}
        for param in final_fields:
            assert final_fields[param] == getattr(noise_model, param)

    def test_relevant_params(self):
        assert NoiseModel._find_relevant_params(
            {"SPAM"},
            0.0,
            0.5,
            100,
        ) == {
            "state_prep_error",
            "p_false_pos",
            "p_false_neg",
        }
        assert NoiseModel._find_relevant_params(
            {"SPAM"},
            0.1,
            0.5,
            100,
        ) == {
            "state_prep_error",
            "p_false_pos",
            "p_false_neg",
            "runs",
            "samples_per_run",
        }

        assert NoiseModel._find_relevant_params(
            {"register"}, 0.0, 0.0, None
        ) == {
            "trap_waist",
            "trap_depth",
            "runs",
            "samples_per_run",
        }

        assert NoiseModel._find_relevant_params(
            {"doppler"},
            0.0,
            0.0,
            None,
        ) == {"temperature", "runs", "samples_per_run"}
        assert NoiseModel._find_relevant_params(
            {"amplitude"},
            0.0,
            1.0,
            None,
        ) == {"amp_sigma", "runs", "samples_per_run"}
        assert NoiseModel._find_relevant_params(
            {"amplitude"},
            0.0,
            0.0,
            100.0,
        ) == {"amp_sigma", "laser_waist"}
        assert NoiseModel._find_relevant_params(
            {"amplitude"},
            0.0,
            0.5,
            100.0,
        ) == {"amp_sigma", "laser_waist", "runs", "samples_per_run"}
        assert NoiseModel._find_relevant_params(
            {"dephasing", "leakage"},
            0.0,
            0.0,
            None,
        ) == {"dephasing_rate", "hyperfine_dephasing_rate", "with_leakage"}
        assert NoiseModel._find_relevant_params(
            {"relaxation", "leakage"},
            0.0,
            0.0,
            None,
        ) == {"relaxation_rate", "with_leakage"}
        assert NoiseModel._find_relevant_params(
            {"depolarizing", "leakage"},
            0.0,
            0.0,
            None,
        ) == {"depolarizing_rate", "with_leakage"}
        assert NoiseModel._find_relevant_params(
            {"eff_noise", "leakage"},
            0.0,
            0.0,
            None,
        ) == {"eff_noise_rates", "eff_noise_opers", "with_leakage"}
        assert NoiseModel._find_relevant_params(
            {"detuning"},
            0.0,
            0.0,
            None,
        ) == {"detuning_sigma", "runs", "samples_per_run"}

    def test_repr(self):
        assert repr(NoiseModel()) == "NoiseModel(noise_types=())"
        assert (
            repr(NoiseModel(p_false_pos=0.1, relaxation_rate=0.2))
            == "NoiseModel(noise_types=('SPAM', 'relaxation'), "
            "state_prep_error=0.0, p_false_pos=0.1, p_false_neg=0.0, "
            "relaxation_rate=0.2)"
        )
        assert (
            repr(NoiseModel(hyperfine_dephasing_rate=0.2))
            == "NoiseModel(noise_types=('dephasing',), "
            "dephasing_rate=0.0, hyperfine_dephasing_rate=0.2)"
        )
        assert (
            repr(NoiseModel(amp_sigma=0.3, runs=100, samples_per_run=1))
            == "NoiseModel(noise_types=('amplitude',), "
            "runs=100, samples_per_run=1, amp_sigma=0.3)"
        )
        assert (
            repr(NoiseModel(laser_waist=100.0))
            == "NoiseModel(noise_types=('amplitude',), "
            "laser_waist=100.0, amp_sigma=0.0)"
        )
        assert (
            repr(
                NoiseModel(
                    hyperfine_dephasing_rate=0.2,
                    eff_noise_opers=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                    eff_noise_rates=[0.1],
                    with_leakage=True,
                )
            )
            == "NoiseModel(noise_types=('dephasing', 'eff_noise', 'leakage'), "
            "dephasing_rate=0.0, hyperfine_dephasing_rate=0.2, "
            "eff_noise_rates=(0.1,), eff_noise_opers=(((1, 0, 0), (0, 1, 0), "
            "(0, 0, 1)),), with_leakage=True)"
        )

        assert (
            repr(
                NoiseModel(
                    temperature=15.0,
                    trap_depth=150.0,  # same units as temperature
                    trap_waist=1.0,
                    runs=1,
                    samples_per_run=1,
                )
            )
            == "NoiseModel(noise_types=('doppler', 'register'), runs=1, "
            "samples_per_run=1, temperature=15.0, trap_waist=1.0, "
            "trap_depth=150.0)"
        )


@pytest.mark.parametrize(
    "register2D",
    [
        True,
        False,
    ],
)
def test_noisy_register(register2D) -> None:
    """Testing noisy_register function in case 2D and 3D register."""
    noise_model = MagicMock()
    noise_model._register_sigma_xy_z.return_value = (0.13, 0.8)

    if register2D:
        qdict = {
            "q0": np.array([-15.0, 0.0]),
            "q1": np.array([-5.0, 0.0]),
            "q2": np.array([5.0, 0.0]),
            "q3": np.array([15.0, 0.0]),
        }
    else:
        qdict = {
            "q0": np.array([-15.0, 0.0, 0.0]),
            "q1": np.array([-5.0, 0.0, 0.0]),
            "q2": np.array([5.0, 0.0, 0.0]),
            "q3": np.array([15.0, 0.0, 0.0]),
        }

    # Predefined deterministic noise
    fake_normal_xy_noise = np.array(
        [
            [0.1, -0.1],
            [0.2, -0.2],
            [0.3, -0.3],
            [0.5, -0.5],
        ]
    )
    fake_normal_z_noise = np.array([0.05, 0.07, 0.09, 0.11])
    with patch("numpy.random.normal") as mock_normal:
        # moke the noise generation
        mock_normal.side_effect = [
            fake_normal_xy_noise,
            fake_normal_z_noise,
        ]
        result = _noisy_register(qdict, noise_model)
    expected_positions = {
        "q0": np.array([-15.0 + 0.1, 0.0 - 0.1, 0.05]),
        "q1": np.array([-5.0 + 0.2, 0.0 - 0.2, 0.07]),
        "q2": np.array([5.0 + 0.3, 0.0 - 0.3, 0.09]),
        "q3": np.array([15.0 + 0.5, 0.0 - 0.5, 0.11]),
    }
    for q in qdict:
        np.testing.assert_array_almost_equal(result[q], expected_positions[q])


def test_register_noise_sigma_xy_z_raises_if_trap_depth_is_none():
    """Trap_depth is None."""
    with pytest.raises(
        ValueError, match="'trap_depth' must be greater than zero, not None"
    ):
        noise_model = NoiseModel(
            temperature=1.0,
            trap_depth=None,
            trap_waist=1.0,
            runs=1,
            samples_per_run=1,
        )
        noise_model._register_sigma_xy_z()


def test_register_noise_no_warning_when_all_params_defined():
    """Register noise with all parameters.

    Doing this also defines doppler noise.
    """
    noise_model = NoiseModel(
        temperature=15.0,
        trap_waist=1.0,
        trap_depth=150.0,  # the same units as temperature
        runs=1,
        samples_per_run=1,
    )
    assert noise_model.noise_types == ("doppler", "register")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        noise_model._check_register_and_doppler_noise_params(
            noise_model.noise_types
        )
        assert (
            len(rec) == 0
        ), f"Expected no warnings,got: {[r.message for r in rec]}"


def test_register_not_activated_warns_when_temperature_zero():
    """Trap parameters are turned on, but temperature=0.

    Warning: register noise not activated.
    """
    with pytest.warns(UserWarning, match=r"Register noise is not activated"):
        noise_model = NoiseModel(
            temperature=0.0,
            trap_waist=1.0,
            trap_depth=150.0,
            runs=1,
            samples_per_run=1,
        )
        assert "register" in noise_model.noise_types


def test_register_only_doppler_warns_when_trap_params_missing():
    """trap_waist == 0.0, Warning: Only doppler noise will be used."""
    with pytest.warns(UserWarning, match=r"Only doppler noise will be used"):
        noise_model = NoiseModel(
            temperature=15.0,
            trap_waist=0.0,
            trap_depth=150.0,
            runs=1,
            samples_per_run=1,
        )
        assert ("doppler", "register") == noise_model.noise_types
