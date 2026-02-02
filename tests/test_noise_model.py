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

import contextlib
import dataclasses
import re
import warnings

import numpy as np
import pytest
import qutip

from pulser.noise_model import (
    _NOISE_TYPE_PARAMS,
    _PARAM_TO_NOISE_TYPE,
    NoiseModel,
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
            (set(), set()),
            ({"disable_doppler"}, set()),
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
                {
                    "temperature",
                    "depolarizing_rate",
                    "runs",
                    "samples_per_run",
                    "disable_doppler",
                },
                {"depolarizing"},
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
            (
                {
                    "temperature",
                    "trap_waist",
                    "trap_depth",
                    "runs",
                    "samples_per_run",
                    "disable_doppler",
                },
                {"register"},
            ),
        ],
    )
    def test_init(self, params, noise_types):

        with (
            pytest.deprecated_call(match="NoiseModel.runs")
            if "runs" in params
            else contextlib.nullcontext()
        ):
            noise_model = NoiseModel(
                **{
                    p: (1.0 if p != "disable_doppler" else True)
                    for p in params
                }
            )
        assert set(noise_model.noise_types) == noise_types
        relevant_params = NoiseModel._find_relevant_params(
            noise_types,
            noise_model.state_prep_error,
            noise_model.amp_sigma,
            noise_model.laser_waist,
        )
        assert "disable_doppler" not in relevant_params
        assert noise_model.disable_doppler == ("disable_doppler" in params)
        params.discard("disable_doppler")
        assert all(getattr(noise_model, p) == 1.0 for p in params)
        assert all(
            not getattr(noise_model, p) for p in relevant_params - params
        )

    @pytest.mark.parametrize(
        "noise_param", ["relaxation_rate", "p_false_neg", "laser_waist"]
    )
    @pytest.mark.parametrize("unused_param", ["runs", "samples_per_run"])
    @pytest.mark.filterwarnings(
        "ignore:.*'NoiseModel.runs' is deprecated:DeprecationWarning"
    )
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
    @pytest.mark.filterwarnings(
        "ignore:.*'NoiseModel.runs' is deprecated:DeprecationWarning"
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

    @pytest.mark.filterwarnings(
        "ignore:.*'NoiseModel.runs' is deprecated:DeprecationWarning"
    )
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
        noise_model = NoiseModel(disable_doppler=value)
        assert noise_model.disable_doppler == value

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
        with pytest.raises(
            ValueError,
            match=f"'disable_doppler' must be a boolean, not {value}",
        ):
            NoiseModel(disable_doppler=value)

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
        id_nested_tuple = ((1.0, 0.0), (0.0, 1.0))
        assert NoiseModel(
            eff_noise_opers=[matrices["I"]],
            eff_noise_rates=[1.0],
        ).eff_noise_opers == (id_nested_tuple,)
        assert NoiseModel(
            eff_noise_opers=[matrices["I"].tolist()],
            eff_noise_rates=[1.0],
        ).eff_noise_opers == (id_nested_tuple,)
        assert NoiseModel(
            eff_noise_opers=[qutip.Qobj(matrices["I"])],
            eff_noise_rates=[1.0],
        ).eff_noise_opers == (id_nested_tuple,)

    def test_eff_noise_opers_torch(self, matrices):
        torch = pytest.importorskip("torch")
        assert NoiseModel(
            eff_noise_opers=[torch.from_numpy(matrices["I"])],
            eff_noise_rates=[1.0],
        ).eff_noise_opers == (((1.0, 0.0), (0.0, 1.0)),)

    def test_hf_detuning_noise_validation(self):
        # list - expected format
        noise_mod = NoiseModel(
            detuning_hf_psd=[1, 4, 2], detuning_hf_omegas=[3, 6, 7]
        )
        # np.array - other expected format
        noise_mod = NoiseModel(
            detuning_hf_psd=np.array([1, 4, 2]),
            detuning_hf_omegas=np.array([3, 6, 7]),
        )
        # tuple - other expected format
        noise_mod = NoiseModel(
            detuning_hf_psd=(1, 4, 2),
            detuning_hf_omegas=(3, 6, 7),
        )

        # not provided psd and freqs
        noise_mod = NoiseModel()
        assert (
            noise_mod.detuning_hf_psd == ()
            and noise_mod.detuning_hf_omegas == ()
        )

        # only psd are provided
        with pytest.raises(
            ValueError, match=("empty tuples or both be provided")
        ):
            NoiseModel(detuning_hf_psd=(1, 2, 3))

        # only freqs are provided
        with pytest.raises(
            ValueError, match=("empty tuples or both be provided")
        ):
            NoiseModel(detuning_hf_omegas=(4, 5, 6))

        # psd dim != 1
        with pytest.raises(ValueError, match=("1D tuples")):
            NoiseModel(
                detuning_hf_psd=[[1, 2, 3]], detuning_hf_omegas=[3, 4, 5]
            )

        # freqs dim != 1
        with pytest.raises(ValueError, match=("1D tuples")):
            NoiseModel(
                detuning_hf_psd=[1, 2, 3], detuning_hf_omegas=[[3, 4, 5]]
            )

        # len psd != len freqs
        with pytest.raises(ValueError, match=("same length")):
            NoiseModel(detuning_hf_psd=[1, 2], detuning_hf_omegas=[3, 4, 5])

        # psd len <= 1
        with pytest.raises(ValueError, match=("length > 1")):
            NoiseModel(detuning_hf_psd=[1], detuning_hf_omegas=[3])

        # psd < 0
        with pytest.raises(ValueError, match=("positive values")):
            NoiseModel(detuning_hf_psd=[-1, 2], detuning_hf_omegas=[3, 4])

        # freqs < 0
        with pytest.raises(ValueError, match=("positive values")):
            NoiseModel(detuning_hf_psd=[1, 2], detuning_hf_omegas=[3, -4])

        # freqs should monotonously grow
        with pytest.raises(ValueError, match=("monotonously growing")):
            NoiseModel(detuning_hf_psd=[1, 2], detuning_hf_omegas=[4, 3])

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
            {"register", "doppler"},
            0.0,
            0.0,
            None,
        ) == {
            "temperature",
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
        ) == {
            "detuning_sigma",
            "detuning_hf_psd",
            "detuning_hf_omegas",
            "runs",
            "samples_per_run",
        }

    @pytest.mark.filterwarnings(
        "ignore:.*'NoiseModel.runs' is deprecated:DeprecationWarning"
    )
    def test_repr(self):
        assert repr(NoiseModel()) == "NoiseModel(noise_types=())"
        assert (
            repr(NoiseModel(temperature=1.0, runs=10, disable_doppler=True))
            == "NoiseModel(noise_types=())"
        )
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
            == "NoiseModel(noise_types=('amplitude',), amp_sigma=0.3)"
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
            == "NoiseModel(noise_types=('doppler', 'register'), "
            "temperature=15.0, trap_waist=1.0, trap_depth=150.0)"
        )
        assert (
            repr(
                NoiseModel(
                    temperature=15.0,
                    trap_depth=150.0,  # same units as temperature
                    trap_waist=1.0,
                    runs=None,
                    samples_per_run=1,
                    disable_doppler=True,
                )
            )
            == "NoiseModel(noise_types=('register',), "
            "temperature=15.0, trap_waist=1.0, trap_depth=150.0)"
        )


def test_register_noise_no_warning_when_all_params_defined():
    """Register noise with all parameters.

    Doing this also defines doppler noise.
    """
    noise_model = NoiseModel(
        temperature=15.0,
        trap_waist=1.0,
        trap_depth=150.0,  # the same units as temperature
    )
    assert noise_model.noise_types == ("doppler", "register")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        noise_model._check_register_noise_params(
            noise_model.noise_types,
            noise_model.trap_waist,
            noise_model.trap_depth,
            noise_model.temperature,
        )
        assert (
            len(rec) == 0
        ), f"Expected no warnings,got: {[r.message for r in rec]}"


def test_trap_param_default_and_temperature_set():
    """Behavior of default trap parameters in presence of temperature.

    Doppler noise is activated when temperature is set but trap parameters
    are not.
    """
    noise_model = NoiseModel(
        trap_waist=0.0,  # default
        trap_depth=None,  # default
        temperature=10.0,
    )
    assert noise_model.noise_types == ("doppler",)


def test_check_register_noise_params_invalid_params():
    """Gives a ValueError!

    if trap_waist == 0.0 or trap_depth is None or temperature == 0.0.
    """
    with pytest.raises(
        ValueError, match="defined in order to simulate register noise"
    ):
        _ = NoiseModel(
            trap_depth=150.0,
            trap_waist=0.0,
            temperature=10.0,
        )
    with pytest.raises(
        ValueError, match="defined in order to simulate register noise"
    ):
        _ = NoiseModel(
            trap_waist=2.0,
            trap_depth=150,
            temperature=0.0,
        )


def test_noise_table_summary():
    # Start with a NoiseModel with only register noise
    noise_model = NoiseModel(
        temperature=10,
        trap_depth=1.0,
        trap_waist=1.0,
        disable_doppler=True,
    )
    noise_table = {
        "register_sigma_xy": (0.0015811388300841897, "µm"),
        "register_sigma_z": (0.008264487918871443, "µm"),
    }
    assert noise_model.get_noise_table() == noise_table
    summary = (
        "Noise summary:\n"
        + "- Register Position Fluctuations**:\n"
        + "  - XY-Plane Position Fluctuations: 0.00158114 µm\n"
        + "  - Z-Axis Position Fluctuations: 0.00826449 µm\n"
    )
    end_summary = (
        "**: Emulation will generate EmulationConfig.n_trajectories"
        " trajectories with different register"
    )
    assert summary + end_summary == noise_model.summary()
    # Include doppler noise
    noise_model = NoiseModel(temperature=10, trap_depth=1.0, trap_waist=1.0)
    noise_table["doppler_sigma"] = (0.2683952309561405, "rad/µs")
    assert noise_model.get_noise_table() == noise_table
    detuning_summary = (
        "- Detuning fluctuations**:\n"
        + "  - Shot-to-Shot Detuning fluctuations:\n"
    )
    doppler_summary = "       - Doppler fluctuations: 0.268395 rad/µs\n"
    assert (
        noise_model.summary()
        == summary
        + detuning_summary
        + doppler_summary
        + end_summary
        + ", detuning"
    )
    # Include detuning shot to shot
    noise_model = NoiseModel(
        temperature=10, trap_depth=1.0, trap_waist=1.0, detuning_sigma=1.0
    )
    noise_table["detuning_sigma"] = (1.0, "rad/µs")
    assert noise_model.get_noise_table() == noise_table
    detuning_summary += (
        "       - Laser's Detuning fluctuations: 1 rad/µs\n" + doppler_summary
    )
    assert (
        noise_model.summary()
        == summary + detuning_summary + end_summary + ", detuning"
    )
    # Include state preparation error
    noise_model = dataclasses.replace(noise_model, state_prep_error=0.1)
    noise_table["state_prep_error"] = (0.1, "")
    assert noise_model.get_noise_table() == noise_table
    summary += "- State Preparation Error Probability**: 0.1\n"
    end_summary += ", initial state"
    assert (
        noise_model.summary()
        == summary + detuning_summary + end_summary + ", detuning"
    )
    # Include amplitude fluctuations noise
    noise_model = dataclasses.replace(
        noise_model, amp_sigma=0.1, laser_waist=100
    )
    noise_table["amp_sigma"] = (10.0, "%")
    noise_table["laser_waist"] = (100, "µm")
    assert noise_model.get_noise_table() == noise_table
    summary += (
        "- Amplitude inhomogeneities:\n"
        + "  - Finite-waist Gaussian damping σ=100 µm\n"
        + "  - Shot-to-shot Amplitude Fluctuations**: 10 %\n"
        + detuning_summary
    )
    end_summary += ", amplitude, detuning"
    assert noise_model.summary() == summary + end_summary
    # Include PSD noise
    noise_model = dataclasses.replace(
        noise_model, detuning_hf_omegas=[1.0, 2.0], detuning_hf_psd=[1.0, 0.5]
    )
    noise_table["detuning_psd"] = (
        [(1.0, 1.0), (2.0, 0.5)],
        "(rad/µs, rad/µs)",
    )
    assert noise_table == noise_model.get_noise_table()
    summary += (
        "  - High-Frequency Detuning fluctuations. See PSD "
        "in get_noise_table()['detuning_psd'].\n"
    )
    assert noise_model.summary() == summary + end_summary
    # Include relaxation and dephasing rates
    noise_model = dataclasses.replace(
        noise_model, relaxation_rate=0.1, dephasing_rate=0.5
    )
    noise_table["T1"] = (10.0, "µs")
    noise_table["T2* (r-g)"] = (2.0, "µs")
    assert noise_model.get_noise_table() == noise_table
    summary += (
        "- Dissipation parameters:\n"
        + "   - T1: 10 µs\n"
        + "   - T2* (r-g): 2 µs\n"
    )
    assert noise_model.summary() == summary + end_summary
    # Include hyperfine dephasing rate
    noise_model = dataclasses.replace(
        noise_model, hyperfine_dephasing_rate=0.25
    )
    noise_table["T2* (g-h)"] = (4.0, "µs")
    assert noise_model.get_noise_table() == noise_table
    summary += "   - T2* (g-h): 4 µs\n"
    assert noise_model.summary() == summary + end_summary
    # Include depolarizing rate
    noise_model = dataclasses.replace(noise_model, depolarizing_rate=0.2)
    noise_table["depolarizing_rate"] = (0.2, "1/µs")
    assert noise_model.get_noise_table() == noise_table
    summary += (
        "- Other Decoherence Processes:\n"
        + "   - Depolarization at rate 0.2 1/µs\n"
    )
    assert noise_model.summary() == summary + end_summary
    # Include eff noise
    noise_model = dataclasses.replace(
        noise_model,
        eff_noise_rates=(1.0,),
        eff_noise_opers=(((1.0, 0.0), (0.0, -1.0)),),
    )
    noise_table["eff_noise"] = (
        [(1.0, ((1.0, 0.0), (0.0, -1.0)))],
        "(1/µs, '')",
    )
    noise_table["with_leakage"] = (False, "")
    assert noise_model.get_noise_table() == noise_table
    assert (
        noise_model.summary()
        == summary
        + "   - Custom Lindblad operators (in 1/µs):\n"
        + "       - 1 * ((1.0, 0.0), (0.0, -1.0))\n"
        + end_summary
    )
    # With leakage
    new_oper_tuple = ((1.0, 0.0, 1 / 3), (0.0, -1.0, 2.0), (0.0, 2.0, 1.0))
    new_oper = np.array(new_oper_tuple)  # with an array
    noise_model = dataclasses.replace(
        noise_model,
        with_leakage=True,
        eff_noise_opers=(new_oper,),
    )
    noise_table["eff_noise"] = ([(1.0, new_oper_tuple)], "(1/µs, '')")
    noise_table["with_leakage"] = (True, "")
    assert noise_model.get_noise_table() == noise_table
    summary += (
        "   - Custom Lindblad operators (in 1/µs) including a leakage state:\n"
        "       - 1 * ((1.0, 0.0, 0.333333), (0.0, -1.0, 2.0), (0.0, 2.0, 1.0)"
        ")\n"
    )
    assert noise_model.summary() == summary + end_summary
    # Add measurement errors
    noise_model = dataclasses.replace(
        noise_model, p_false_pos=0.5, p_false_neg=0.2
    )
    noise_table["p_false_neg"] = (0.2, "")
    noise_table["p_false_pos"] = (0.5, "")
    assert noise_model.get_noise_table() == noise_table
    summary += (
        "- Measurement noises:\n"
        + "   - False Positive Meas. Probability: 0.5\n"
        + "   - False Negative Meas. Probability: 0.2\n"
    )
    assert noise_model.summary() == summary + end_summary
    # Without repetition
    noise_model = NoiseModel(relaxation_rate=0.2)
    assert noise_model.get_noise_table() == {"T1": (5.0, "µs")}
    assert (
        noise_model.summary()
        == "Noise summary:\n- Dissipation parameters:\n   - T1: 5 µs"
    )
