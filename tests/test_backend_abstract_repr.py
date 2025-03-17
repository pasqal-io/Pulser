import json

import numpy as np
import pytest
from pytest import mark

from pulser.backend import (
    BitStrings,
    CorrelationMatrix,
    EmulationConfig,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Fidelity,
    Occupation,
    StateResult,
)
from pulser.backend.state import StateRepr

# TODO: decide where to put these tests


@mark.parametrize(
    "observable, expected_kwargs",
    [
        (
            StateResult,
            {
                "evaluation_times": [0.1, 0.3, 1.0],
            },
        ),
        (
            BitStrings,
            {
                "evaluation_times": [i * 0.05 for i in range(10)],
                "num_shots": 211,
                "one_state": "r",
                "tag_suffix": "7",
            },
        ),
        (
            CorrelationMatrix,
            {"one_state": "r"},
        ),
        (
            Occupation,
            {"one_state": "g"},
        ),
        (
            Energy,
            {"evaluation_times": [i * 0.05 for i in range(10)]},
        ),
        (
            EnergyVariance,
            {"evaluation_times": np.linspace(0, 1, 13)},
        ),
        (
            EnergySecondMoment,
            {"evaluation_times": [i * 0.1 for i in range(5)]},
        ),
    ],
)
def test_observable_repr(observable, expected_kwargs):
    obs = observable(**expected_kwargs)
    obs_repr = obs._to_abstract_repr()
    assert isinstance(obs_repr, dict)

    # test default values
    assert obs_repr["observable"] == obs._base_tag
    if "evaluation_times" not in expected_kwargs.keys():
        assert obs_repr["evaluation_times"] is None
    if "tag_suffix" not in expected_kwargs.keys():
        assert obs_repr["tag_suffix"] is None

    # test kwargs
    for key, expected_value in expected_kwargs.items():
        assert obs_repr[key] is expected_value


def test_fidelity_repr():
    evaluation_times = [0.1, 0.3, 0.9]
    kwargs = {"evaluation_times": evaluation_times}

    basis = {"r", "g"}
    amplitudes = {"rgr": 1.0, "grg": 1.0}
    state = StateRepr(basis, amplitudes)
    fidelity = Fidelity(state, **kwargs)

    fidelity_repr = fidelity._to_abstract_repr()

    state_repr = fidelity_repr["state"]
    assert state_repr is state


def test_config_repr():
    evaluation_times = [0.1, 0.3, 0.9]
    bitstrings = BitStrings(evaluation_times=evaluation_times)
    correlation = CorrelationMatrix()
    observables = (bitstrings, correlation)
    with_modulation = True
    default_evaluation_times = "Full"
    expected_kwargs = {
        "observables": observables,
        "with_modulation": with_modulation,
        "default_evaluation_times": default_evaluation_times,
    }

    config = EmulationConfig(**expected_kwargs)
    # dump with AbstrctReprEncoder
    config_str = config.to_abstract_repr()
    # load and redump but with default JSON encoder
    # equivalent to go key by key and check single str repr
    config_load_dump_str = json.dumps(json.loads(config_str))

    assert config_str == config_load_dump_str


class TestStateRepr:
    def test_state_repr(self):
        basis = {"r", "g"}
        amplitudes = {"rgr": 1.0j + 0.2, "grg": 1.0}
        expected_repr = {"eigenstates": basis, "amplitudes": amplitudes}
        state = StateRepr(basis, amplitudes)
        state_repr = state._to_abstract_repr()
        assert state_repr == expected_repr

    def test_state_repr_invalid_eigenstates(self):
        basis = {"av", "b", "c"}
        amplitudes = {"rgr": 1.0, "grg": 1.0}
        with pytest.raises(
            ValueError,
            match="All eigenstates must be represented by single characters.",
        ):
            StateRepr(basis, amplitudes)

    def test_state_repr_not_implemented(self):
        basis = {"r", "g"}
        amplitudes = {"rgr": 1.0, "grg": 1.0}
        state = StateRepr(eigenstates=basis, amplitudes=amplitudes)
        with pytest.raises(NotImplementedError):
            state.from_state_amplitudes(
                eigenstates=basis, amplitudes=amplitudes
            )
        with pytest.raises(NotImplementedError):
            state.n_qudits
        with pytest.raises(NotImplementedError):
            state.overlap(state)
        with pytest.raises(NotImplementedError):
            state.sample(num_shots=10)
