import json
from unittest.mock import MagicMock

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
    Expectation,
    Fidelity,
    Occupation,
    StateResult,
)
from pulser.backend.operator import OperatorRepr
from pulser.backend.state import StateRepr
from pulser.json.exceptions import AbstractReprError

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
    assert obs_repr["evaluation_times"] is expected_kwargs.get(
        "evaluation_times", None
    )
    assert obs_repr["tag_suffix"] == expected_kwargs.get("tag_suffix", None)

    # test kwargs
    for key, expected_value in expected_kwargs.items():
        assert obs_repr[key] is expected_value


@mark.parametrize(
    "state_kwargs",
    [
        {
            "eigenstates": ("r", "g"),
            "amplitudes": {"rgr": 1.0, "grg": 1.0},
        },
        {
            "eigenstates": ("0", "1"),
            "amplitudes": {"1000": 1.0 + 0.5j, "0001": 1.0 - 0.5j},
        },
    ],
)
def test_fidelity_repr(state_kwargs):
    state = StateRepr.from_state_amplitudes(**state_kwargs)
    fidelity = Fidelity(state)
    fidelity_repr = fidelity._to_abstract_repr()
    state_in_repr = fidelity_repr["state"]
    assert state_in_repr is state
    assert state_in_repr._eigenstates == state_kwargs["eigenstates"]
    assert state_in_repr._amplitudes == state_kwargs["amplitudes"]


@mark.parametrize(
    "op_kwargs",
    [
        {
            "eigenstates": ("0", "1"),
            "n_qudits": 3,
            "operations": [],
        },
        {
            "eigenstates": ("r", "g"),
            "n_qudits": 5,
            "operations": [
                (
                    1.0j,
                    [
                        ({"rg": 0.72j}, [0, 2]),
                        ({"rr": 1.0, "gg": -1.0}, [1, 3]),
                    ],
                ),
                (
                    0.5j,
                    [({"gr": 1.0j}, [4])],
                ),
            ],
        },
    ],
)
def test_expectation_repr(op_kwargs):
    op = OperatorRepr.from_operator_repr(**op_kwargs)
    expectation = Expectation(op)

    expectation_repr = expectation._to_abstract_repr()

    op_in_repr = expectation_repr["operator"]
    assert op_in_repr is op
    assert op_in_repr._eigenstates == op_kwargs["eigenstates"]
    assert op_in_repr._n_qudits == op_kwargs["n_qudits"]
    assert op_in_repr._operations == op_kwargs["operations"]


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
    # dump with AbstrctReprEncoder & validation
    config_str = config.to_abstract_repr()
    # load and redump but with default JSON encoder
    # equivalent to go key by key and check single str repr
    config_load_dump_str = json.dumps(json.loads(config_str))

    assert config_str == config_load_dump_str


class TestStateRepr:
    def test_n_qudits(self):
        basis = ("0", "1")
        amplitudes = {"00000": 1.0j}
        state = StateRepr.from_state_amplitudes(
            eigenstates=basis, amplitudes=amplitudes
        )
        assert state.n_qudits == 5

    def test_state_repr(self):
        basis = ("r", "g")
        amplitudes = {"rgr": 1.0j + 0.2, "grg": 1.0}
        expected_repr = {"eigenstates": basis, "amplitudes": amplitudes}
        state = StateRepr.from_state_amplitudes(
            eigenstates=basis, amplitudes=amplitudes
        )
        state_repr = state._to_abstract_repr()
        assert state_repr == expected_repr

    def test_state_repr_invalid_eigenstates(self):
        basis = ("av", "b", "c")
        with pytest.raises(
            ValueError,
            match="All eigenstates must be represented by single characters.",
        ):
            StateRepr(eigenstates=basis)

    def test_not_from_amplitudes(self):
        state = StateRepr(eigenstates=("r", "g"))
        with pytest.raises(AbstractReprError):
            state._to_abstract_repr()

    def test_state_repr_not_implemented(self):
        state = StateRepr.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"rgr": 1.0, "grg": 1.0}
        )
        with pytest.raises(NotImplementedError):
            state.overlap(state)
        with pytest.raises(NotImplementedError):
            state.sample(num_shots=10)


class TestOperatorRepr:
    def test_operator_repr(self):
        basis = ("r", "g")
        n_qudits = 5
        # am I a valid operations
        operations = [
            (
                1.0,
                [
                    ({"gr": 1.0, "rg": 1.0}, [0, 2]),
                    ({"rr": 1.0, "gg": -1.0}, [1, 3]),
                ],
            )
        ]
        expected_op_repr = {
            "eigenstates": basis,
            "n_qudits": n_qudits,
            "operations": operations,
        }

        op = OperatorRepr.from_operator_repr(
            eigenstates=basis, n_qudits=n_qudits, operations=operations
        )

        op_repr = op._to_abstract_repr()

        assert op_repr == expected_op_repr

    def test_operator_repr_not_implemented(self):
        op_repr = {"eigenstates": ("r", "g"), "n_qudits": 5, "operations": []}
        op = OperatorRepr.from_operator_repr(**op_repr)
        mock_state = MagicMock()
        with pytest.raises(NotImplementedError):
            op.apply_to(mock_state)
        with pytest.raises(NotImplementedError):
            op.expect(mock_state)
        with pytest.raises(NotImplementedError):
            op.__add__(op)
        with pytest.raises(NotImplementedError):
            op.__rmul__(3.0)
        with pytest.raises(NotImplementedError):
            op.__matmul__(op)
