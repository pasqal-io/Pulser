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
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
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
            BitStrings,
            {},
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
    obs_repr = json.loads(json.dumps(obs, cls=AbstractReprEncoder))

    # test default values
    assert obs_repr["observable"] == obs._base_tag
    assert obs_repr["tag_suffix"] == expected_kwargs.get("tag_suffix", None)
    if obs_repr["evaluation_times"] is None:
        assert "evaluation_times" not in expected_kwargs
    else:
        assert np.allclose(
            obs_repr["evaluation_times"], expected_kwargs["evaluation_times"]
        )
        assert obs_repr["evaluation_times"] == json.loads(
            json.dumps(
                expected_kwargs["evaluation_times"], cls=AbstractReprEncoder
            )
        )

    if "one_state" in obs_repr:
        assert obs_repr["one_state"] == expected_kwargs.get("one_state", None)
    if "num_shots" in obs_repr:
        assert obs_repr["num_shots"] == expected_kwargs.get("num_shots", 1000)


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

    def test_state_repr_invalid_eigenstates(self):
        basis = ("av", "b", "c")
        with pytest.raises(
            ValueError,
            match="All eigenstates must be represented by single characters.",
        ):
            StateRepr(eigenstates=basis)

    def test_invalid_amplitudes(self):
        basis = ("0", "1")
        amplitudes = {"00000": 1.0j, "rrrrr": 1.0}
        with pytest.raises(
            ValueError,
            match="must be combinations of eigenstates with the same length",
        ):
            StateRepr.from_state_amplitudes(
                eigenstates=basis, amplitudes=amplitudes
            )

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

    @mark.parametrize(
        "expected_repr",
        [
            {
                "eigenstates": ("r", "g"),
                "amplitudes": {"rgr": 1.0j + 0.2, "grg": 0.22j, "rrr": -2.0},
            },
            {
                "eigenstates": ("0", "1"),
                "amplitudes": {"10001": 0.5, "01010": 0.5},
            },
        ],
    )
    def test_state_repr(self, expected_repr):
        state = StateRepr.from_state_amplitudes(**expected_repr)
        state_repr = state._to_abstract_repr()
        assert state_repr == expected_repr

        # dump and reload repr
        state_repr = json.loads(json.dumps(state, cls=AbstractReprEncoder))
        assert state_repr["eigenstates"] == list(expected_repr["eigenstates"])
        amplitudes = state_repr["amplitudes"]
        expected_amplitudes = expected_repr["amplitudes"]
        for k, expected_amp in expected_amplitudes.items():
            amp = amplitudes[k]
            if isinstance(expected_amp, complex):
                amp = complex(amp["real"], amp["imag"])
            assert amp == expected_amp


class TestOperatorRepr:
    @mark.parametrize(
        "expected_repr",
        [
            {
                "eigenstates": ("r", "g"),
                "n_qudits": 5,
                "operations": [
                    (
                        1.0,
                        [
                            ({"gr": 1.0, "rg": 1.0}, [0, 2]),
                            ({"rr": 1.0, "gg": -1.0}, [1, 3, 4]),
                        ],
                    )
                ],
            },
            {
                "eigenstates": ("0", "1"),
                "n_qudits": 3,
                "operations": [
                    (
                        0.1j,
                        [
                            ({"01": -1.0j, "10": 1.0j}, [0, 2]),
                        ],
                    ),
                    (
                        0.7j,
                        [
                            ({"11": -0.7j, "00": 2.3 + 0.22j}, [1, 2]),
                        ],
                    ),
                ],
            },
        ],
    )
    def test_operator_repr(self, expected_repr):
        operator = OperatorRepr.from_operator_repr(**expected_repr)
        op_repr = operator._to_abstract_repr()
        assert op_repr == expected_repr

        # dump and reload repr
        op_repr = json.loads(json.dumps(operator, cls=AbstractReprEncoder))
        assert op_repr["eigenstates"] == list(expected_repr["eigenstates"])
        assert op_repr["n_qudits"] == expected_repr["n_qudits"]
        operations = op_repr["operations"]
        for i, tensor_op in enumerate(operations):
            if isinstance(tensor_op[0], dict):
                tensor_op[0] = complex(
                    tensor_op[0]["real"], tensor_op[0]["imag"]
                )
            for j, qudit_op in enumerate(tensor_op[1]):
                assert len(qudit_op) == 2
                assert isinstance(qudit_op[0], dict)
                assert isinstance(qudit_op[1], list)
                for k, v in qudit_op[0].items():
                    if isinstance(v, dict):
                        qudit_op[0][k] = complex(v["real"], v["imag"])
                # repack as tuple
                tensor_op[1][j] = tuple(qudit_op)
            operations[i] = tuple(tensor_op)
        assert operations == expected_repr["operations"]

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
