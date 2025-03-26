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
from pulser.json.abstract_repr.backend import _deserialize_operator
from pulser.backend.state import StateRepr
from pulser.json.abstract_repr.backend import _deserialize_state
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.exceptions import AbstractReprError
from pulser.noise_model import NoiseModel


class TestObservableRepr:
    example_state = StateRepr.from_state_amplitudes(
        eigenstates=("0", "1"), amplitudes={"11": 0.1}
    )
    example_operator = OperatorRepr.from_operator_repr(
        eigenstates=("r", "g"),
        n_qudits=3,
        operations=[(0.3, [({"rr": 0.2j}, [0, 2])])],
    )

    @mark.parametrize(
        "observable, arg, expected_kwargs",
        [
            (
                BitStrings,
                (),
                {
                    "evaluation_times": [i * 0.05 for i in range(10)],
                    "num_shots": 211,
                    "one_state": "r",
                    "tag_suffix": "7",
                },
            ),
            (
                BitStrings,
                (),
                {},
            ),
            (
                CorrelationMatrix,
                (),
                {"one_state": "r"},
            ),
            (
                Occupation,
                (),
                {"one_state": "g"},
            ),
            (
                Energy,
                (),
                {"evaluation_times": [i * 0.05 for i in range(10)]},
            ),
            (
                EnergyVariance,
                (),
                {"evaluation_times": np.linspace(0, 1, 13)},
            ),
            (
                EnergySecondMoment,
                (),
                {"evaluation_times": [i * 0.1 for i in range(5)]},
            ),
            (
                Fidelity,
                (example_state,),
                {"evaluation_times": [i / 7.2 for i in range(5)]},
            ),
            (
                Expectation,
                (example_operator,),
                {"tag_suffix": "my_op"},
            ),
        ],
    )
    def test_observable_repr(self, observable, arg, expected_kwargs):
        obs = observable(*arg, **expected_kwargs)
        obs_repr = json.loads(json.dumps(obs, cls=AbstractReprEncoder))

        # test default values
        assert obs_repr["observable"] == obs._base_tag
        assert obs_repr["tag_suffix"] == expected_kwargs.get(
            "tag_suffix", None
        )
        if obs_repr["evaluation_times"] is None:
            assert "evaluation_times" not in expected_kwargs
        else:
            assert np.allclose(
                obs_repr["evaluation_times"],
                expected_kwargs["evaluation_times"],
            )
            assert obs_repr["evaluation_times"] == json.loads(
                json.dumps(
                    expected_kwargs["evaluation_times"],
                    cls=AbstractReprEncoder,
                )
            )
        assert obs_repr.get("one_state", None) == expected_kwargs.get(
            "one_state", None
        )
        assert obs_repr.get("num_shots", 1000) == expected_kwargs.get(
            "num_shots", 1000
        )

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
    def test_state_in_fidelity_repr(self, state_kwargs):
        state = StateRepr.from_state_amplitudes(**state_kwargs)
        fidelity = Fidelity(state)
        fidelity_repr = fidelity._to_abstract_repr()
        # test default values
        assert fidelity_repr["observable"] == "fidelity"
        assert fidelity_repr["tag_suffix"] is None
        assert fidelity_repr["evaluation_times"] is None
        # test state in repr
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
    def test_operator_in_expectation_repr(self, op_kwargs):
        op = OperatorRepr.from_operator_repr(**op_kwargs)
        expectation = Expectation(op)
        expectation_repr = expectation._to_abstract_repr()
        op_in_repr = expectation_repr["operator"]
        assert op_in_repr is op
        assert op_in_repr._eigenstates == op_kwargs["eigenstates"]
        assert op_in_repr._n_qudits == op_kwargs["n_qudits"]
        assert op_in_repr._operations == op_kwargs["operations"]

    def test_state_result_not_supported(self):
        with pytest.raises(
            AbstractReprError, match="not supported in any remote backend"
        ):
            json.dumps(StateResult(), cls=AbstractReprEncoder)


class TestConfigRepr:
    example_state = StateRepr.from_state_amplitudes(
        eigenstates=("0", "1"), amplitudes={"1111": 0.1}
    )

    @mark.parametrize(
        "observables",
        [
            (
                BitStrings(evaluation_times=[i * 0.01 for i in range(10)]),
                CorrelationMatrix(),
            ),
            (Energy(), Occupation(one_state="0")),
        ],
    )
    @mark.parametrize(
        "kwargs",
        [
            {"with_modulation": True, "initial_state": example_state},
            {
                "default_evaluation_times": [0.1, 0.2, 0.3],
                "prefer_device_noise_model": True,
            },
            {
                "default_evaluation_times": "Full",
                "interaction_matrix": [[0.0, 0.5], [0.5, 0.0]],
            },
            {"noise_model": NoiseModel(p_false_pos=0.1, dephasing_rate=0.01)},
        ],
    )
    def test_config_repr(self, observables, kwargs):
        expected_kwargs = {"observables": observables}
        expected_kwargs |= kwargs

        config = EmulationConfig(**expected_kwargs)
        # dump with AbstrctReprEncoder & validation
        config_str = config.to_abstract_repr()
        config_repr = json.loads(config_str)

        # check observables
        assert len(config_repr["observables"]) == len(
            expected_kwargs["observables"]
        )
        for obs, expected_obs in zip(
            config_repr["observables"], expected_kwargs["observables"]
        ):
            assert obs == json.loads(
                json.dumps(expected_obs, cls=AbstractReprEncoder)
            )

        # check defaults args vs simple type or dump/reload as dict
        assert config_repr["default_evaluation_times"] == expected_kwargs.get(
            "default_evaluation_times", [1.0]
        )
        if config_repr["initial_state"] is None:
            assert "initial_state" not in expected_kwargs
        else:
            ini_state_repr = config_repr["initial_state"]
            expected = json.loads(
                json.dumps(
                    expected_kwargs["initial_state"], cls=AbstractReprEncoder
                )
            )
            assert ini_state_repr == expected
        assert config_repr["with_modulation"] == expected_kwargs.get(
            "with_modulation", False
        )
        if config_repr["interaction_matrix"] is None:
            assert "interaction_matrix" not in expected_kwargs
        else:
            assert np.allclose(
                config_repr["interaction_matrix"],
                expected_kwargs["interaction_matrix"],
            )
            assert config_repr["interaction_matrix"] == json.loads(
                json.dumps(
                    expected_kwargs["interaction_matrix"],
                    cls=AbstractReprEncoder,
                )
            )
        assert config_repr["prefer_device_noise_model"] == expected_kwargs.get(
            "prefer_device_noise_model", False
        )
        expected_noise_model = expected_kwargs.get("noise_model", NoiseModel())
        assert config_repr["noise_model"] == json.loads(
            json.dumps(expected_noise_model, cls=AbstractReprEncoder)
        )


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
                "eigenstates": ["r", "g"],
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
        assert state_repr["eigenstates"] == tuple(expected_repr["eigenstates"])
        assert state_repr["amplitudes"] == dict(expected_repr["amplitudes"])

        # repr is preserved serializing an deserializing a state
        state_repr_from_str = json.loads(
            json.dumps(state, cls=AbstractReprEncoder)
        )
        deserialized_state = _deserialize_state(state_repr_from_str, StateRepr)
        assert isinstance(deserialized_state, StateRepr)
        deserialized_state_repr = deserialized_state._to_abstract_repr()
        assert deserialized_state_repr == state_repr


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
            {
                "eigenstates": ["r", "g", "l"],
                "n_qudits": 2,
                "operations": [
                    (
                        -1.0j,
                        [
                            ({"gr": 1.0, "rg": 1.0}, [0]),
                            ({"ll": 1.0}, [1]),
                        ],
                    )
                ],
            },
        ],
    )
    def test_operator_repr(self, expected_repr):
        operator = OperatorRepr.from_operator_repr(**expected_repr)

        operator_repr = operator._to_abstract_repr()
        assert operator_repr["eigenstates"] == tuple(
            expected_repr["eigenstates"]
        )
        assert operator_repr["n_qudits"] == expected_repr["n_qudits"]
        assert operator_repr["operations"] == expected_repr["operations"]

        # repr is preserved serializing an deserializing an operator
        operator_repr_from_str = json.loads(
            json.dumps(operator, cls=AbstractReprEncoder)
        )
        deserialized_operator = _deserialize_operator(
            operator_repr_from_str, OperatorRepr
        )
        assert isinstance(deserialized_operator, OperatorRepr)
        deserialized_operator_repr = deserialized_operator._to_abstract_repr()
        assert deserialized_operator_repr == operator_repr

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
