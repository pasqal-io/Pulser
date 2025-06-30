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
    Results,
    StateResult,
)
from pulser.backend.operator import OperatorRepr
from pulser.backend.state import StateRepr
from pulser.exceptions.serialization import AbstractReprError
from pulser.json.abstract_repr.backend import (
    _deserialize_observable,
    _deserialize_operator,
    _deserialize_state,
)
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
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

        # serialized repr
        obs_repr = json.loads(json.dumps(obs, cls=AbstractReprEncoder))

        # deserialized repr
        deserialized_obs = _deserialize_observable(
            obs_repr, StateRepr, OperatorRepr
        )
        deserialized_obs_repr = deserialized_obs._to_abstract_repr()

        for repr in [obs_repr, deserialized_obs_repr]:
            # test default values
            assert repr["observable"] == obs._base_tag
            assert repr["tag_suffix"] == expected_kwargs.get(
                "tag_suffix", None
            )
            if repr["evaluation_times"] is None:
                assert "evaluation_times" not in expected_kwargs
            else:
                assert np.allclose(
                    repr["evaluation_times"],
                    expected_kwargs["evaluation_times"],
                )
                assert repr["evaluation_times"] == json.loads(
                    json.dumps(
                        expected_kwargs["evaluation_times"],
                        cls=AbstractReprEncoder,
                    )
                )
            assert repr.get("one_state", None) == expected_kwargs.get(
                "one_state", None
            )
            assert repr.get("num_shots", 1000) == expected_kwargs.get(
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
            {
                "eigenstates": ["u", "d", "x"],
                "amplitudes": {"uuddx": 1 / 2},
            },
        ],
    )
    def test_state_in_fidelity_repr(self, state_kwargs):
        state = StateRepr.from_state_amplitudes(**state_kwargs)
        fidelity = Fidelity(state)

        # state repr as dict
        fidelity_repr = fidelity._to_abstract_repr()
        state_repr = fidelity_repr["state"]
        assert state_repr._eigenstates == state_kwargs["eigenstates"]
        assert state_repr._amplitudes == state_kwargs["amplitudes"]

        # deserialized state
        dumped_fidelity_repr = json.loads(
            json.dumps(fidelity, cls=AbstractReprEncoder)
        )
        deserialized_fidelity = _deserialize_observable(
            dumped_fidelity_repr, StateRepr, OperatorRepr
        )
        deserialized_state = deserialized_fidelity.state
        assert isinstance(deserialized_state, StateRepr)
        assert deserialized_state._eigenstates == list(
            state_kwargs["eigenstates"]
        )
        assert deserialized_state._amplitudes == dict(
            state_kwargs["amplitudes"]
        )

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
        op_repr = expectation_repr["operator"]
        assert op_repr._eigenstates == op_kwargs["eigenstates"]
        assert op_repr._n_qudits == op_kwargs["n_qudits"]
        assert op_repr._operations == op_kwargs["operations"]

        # deserialized operator
        dumped_expectation_repr = json.loads(
            json.dumps(expectation, cls=AbstractReprEncoder)
        )
        deserialized_expectation = _deserialize_observable(
            dumped_expectation_repr, StateRepr, OperatorRepr
        )
        deserialized_op = deserialized_expectation.operator
        assert isinstance(deserialized_op, OperatorRepr)
        assert deserialized_op._eigenstates == list(op_kwargs["eigenstates"])
        assert deserialized_op._n_qudits == op_kwargs["n_qudits"]
        assert deserialized_op._operations == op_kwargs["operations"]

    def test_state_result_not_supported(self):
        with pytest.raises(
            AbstractReprError, match="not supported in any remote backend"
        ):
            json.dumps(StateResult(), cls=AbstractReprEncoder)

    def test_not_supported_observable(self):
        obs = BitStrings()

        corrupted_obs_repr = json.loads(
            json.dumps(obs, cls=AbstractReprEncoder)
        )
        corrupted_obs_repr["observable"] = "I'm not valid"
        with pytest.raises(AbstractReprError, match="Failed to deserialize"):
            _deserialize_observable(
                corrupted_obs_repr, StateRepr, OperatorRepr
            )


class TestConfigRepr:
    example_state = StateRepr.from_state_amplitudes(
        eigenstates=("0", "1"), amplitudes={"1111": 0.1}
    )

    def test_config_not_from_str(self):
        with pytest.raises(
            TypeError,
            match="The serialized EmulationConfig must be given as a string. ",
        ):
            EmulationConfig.from_abstract_repr(1.0)

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
            {"max_bond_dim": 10, "precision": 1e-6, "gpu": True},
        ],
    )
    def test_config_repr(self, observables, kwargs):
        expected_kwargs = {"observables": observables}
        expected_kwargs |= kwargs

        config = EmulationConfig(**expected_kwargs)
        config_str = config.to_abstract_repr()
        deserialized_config = EmulationConfig.from_abstract_repr(config_str)

        # check that config attributes are preserved by
        # serializing and deserializing back
        assert isinstance(deserialized_config, EmulationConfig)

        # check each observables with their repr
        for obs, expected_obs in zip(
            deserialized_config.observables, config.observables
        ):
            assert obs._to_abstract_repr() == expected_obs._to_abstract_repr()
        if isinstance(config.default_evaluation_times, np.ndarray):
            assert np.allclose(
                config.default_evaluation_times,
                deserialized_config.default_evaluation_times,
            )
        else:
            (
                config.default_evaluation_times
                == deserialized_config.default_evaluation_times
            )

        # check that initial state has the same repr in both config
        if config.initial_state is None:
            assert deserialized_config.initial_state is None
        else:
            expected_state = config.initial_state
            expected_state_repr = expected_state._to_abstract_repr()
            state = deserialized_config.initial_state
            assert isinstance(state, StateRepr)
            state_repr = state._to_abstract_repr()
            assert state_repr == expected_state_repr

        assert deserialized_config.with_modulation == config.with_modulation
        if config.interaction_matrix is None:
            assert deserialized_config.interaction_matrix is None
        else:
            assert np.allclose(
                deserialized_config.interaction_matrix,
                config.interaction_matrix,
            )
        assert (
            deserialized_config.prefer_device_noise_model
            == config.prefer_device_noise_model
        )
        assert deserialized_config.noise_model == config.noise_model

        # check additional kwargs
        additional_kwargs = expected_kwargs.keys() - config._expected_kwargs()
        for key in additional_kwargs:
            assert getattr(config, key) == expected_kwargs[key]
            assert getattr(deserialized_config, key) == expected_kwargs[key]


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


@pytest.mark.parametrize(
    "test_torch",
    [True, False],
)
def test_result_serialization(test_torch: bool):
    bitstrings = BitStrings()
    corr = CorrelationMatrix()
    energy = Energy()
    occ = Occupation()
    results = Results(atom_order=(), total_duration=100)
    bitstring = "rgrgrg"
    results._store(observable=bitstrings, time=0.1, value=bitstring)
    if test_torch:
        torch = pytest.importorskip("torch")
        cor_mat = torch.randn(6, 6)
    else:
        cor_mat = np.random.randn(6, 6)
    results._store(observable=corr, time=0.2, value=cor_mat)
    energy_val = 5.0
    results._store(observable=energy, time=0.3, value=energy_val)
    if test_torch:
        torch = pytest.importorskip("torch")
        occ_vec = torch.randn(6)
    else:
        occ_vec = np.random.randn(6)
    results._store(observable=occ, time=0.4, value=occ_vec)

    dict = results._to_abstract_repr()
    assert dict["results"][str(bitstrings.uuid)] == [bitstring]

    assert len(dict["results"][str(corr.uuid)]) == 1
    assert type(dict["results"][str(corr.uuid)][0]) is type(cor_mat)
    assert dict["results"][str(corr.uuid)][0].tolist() == cor_mat.tolist()

    assert dict["results"][str(energy.uuid)] == [energy_val]

    assert len(dict["results"][str(occ.uuid)]) == 1
    assert type(dict["results"][str(occ.uuid)][0]) is type(occ_vec)
    assert dict["results"][str(occ.uuid)][0].tolist() == occ_vec.tolist()

    assert dict["tagmap"] == {
        bitstrings.tag: str(bitstrings.uuid),
        corr.tag: str(corr.uuid),
        energy.tag: str(energy.uuid),
        occ.tag: str(occ.uuid),
    }

    assert dict["times"] == {
        str(bitstrings.uuid): [0.1],
        str(corr.uuid): [0.2],
        str(energy.uuid): [0.3],
        str(occ.uuid): [0.4],
    }

    abstract_repr = results.to_abstract_repr()

    assert abstract_repr == json.dumps(dict, cls=AbstractReprEncoder)

    deserialized = Results.from_abstract_repr(abstract_repr)

    assert results.energy == deserialized.energy
    assert results.bitstrings == deserialized.bitstrings
    assert [x.tolist() for x in results.occupation] == deserialized.occupation
    assert [
        x.tolist() for x in results.correlation_matrix
    ] == deserialized.correlation_matrix
    for obs in [bitstrings, occ, corr, energy]:
        assert results.get_result_times(obs) == deserialized.get_result_times(
            obs
        )
    assert results.get_result_tags() == deserialized.get_result_tags()


def test_result_atom_order_serialization():
    with pytest.warns(UserWarning, match="converts all qubit ID's to strings"):
        assert Results.from_abstract_repr(
            Results(
                atom_order=(0, 1, 2), total_duration=1000
            ).to_abstract_repr()
        ) == Results(atom_order=("0", "1", "2"), total_duration=1000)

        with pytest.raises(
            AbstractReprError, match="Name collisions encountered"
        ):
            Results(atom_order=(0, "0"), total_duration=10).to_abstract_repr()
