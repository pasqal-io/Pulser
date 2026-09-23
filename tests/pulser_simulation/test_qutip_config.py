import json
import re

import numpy as np
import pytest
import qutip

from pulser import NoiseModel
from pulser.backend import EmulationConfig, OperatorRepr, StateRepr
from pulser.backend.default_observables import (
    BitStrings,
    Expectation,
    Fidelity,
    StateResult,
)
from pulser_simulation import QutipBackendV2
from pulser_simulation.qutip_config import (
    QutipConfig,
    QutipOperator,
    QutipState,
    Solver,
)


def test_no_interaction_matrix():
    with pytest.raises(
        NotImplementedError,
        match="'QutipBackendV2' does not handle custom interaction matrices.",
    ):
        QutipConfig(
            observables=[
                StateResult(evaluation_times=[1.0]),
            ],
            interaction_matrix=np.eye(4),
        )


def test_sampling_rate():
    with pytest.raises(
        ValueError, match="be greater than 0 and less than or equal to 1"
    ):
        QutipConfig(
            observables=[
                StateResult(evaluation_times=[1.0]),
            ],
            sampling_rate=1.2,
        )

    config = QutipConfig(
        observables=[
            StateResult(evaluation_times=[1.0]),
        ],
        sampling_rate=0.5,
    )

    assert "sampling_rate" in config._expected_kwargs()


def test_samples_per_run():
    with pytest.warns(
        UserWarning,
        match="The number of samples per run .* is ignored "
        "when using QutipBackendV2.",
    ):
        with pytest.warns(
            DeprecationWarning,
            match="Setting samples_per_run different to 1 is",
        ):
            QutipConfig(
                observables=[
                    StateResult(evaluation_times=[1.0]),
                ],
                noise_model=NoiseModel(temperature=45, samples_per_run=5),
            )


def test_initial_state():
    with pytest.raises(
        TypeError,
        match=re.escape("'initial_state' must be an instance of State"),
    ):
        QutipConfig(
            observables=[
                StateResult(evaluation_times=[1.0]),
            ],
            initial_state="all-ground",
        )


def _repr_state_and_op():
    state = StateRepr.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rr": 1.0}
    )
    op = OperatorRepr.from_operator_repr(
        eigenstates=("r", "g"),
        n_qudits=2,
        operations=[(1.0, [({"rr": 1.0}, [0])])],
    )
    return state, op


def test_implicit_cast():
    state, op = _repr_state_and_op()
    fid = Fidelity(state)
    exp = Expectation(op)
    config = QutipConfig(initial_state=state, observables=[fid, exp])

    assert isinstance(config.initial_state, QutipState)
    assert config.initial_state._amplitudes == {"rr": 1.0}
    new_fid, new_exp = config.observables
    assert isinstance(new_fid.state, QutipState)
    assert isinstance(new_exp.operator, QutipOperator)
    # UUIDs are kept, so results can be retrieved with the originals
    assert new_fid.uuid == fid.uuid and new_exp.uuid == exp.uuid
    assert new_fid.tag == fid.tag and new_exp.tag == exp.tag
    # The originals are not modified
    assert fid.state is state and exp.operator is op
    assert new_fid is not fid and new_exp is not exp


def test_implicit_cast_from_base_config():
    state, op = _repr_state_and_op()
    base_config = EmulationConfig(
        initial_state=state, observables=[Fidelity(state), Expectation(op)]
    )
    # Base config keeps the backend-agnostic types
    assert type(base_config.initial_state) is StateRepr
    assert type(base_config.observables[0].state) is StateRepr
    assert type(base_config.observables[1].operator) is OperatorRepr
    # Casting happens when the config is given to a backend
    config = QutipBackendV2.validate_config(base_config)
    assert isinstance(config, QutipConfig)
    assert isinstance(config.initial_state, QutipState)
    assert isinstance(config.observables[0].state, QutipState)
    assert isinstance(config.observables[1].operator, QutipOperator)


def test_no_cast_needed():
    qutip_state = QutipState(qutip.basis(4, 0), eigenstates=("r", "g"))
    config = QutipConfig(
        initial_state=qutip_state, observables=[Fidelity(qutip_state)]
    )
    # Non-serializable states are fine when they already have the right type
    assert config.initial_state._amplitudes is None
    assert config.initial_state.overlap(qutip_state) == pytest.approx(1.0)
    assert config.observables[0].state._amplitudes is None
    # ...and also in the backend-agnostic config
    base_config = EmulationConfig(
        initial_state=qutip_state, observables=[StateResult()]
    )
    assert type(base_config.initial_state) is QutipState


def test_failed_cast():
    qutip_state = QutipState(qutip.basis(4, 0), eigenstates=("r", "g"))
    base_config = EmulationConfig(
        initial_state=StateRepr.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"rr": 1.0}
        ),
        observables=[Fidelity(qutip_state)],
    )

    class OtherState(StateRepr):
        pass

    class OtherConfig(EmulationConfig):
        _state_type = OtherState

    with pytest.raises(
        TypeError,
        match="Failed to convert the state of observable 'fidelity' of type "
        "'QutipState' to the expected state type 'OtherState'",
    ):
        OtherConfig(**base_config._backend_options)


def test_preferred_types():
    assert QutipConfig.state_type is QutipState
    assert QutipConfig.operator_type is QutipOperator


def test_progress_bar():
    config = QutipConfig(
        observables=[
            StateResult(evaluation_times=[1.0]),
        ],
        progress_bar=True,
    )
    assert config.progress_bar
    assert "progress_bar" in config._expected_kwargs()


def test_evaluation_times_as_numpy_arrays():
    default_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    obs_times_1 = np.array([0.2, 0.4, 0.8])
    obs_times_2 = np.array([0.15, 0.35, 0.65, 0.95])

    config = QutipConfig(
        observables=[
            StateResult(evaluation_times=obs_times_1),
            StateResult(evaluation_times=obs_times_2, tag_suffix="second"),
        ],
        default_evaluation_times=default_times,
    )

    expected_times = np.union1d(
        np.union1d(default_times, obs_times_1), obs_times_2
    )

    # By putting total_duration = 1000 ns, we get the legacy evaluation times
    # in microseconds matching the relative evaluation times
    np.testing.assert_almost_equal(
        config._get_legacy_evaluation_times(1000), expected_times
    )


@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("solver", list(Solver))
def test_solver_deserialization(solver, as_str):
    config = QutipConfig(
        observables=[
            BitStrings(evaluation_times=[1.0]),
        ],
        solver=solver if not as_str else str(solver.value),
    )

    ser_config = config.to_abstract_repr()
    assert json.loads(ser_config)["solver"] == str(solver.value)
    re_config = QutipConfig.from_abstract_repr(ser_config)
    assert re_config.solver is solver


def test_invalid_solver_error():
    with pytest.raises(ValueError, match="Invalid solver 'fakesolver'"):
        QutipConfig(
            observables=[
                BitStrings(evaluation_times=[1.0]),
            ],
            solver="fakesolver",
        )
