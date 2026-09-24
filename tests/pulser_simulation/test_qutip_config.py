import copy
import json
import re

import numpy as np
import pytest
import qutip

from pulser import NoiseModel
from pulser.backend import (
    Callback,
    EmulationConfig,
    OperatorRepr,
    StateRepr,
)
from pulser.backend.abc import EmulatorBackend
from pulser.backend.default_observables import (
    BitStrings,
    Expectation,
    Fidelity,
    StateResult,
)
from pulser.backend.state import _cast_state
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
        match=re.escape(
            "If provided, `initial_state` must be an instance of `QutipState`"
        ),
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


class _StateCallback(Callback):
    """A custom callback holding a state, as a dependent package might."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def __call__(self, config, t, state, hamiltonian, result):
        pass

    def _cast_to(self, state_type, operator_type):
        new_cb = copy.copy(self)
        new_cb.state = _cast_state(self.state, state_type)
        return new_cb


def test_implicit_cast_in_validate_config():
    state, op = _repr_state_and_op()
    fid = Fidelity(state)
    exp = Expectation(op)
    cb = _StateCallback(state)
    base_config = EmulationConfig(
        initial_state=state, observables=[fid, exp], callbacks=[cb]
    )
    # The config itself keeps the types it was given
    assert type(base_config.initial_state) is StateRepr
    assert type(base_config.observables[0].state) is StateRepr
    assert type(base_config.observables[1].operator) is OperatorRepr

    # The cast happens when the config is given to the backend, before
    # QutipConfig's own 'initial_state' check runs
    config = QutipBackendV2.validate_config(base_config)
    assert isinstance(config, QutipConfig)
    assert isinstance(config.initial_state, QutipState)
    assert config.initial_state._amplitudes == {"rr": 1.0}
    new_fid, new_exp = config.observables
    assert isinstance(new_fid.state, QutipState)
    assert isinstance(new_exp.operator, QutipOperator)
    assert isinstance(config.callbacks[0].state, QutipState)
    # UUIDs and tags are kept, so results can be retrieved with the originals
    assert new_fid.uuid == fid.uuid and new_exp.uuid == exp.uuid
    assert config.callbacks[0].uuid == cb.uuid
    assert new_fid.tag == fid.tag and new_exp.tag == exp.tag
    # The user's objects are not modified
    assert fid.state is state and exp.operator is op and cb.state is state
    assert type(base_config.initial_state) is StateRepr


def test_no_cast_needed():
    qutip_state = QutipState(qutip.basis(4, 0), eigenstates=("r", "g"))
    fid = Fidelity(qutip_state)
    # Non-serializable states are fine when they already have the right type
    assert fid._cast_to(QutipState, QutipOperator) is fid
    assert _cast_state(qutip_state, QutipState) is qutip_state
    # ...or when the target is the backend-agnostic StateRepr
    assert _cast_state(qutip_state, StateRepr) is qutip_state
    # Callbacks and observables without states are returned as they are
    obs = StateResult()
    assert obs._cast_to(QutipState, QutipOperator) is obs

    config = QutipBackendV2.validate_config(
        EmulationConfig(initial_state=qutip_state, observables=[fid])
    )
    assert config.initial_state._amplitudes is None
    assert config.initial_state.overlap(qutip_state) == pytest.approx(1.0)
    assert config.observables[0].state._amplitudes is None
    assert config.observables[0].uuid == fid.uuid


class _OtherState(StateRepr):
    pass


class _OtherOperator(OperatorRepr):
    pass


class _OtherConfig(EmulationConfig):
    _state_type = _OtherState
    _operator_type = _OtherOperator


class _OtherBackend(EmulatorBackend):
    default_config = _OtherConfig(observables=[StateResult()])

    def run(self):
        pass


def test_failed_cast():
    qutip_state = QutipState(qutip.basis(4, 0), eigenstates=("r", "g"))
    qutip_op = QutipOperator(qutip.qeye([2, 2]), eigenstates=("r", "g"))
    state, op = _repr_state_and_op()

    # Serializable objects of another type are cast
    config = _OtherBackend.validate_config(
        EmulationConfig(
            initial_state=state, observables=[Fidelity(state), Expectation(op)]
        )
    )
    assert type(config.initial_state) is _OtherState
    assert type(config.observables[0].state) is _OtherState
    assert type(config.observables[1].operator) is _OtherOperator

    # Non-serializable objects of another type can't be cast
    with pytest.raises(
        TypeError,
        match="Failed to convert 'initial_state' of type 'QutipState' "
        "to the expected state type '_OtherState'",
    ):
        _OtherBackend.validate_config(
            EmulationConfig(
                initial_state=qutip_state, observables=[Fidelity(state)]
            )
        )
    with pytest.raises(
        TypeError,
        match="Failed to convert the state of observable 'fidelity' of type "
        "'QutipState' to the expected state type '_OtherState'",
    ):
        _OtherBackend.validate_config(
            EmulationConfig(observables=[Fidelity(qutip_state)])
        )
    with pytest.raises(
        TypeError,
        match="Failed to convert the operator of observable 'expectation' of "
        "type 'QutipOperator' to the expected operator type '_OtherOperator'",
    ):
        _OtherBackend.validate_config(
            EmulationConfig(observables=[Expectation(qutip_op)])
        )


class _TwoLevelState(StateRepr):
    """A state type that only supports ('r', 'g'), like some emulators."""

    @classmethod
    def _from_state_amplitudes(cls, *, eigenstates, n_qudits, amplitudes):
        if tuple(eigenstates) != ("r", "g"):
            raise ValueError("Only ('r', 'g') eigenstates are supported.")
        return super()._from_state_amplitudes(
            eigenstates=eigenstates, n_qudits=n_qudits, amplitudes=amplitudes
        )


class _TwoLevelConfig(EmulationConfig):
    _state_type = _TwoLevelState


class _TwoLevelBackend(EmulatorBackend):
    default_config = _TwoLevelConfig(observables=[StateResult()])

    def run(self):
        pass


def test_failed_cast_unsupported_eigenstates():
    state = StateRepr.from_state_amplitudes(
        eigenstates=("0", "1"), amplitudes={"11": 1.0}
    )
    with pytest.raises(
        TypeError,
        match="Failed to convert 'initial_state' of type 'StateRepr' "
        "to the expected state type '_TwoLevelState'",
    ) as exc_info:
        _TwoLevelBackend.validate_config(
            EmulationConfig(initial_state=state, observables=[StateResult()])
        )
    # The original error is kept as the cause
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "Only ('r', 'g')" in str(exc_info.value.__cause__)


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
