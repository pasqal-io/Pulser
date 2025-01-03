# Copyright 2023 Pulser Development Team
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
import typing
import uuid
from collections import Counter

import numpy as np
import pytest

import pulser
from pulser.backend.abc import Backend, EmulatorBackend
from pulser.backend.config import (
    BackendConfig,
    EmulationConfig,
    EmulatorConfig,
)
from pulser.backend.default_observables import (
    BitStrings,
    CorrelationMatrix,
    Energy,
    EnergyVariance,
    Expectation,
    Fidelity,
    Occupation,
    SecondMomentOfEnergy,
    StateResult,
)
from pulser.backend.qpu import QPUBackend
from pulser.backend.remote import (
    BatchStatus,
    JobStatus,
    RemoteConnection,
    RemoteResults,
    RemoteResultsError,
    _OpenBatchContextManager,
)
from pulser.backend.results import Results
from pulser.devices import AnalogDevice, MockDevice
from pulser.register import SquareLatticeLayout
from pulser.result import Result, SampledResult
from pulser_simulation import QutipState, QutipOperator


@pytest.fixture
def sequence() -> pulser.Sequence:
    reg = pulser.Register.square(2, spacing=5, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(1000, 1, -1, 0), "rydberg_global")
    return seq


def test_abc_backend(sequence):
    with pytest.raises(
        TypeError, match="Can't instantiate abstract class Backend"
    ):
        Backend(sequence)

    class ConcreteBackend(Backend):
        def run(self):
            pass

    with pytest.raises(
        TypeError, match="'sequence' should be a `Sequence` instance"
    ):
        ConcreteBackend(sequence.to_abstract_repr())


@pytest.mark.parametrize(
    "param, value, msg",
    [
        ("sampling_rate", 0, "must be greater than 0"),
        ("evaluation_times", "full", "one of the following"),
        ("evaluation_times", 1.001, "less than or equal to 1"),
        ("evaluation_times", [-1e9, 1], "must not contain negative values"),
        ("initial_state", "all_ground", "must be 'all-ground'"),
    ],
)
def test_emulator_config_value_errors(param, value, msg):
    with pytest.raises(ValueError, match=msg):
        EmulatorConfig(**{param: value})


@pytest.mark.parametrize(
    "param, msg",
    [
        ("evaluation_times", "not a valid type for 'evaluation_times'"),
        ("initial_state", "not a valid type for 'initial_state'"),
        ("noise_model", "must be a NoiseModel instance"),
    ],
)
def test_emulator_config_type_errors(param, msg):
    with pytest.raises(TypeError, match=msg):
        EmulatorConfig(**{param: None})


class _MockConnection(RemoteConnection):
    def __init__(self):
        self._status_calls = 0
        self._support_open_batch = True
        self._got_closed = ""
        self._progress_calls = 0
        self.result = SampledResult(
            ("q0", "q1"),
            meas_basis="ground-rydberg",
            bitstring_counts={"00": 100},
        )

    def submit(
        self,
        sequence,
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        **kwargs,
    ) -> RemoteResults:
        if batch_id:
            return RemoteResults("dcba", self)
        return RemoteResults("abcd", self)

    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None = None
    ) -> typing.Sequence[Result]:
        self._progress_calls += 1
        if self._progress_calls == 1:
            raise RemoteResultsError("Results not available")

        return (self.result,)

    def _query_job_progress(
        self, batch_id: str
    ) -> typing.Mapping[str, tuple[JobStatus, Result | None]]:
        return {"abcd": (JobStatus.DONE, self.result)}

    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        return BatchStatus.DONE

    def _close_batch(self, batch_id: str) -> None:
        self._got_closed = batch_id

    def supports_open_batch(self) -> bool:
        return bool(self._support_open_batch)


def test_remote_connection():
    connection = _MockConnection()

    with pytest.raises(NotImplementedError, match="Unable to find job IDs"):
        connection._get_job_ids("abc")

    with pytest.raises(
        NotImplementedError, match="Unable to fetch the available devices"
    ):
        connection.fetch_available_devices()


def test_qpu_backend(sequence):
    connection = _MockConnection()

    with pytest.raises(
        TypeError, match="must be a real device, instance of 'Device'"
    ):
        QPUBackend(sequence, connection)

    with pytest.warns(
        UserWarning, match="device with a different Rydberg level"
    ):
        seq = sequence.switch_device(AnalogDevice)
    with pytest.raises(ValueError, match="defined from a `RegisterLayout`"):
        QPUBackend(seq, connection)
    seq = seq.switch_register(SquareLatticeLayout(5, 5, 5).square_register(2))

    with pytest.raises(
        ValueError, match="does not accept new register layouts"
    ):
        QPUBackend(seq, connection)
    seq = seq.switch_register(
        AnalogDevice.pre_calibrated_layouts[0].define_register(1, 2, 3)
    )
    qpu_backend = QPUBackend(seq, connection)
    with pytest.raises(ValueError, match="'job_params' must be specified"):
        qpu_backend.run()
    with pytest.raises(TypeError, match="'job_params' must be a list"):
        qpu_backend.run(job_params={"runs": 100})
    with pytest.raises(
        TypeError, match="All elements of 'job_params' must be dictionaries"
    ):
        qpu_backend.run(job_params=[{"runs": 100}, "foo"])
    with pytest.raises(
        ValueError,
        match="All elements of 'job_params' must specify 'runs'",
    ):
        qpu_backend.run(job_params=[{"n_runs": 10}, {"runs": 11}])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All 'runs' must be below the maximum allowed by the device"
        ),
    ):
        qpu_backend.run(job_params=[{"runs": 100000}])

    remote_results = qpu_backend.run(job_params=[{"runs": 10}])

    with pytest.raises(AttributeError, match="no attribute 'result'"):
        # Cover the custom '__getattr__' default behavior
        remote_results.result

    with pytest.raises(
        RemoteResultsError,
        match=(
            "Results are not available for all jobs. "
            "Use the `get_available_results` method to retrieve partial "
            "results."
        ),
    ):
        remote_results.results

    results = remote_results.results
    assert results[0].sampling_dist == {"00": 1.0}

    # Test create a batch and submitting jobs via a context manager
    # behaves as expected.
    qpu = QPUBackend(seq, connection)
    assert connection._got_closed == ""
    with qpu.open_batch() as ob:
        assert ob.backend is qpu
        assert ob.backend._batch_id == "abcd"
        assert isinstance(ob, _OpenBatchContextManager)
        results = qpu.run(job_params=[{"runs": 200}])
        # batch_id should differ bc of how MockConnection is written
        # confirms the batch_id was provided to submit()
        assert results.batch_id == "dcba"
        assert isinstance(results, RemoteResults)
    assert qpu._batch_id is None
    assert connection._got_closed == "abcd"

    connection._support_open_batch = False
    qpu = QPUBackend(seq, connection)
    with pytest.raises(
        NotImplementedError,
        match="Unable to execute open_batch using this remote connection",
    ):
        qpu.open_batch()

    available_results = remote_results.get_available_results()
    assert available_results == {"abcd": connection.result}


def test_emulator_backend(sequence):

    class ConcreteEmulator(EmulatorBackend):

        default_config = EmulationConfig(with_modulation=True)

        def run(self):
            pass

    with pytest.raises(
        TypeError, match="must be an instance of 'EmulationConfig'"
    ):
        ConcreteEmulator(sequence, config=EmulatorConfig)

    emu = ConcreteEmulator(
        sequence, config=EmulationConfig(default_evaluation_times="Full")
    )
    assert emu._config.default_evaluation_times == "Full"
    assert not emu._config.with_modulation

    # Uses the default config
    assert ConcreteEmulator(sequence)._config.with_modulation


def test_backend_config():
    with pytest.warns(
        UserWarning,
        match="'BackendConfig' received unexpected keyword arguments",
    ):
        config1 = BackendConfig(prefer_device_noise_model=True)
        assert config1.prefer_device_noise_model

    with pytest.raises(AttributeError, match="'dt' has not been passed"):
        config1.dt

    with pytest.warns(
        DeprecationWarning,
        match="The 'backend_options' argument of 'BackendConfig' has been "
        "deprecated",
    ):
        config2 = BackendConfig(backend_options={"dt": 10})
        assert config2.backend_options["dt"] == 10
        assert config2.dt == 10


def test_emulation_config():
    with pytest.raises(
        TypeError,
        match="All entries in 'observables' must be instances of Observable",
    ):
        EmulationConfig(observables=["fidelity"])
    with pytest.raises(
        ValueError,
        match="Some of the provided 'observables' share identical tags",
    ):
        EmulationConfig(
            observables=[BitStrings(), BitStrings(num_shots=200000)]
        )
    with pytest.raises(
        ValueError, match="All evaluation times must be between 0. and 1."
    ):
        EmulationConfig(default_evaluation_times=[-1e15, 0.0, 0.5, 1.0])
    with pytest.raises(
        TypeError, match="'initial_state' must be an instance of State"
    ):
        EmulationConfig(initial_state=[[1], [0]])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'interaction_matrix' must be a square matrix. Instead, an array"
            " of shape (4, 3) was given"
        ),
    ):
        EmulationConfig(interaction_matrix=np.arange(12).reshape((4, 3)))
    with pytest.raises(TypeError, match="must be a NoiseModel"):
        EmulationConfig(noise_model={"p_false_pos": 0.1})


class TestObservables:
    @pytest.fixture
    def ghz_state(self):
        return QutipState.from_state_amplitudes(
            eigenstates=("r", "g"),
            amplitudes={"rrr": np.sqrt(0.5), "ggg": np.sqrt(0.5)},
        )

    @pytest.fixture
    def ham(self):
        return QutipOperator.from_operator_repr(
            eigenstates=("r", "g"),
            n_qudits=3,
            operations=[(1.0, [])],
        )

    @pytest.fixture
    def config(self):
        return EmulationConfig()

    @pytest.fixture
    def results(self):
        return Results(atom_order=("q0", "q1", "q2"), total_duration=1000)

    @pytest.mark.parametrize("tag_suffix", [None, "foo"])
    @pytest.mark.parametrize("eval_times", [None, (0.0, 0.5, 1.0)])
    def test_base_init(self, eval_times, tag_suffix):
        # We use StateResult because Observable is an ABC
        obs = StateResult(evaluation_times=eval_times, tag_suffix=tag_suffix)
        assert isinstance(obs.uuid, uuid.UUID)
        assert obs.evaluation_times == eval_times
        expected_tag = "state_foo" if tag_suffix else "state"
        assert obs.tag == expected_tag
        assert repr(obs) == f"{expected_tag}:{obs.uuid}"
        with pytest.raises(
            ValueError, match="All evaluation times must be between 0. and 1."
        ):
            StateResult(evaluation_times=[1.000001])

    @pytest.mark.parametrize("eval_times", [None, (0.0, 0.5, 1.0)])
    def test_call(
        self,
        config: EmulationConfig,
        results: Results,
        ghz_state,
        ham,
        eval_times,
    ):
        assert not results.get_result_tags()  # ie it's empty
        assert config.default_evaluation_times == (1.0,)
        # We use StateResult because Observable is an ABC
        obs = StateResult(evaluation_times=eval_times)
        assert obs.apply(state=ghz_state) == ghz_state
        true_eval_times = eval_times or config.default_evaluation_times

        t_ = 0.1
        assert not config.is_time_in_evaluation_times(t_, true_eval_times)
        obs(config, t_, ghz_state, ham, results)
        assert not results.get_result_tags()  # ie it's still empty

        t_ = 1.0
        assert config.is_time_in_evaluation_times(t_, true_eval_times)
        obs(config, t_, ghz_state, ham, results)
        assert results.get_result_tags() == ["state"]
        assert (
            results.get_result_times("state")
            == results.get_result_times(obs)
            == [t_]
        )
        assert results.get_result(obs, t_) == ghz_state
        with pytest.raises(
            RuntimeError,
            match="A value is already stored for observable 'state' at time "
            f"{t_}",
        ):
            obs(config, t_, ghz_state, ham, results)

        expected_tol = 0.5 / results.total_duration
        t_minus_tol = t_ - expected_tol
        assert config.is_time_in_evaluation_times(
            t_minus_tol, true_eval_times, tol=expected_tol
        )
        obs(config, t_minus_tol, ghz_state, ham, results)
        assert results.get_result_times(obs) == [t_, t_minus_tol]
        assert results.get_result(obs, t_minus_tol) == ghz_state

        t_plus_tol = t_ + expected_tol
        assert t_plus_tol > 1.0  # ie it's not an evaluation time
        assert not config.is_time_in_evaluation_times(
            t_plus_tol, true_eval_times, tol=expected_tol
        )
        obs(config, t_plus_tol, ghz_state, ham, results)
        assert t_plus_tol not in results.get_result_times(obs)

    def test_state_result(self, ghz_state):
        obs = StateResult()
        assert obs.apply(state=ghz_state) == ghz_state

    @pytest.mark.parametrize("p_false_pos", [None, 0.4])
    @pytest.mark.parametrize("p_false_neg", [None, 0.3])
    @pytest.mark.parametrize("one_state", [None, "g"])
    @pytest.mark.parametrize("num_shots", [None, 100])
    def test_bitstrings(
        self,
        config: EmulationConfig,
        ghz_state: QutipState,
        num_shots,
        one_state,
        p_false_pos,
        p_false_neg,
    ):
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            BitStrings(num_shots=0)
        kwargs = {}
        if num_shots:
            kwargs["num_shots"] = num_shots
        obs = BitStrings(one_state=one_state, **kwargs)
        assert obs.tag == "bitstrings"
        noise_model = pulser.NoiseModel(
            p_false_pos=p_false_pos, p_false_neg=p_false_neg
        )
        config.noise_model = noise_model
        assert config.noise_model.noise_types == (
            ("SPAM",) if p_false_pos or p_false_neg else ()
        )
        np.random.seed(123)
        expected_shots = num_shots or obs.num_shots
        expected_counts = ghz_state.sample(
            num_shots=expected_shots,
            one_state=one_state or ghz_state.infer_one_state(),
            p_false_pos=p_false_pos or 0,
            p_false_neg=p_false_neg or 0,
        )
        np.random.seed(123)
        counts = obs.apply(config=config, state=ghz_state)
        assert isinstance(counts, Counter)
        assert counts.total() == expected_shots
        if noise_model == pulser.NoiseModel():
            assert set(counts) == {"000", "111"}
        assert counts == expected_counts

    @pytest.mark.parametrize("one_state", [None, "r", "g"])
    def test_correlation_matrix_and_occupation(
        self, ghz_state, ham, one_state
    ):
        corr = CorrelationMatrix(one_state=one_state)
        assert corr.tag == "correlation_matrix"
        occ = Occupation(one_state=one_state)
        assert occ.tag == "occupation"
        expected_corr_matrix = np.full((3, 3), 0.5)
        np.testing.assert_allclose(
            corr.apply(state=ghz_state, hamiltonian=ham), expected_corr_matrix
        )
        np.testing.assert_allclose(
            occ.apply(state=ghz_state, hamiltonian=ham),
            expected_corr_matrix.diagonal(),
        )

        ggg_state = QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"ggg": 1.0}
        )
        expected_corr_matrix = np.ones((3, 3)) * int(one_state == "g")
        np.testing.assert_allclose(
            corr.apply(state=ggg_state, hamiltonian=ham), expected_corr_matrix
        )
        np.testing.assert_allclose(
            occ.apply(state=ggg_state, hamiltonian=ham),
            expected_corr_matrix.diagonal(),
        )

        ggr_state = QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"ggr": 1.0}
        )
        if one_state == "g":
            expected_corr_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        else:
            expected_corr_matrix = np.zeros((3, 3))
            expected_corr_matrix[2, 2] = 1
        np.testing.assert_allclose(
            corr.apply(state=ggr_state, hamiltonian=ham), expected_corr_matrix
        )
        np.testing.assert_allclose(
            occ.apply(state=ggr_state, hamiltonian=ham),
            expected_corr_matrix.diagonal(),
        )

    @pytest.fixture
    def zzz(self):
        return QutipOperator.from_operator_repr(
            eigenstates=("r", "g"),
            n_qudits=3,
            operations=[(1.0, [({"rr": 1.0, "gg": -1.0}, {0, 1, 2})])],
        )

    def test_energy_observables(self, ghz_state, ham, zzz):
        energy = Energy()
        assert energy.tag == "energy"
        var = EnergyVariance()
        assert var.tag == "energy_variance"
        energy2 = SecondMomentOfEnergy()
        assert energy2.tag == "second_moment_of_energy"
        assert np.isclose(energy.apply(state=ghz_state, hamiltonian=ham), 1.0)
        assert np.isclose(energy2.apply(state=ghz_state, hamiltonian=ham), 1.0)
        assert np.isclose(var.apply(state=ghz_state, hamiltonian=ham), 0.0)

        assert np.isclose(energy.apply(state=ghz_state, hamiltonian=zzz), 0.0)
        assert np.isclose(energy2.apply(state=ghz_state, hamiltonian=zzz), 1.0)
        assert np.isclose(var.apply(state=ghz_state, hamiltonian=zzz), 1.0)

        custom_op = QutipOperator.from_operator_repr(
            eigenstates=("r", "g"),
            n_qudits=3,
            operations=[(1.0, [({"gg": -1}, {0, 1, 2})])],
        )
        assert np.isclose(
            energy.apply(state=ghz_state, hamiltonian=custom_op), -0.5
        )
        assert np.isclose(
            energy2.apply(state=ghz_state, hamiltonian=custom_op), 0.5
        )
        assert np.isclose(
            var.apply(state=ghz_state, hamiltonian=custom_op), 0.25
        )

    def test_expectation(self, ghz_state, ham, zzz):
        with pytest.raises(
            TypeError, match="'operator' must be an Operator instance"
        ):
            Expectation(ham.to_qobj())
        h_exp = Expectation(ham)
        assert h_exp.tag == "expectation"
        assert h_exp.apply(state=ghz_state) == ham.expect(ghz_state)
        z_exp = Expectation(zzz, tag_suffix="zzz")
        assert z_exp.tag == "expectation_zzz"
        assert z_exp.apply(state=ghz_state) == zzz.expect(ghz_state)

    def test_fidelity(self, ghz_state):
        with pytest.raises(
            TypeError, match="'state' must be a State instance"
        ):
            Fidelity(ghz_state.to_qobj())

        fid_ggg = Fidelity(
            QutipState.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"ggg": 1.0}
            ),
            tag_suffix="ggg",
        )
        assert fid_ggg.tag == "fidelity_ggg"
        assert np.isclose(fid_ggg.apply(state=ghz_state), 0.5)

        fid_ghz = Fidelity(ghz_state)
        assert fid_ghz.tag == "fidelity"
        assert np.isclose(fid_ghz.apply(state=ghz_state), 1.0)
