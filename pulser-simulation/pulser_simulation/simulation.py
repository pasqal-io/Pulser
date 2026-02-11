# Copyright 2020 Pulser Development Team
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
"""Defines the QutipEmulator, used to simulate a Sequence or its samples."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, NamedTuple, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import qutip
from numpy.typing import ArrayLike

import pulser.sampler as sampler
from pulser import Sequence
from pulser._hamiltonian_data import (
    HamiltonianData,
    has_shot_to_shot_except_spam,
)
from pulser.channels.base_channel import States
from pulser.devices._device_datacls import BaseDevice
from pulser.noise_model import NoiseModel
from pulser.register.base_register import BaseRegister
from pulser.result import SampledResult
from pulser.sampler.samples import ChannelSamples, SequenceSamples
from pulser.sequence._seq_drawer import draw_samples
from pulser_simulation.hamiltonian import Hamiltonian
from pulser_simulation.qutip_result import QutipResult
from pulser_simulation.simconfig import SimConfig
from pulser_simulation.simresults import (
    CoherentResults,
    NoisyResults,
    SimulationResults,
)


class HamiltonianWithReps(NamedTuple):
    """A Hamiltonian and the number of times it should be simulated."""

    hamiltonian: Hamiltonian
    reps: int


def _has_stochastic_noise(noise_model: NoiseModel) -> bool:
    return has_shot_to_shot_except_spam(noise_model) or (
        "SPAM" in noise_model.noise_types and noise_model.state_prep_error != 0
    )


class Solver(str, Enum):
    """QuTiP solver selection.

    If the noise model has no effective noise,
      ``qutip.sesolve`` is used (this setting is ignored).
    If the noise model has effective noise:
        - ``DEFAULT``: auto-select ``qutip.mcsolve``
          for stochastic noise, else ``qutip.mesolve``
        - ``MESOLVER``: master-equation solver ``qutip.mesolve``
        - ``MCSOLVER``: Monte-Carlo solver ``qutip.mcsolve``
    """

    DEFAULT = "default"
    MESOLVER = "MasterEquation"
    MCSOLVER = "MonteCarlo"


class QutipEmulator:
    r"""Emulator of a pulse sequence using QuTiP.

    Args:
        sampled_seq: A pulse sequence samples used in the emulation.
        register: The register associating coordinates to the qubits targeted
            by the pulses within the samples.
        device: The device specifications used in the emulation. Register and
            samples have to satisfy its constraints.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation. *Deprecated
            since v1.6, please use ``noise_model`` instead.*
        evaluation_times: Choose between:

            - "Full": The times are set to be the ones used to define the
              Hamiltonian to the solver.

            - "Minimal": The times are set to only include initial and final
              times.

            - An ArrayLike object of times in µs if you wish to only include
              those specific times.

            - A float to act as a sampling rate for the resulting state.
        noise_model: The noise model for the simulation. Replaces and should
            be preferred over 'config'.
        solver: QuTiP solver selection. If the noise model has no collapse
            operators (i.e. no dephasing/relaxation/depolarizing/eff_noise
            terms), the simulation uses qutip.sesolve and the solver setting
            is ignored. If collapse operators are present, then:

            - ``Solver.DEFAULT``: auto-select ``qutip.mcsolve``
              for stochastic noise, otherwise ``qutip.mesolve``.

            - ``Solver.MCSOLVER``: use the Monte-Carlo
              solver ``qutip.mcsolve``.

            - ``Solver.MESOLVER``: use the master-equation
              solver ``qutip.mesolve``.
        n_trajectories: The number of trajectories to average over when the
            emulation includes stochastic noise or is using a Monte Carlo
            solver. If defined, takes precedence over the (now deprecated)
            `noise_model.runs` or `config.runs`.
    """

    def __init__(
        self,
        sampled_seq: SequenceSamples,
        register: BaseRegister,
        device: BaseDevice,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
        noise_model: NoiseModel | None = None,
        solver: Solver = Solver.DEFAULT,
        n_trajectories: int | None = None,
    ) -> None:
        """Instantiates a QutipEmulator object."""
        # Initializing the samples obj
        if not isinstance(sampled_seq, SequenceSamples):
            raise TypeError(
                "The provided sequence has to be a valid "
                "SequenceSamples instance."
            )
        if sampled_seq.max_duration == 0:
            raise ValueError("SequenceSamples is empty.")
        # Check compatibility of register and device
        self._sampling_rate = sampling_rate
        device.validate_register(register)
        self._register = register
        self.solver = solver
        # Check compatibility of samples and device:
        if sampled_seq._slm_mask.end > 0 and not device.supports_slm_mask:
            raise ValueError(
                "Samples use SLM mask but device does not have one."
            )
        if not sampled_seq.used_bases <= device.supported_bases:
            raise ValueError(
                "Bases used in samples should be supported by device."
            )
        # Check compatibility of masked samples and register
        if not sampled_seq._slm_mask.targets <= set(register.qubit_ids):
            raise ValueError(
                "The ids of qubits targeted in SLM mask"
                " should be defined in register."
            )

        self._tot_duration = sampled_seq.max_duration
        self.samples_obj = sampled_seq.extend_duration(self._tot_duration + 1)
        self._n_trajectories = n_trajectories

        # Testing sampling
        if not (0 < sampling_rate <= 1.0):
            raise ValueError(
                "The sampling rate (`sampling_rate` = "
                f"{sampling_rate}) must be greater than 0 and "
                "less than or equal to 1."
            )
        if int(self._tot_duration * sampling_rate) < 4:
            raise ValueError(
                "`sampling_rate` is too small, less than 4 data points."
            )

        if noise_model is not None and config is not None:
            raise ValueError(
                "'noise_model' and 'config' cannot both be provided to "
                "'QutipEmulator'. Please provide just a 'noise_model'."
            )
        if config is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(
                    "Supplying a 'SimConfig' to QutipEmulator has been "
                    "deprecated. Please instantiate with a 'NoiseModel' "
                    "instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            noise_model = config.to_noise_model()
        if not noise_model:
            noise_model = NoiseModel()

        self._hamiltonian_data = HamiltonianData(
            self.samples_obj,
            register,
            device,
            noise_model,
            self._get_n_trajectories(noise_model, check_value=True),
        )
        # I don't like this, since the iterator is lost, but I don't
        # want to write logic to store and invalidate the iterator either
        # (e.g. if doing run multiple times).
        self._current_hamiltonian = next(self._hamiltonians).hamiltonian
        # Initializing evaluation times
        self._eval_times_array: np.ndarray
        self.set_evaluation_times(evaluation_times)

        if self.samples_obj._measurement:
            self._meas_basis = self.samples_obj._measurement
        else:
            if "all" in self.basis_name:
                self._meas_basis = "digital"
            else:
                self._meas_basis = self.basis_name.replace("_with_error", "")
        self.set_initial_state("all-ground")

    def _get_n_trajectories(
        self, noise_model: NoiseModel, check_value: bool
    ) -> int | None:
        n_trajectories = (
            self._n_trajectories
            if self._n_trajectories is not None
            else noise_model.runs
        )
        if (
            check_value
            and _has_stochastic_noise(noise_model)
            and n_trajectories is None
        ):
            raise ValueError(
                "'n_trajectories' must be defined when the NoiseModel contains"
                " stochastic noise, which is the case for the given noise "
                f"model: {noise_model!r}"
            )
        return n_trajectories

    @property
    def n_trajectories(self) -> int | None:
        """The number of trajectories to average over (when applicable)."""
        return self._get_n_trajectories(self.noise_model, check_value=False)

    @property
    def device(self) -> BaseDevice:
        """The device being simulated."""
        return self._hamiltonian_data.device

    @property
    def _noiseless_hamiltonian(self) -> Hamiltonian:
        return self._get_noiseless_hamiltonian(False)

    @lru_cache(maxsize=2)
    def _get_noiseless_hamiltonian(self, leakage: bool) -> Hamiltonian:
        """Get the noiseless Hamiltonian.

        Args:
            leakage: whether to include the leakage state in the basis.
        """
        if leakage:
            eff_rate = (0.0,)
            eff_ops = (np.zeros((3, 3)),)
            noise = NoiseModel(
                eff_noise_opers=eff_ops,  # feed zero op because needed
                eff_noise_rates=eff_rate,
                with_leakage=leakage,
            )
        else:
            noise = NoiseModel()

        noiseless_data = HamiltonianData(
            self.samples_obj,
            self._register,
            self.device,
            noise,
            n_trajectories=1,
        )
        return Hamiltonian(
            noiseless_data.samples,
            noiseless_data.noise_trajectories[0].trajectory,
            noiseless_data.basis_data,
            noiseless_data.lindblad_data,
            self._sampling_rate,
        )

    @property
    def _hamiltonians(self) -> Iterator[HamiltonianWithReps]:
        for traj, noisy_samples, reps in self._hamiltonian_data.noisy_samples:
            yield HamiltonianWithReps(
                Hamiltonian(
                    noisy_samples,
                    traj,
                    self._hamiltonian_data.basis_data,
                    self._hamiltonian_data.lindblad_data,
                    self._sampling_rate,
                ),
                reps,
            )

    @property
    def sampling_times(self) -> np.ndarray:
        """The times at which hamiltonian is sampled."""
        return self._noiseless_hamiltonian.sampling_times

    @property
    def dim(self) -> int:
        """The dimension of the basis."""
        return self._hamiltonian_data.basis_data.dim

    @property
    def basis_name(self) -> str:
        """The name of the basis."""
        return self._hamiltonian_data.basis_data.basis_name

    @property
    def basis(self) -> dict[States, Any]:
        """The basis in which result is expressed."""
        return self._current_hamiltonian.basis

    @property
    def noise_model(self) -> NoiseModel:
        """The current NoiseModel being used."""
        return self._hamiltonian_data.noise_model

    @property
    def config(self) -> SimConfig:
        """The current configuration, as a SimConfig instance."""
        return SimConfig.from_noise_model(self._hamiltonian_data.noise_model)

    @property
    def total_duration_ns(self) -> int:
        """The total duration of the sequence, in ns."""
        return self._tot_duration

    def set_config(self, cfg: SimConfig) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Warning:
            This method has been deprecated since v1.6.
            Please prefer instantiating a new ``QutipEmulator`` with a custom
            ``noise_model`` instead.

        Args:
            cfg: New configuration.
        """
        warnings.warn(
            "Supplying a 'SimConfig' to QutipEmulator has been deprecated."
            " Please instantiate with a 'NoiseModel' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(cfg, SimConfig):
            raise ValueError(f"Object {cfg} is not a valid `SimConfig`.")
        not_supported = (
            set(cfg.noise)
            - cfg.supported_noises[
                self._hamiltonian_data.basis_data.interaction_type
            ]
        )
        if not_supported:
            v = self._hamiltonian_data.basis_data.interaction_type
            raise NotImplementedError(
                f"Interaction mode '{v}' "
                "does not support simulation of noise types:"
                f"{', '.join(not_supported)}."
            )
        former_dim = self.dim
        former_basis = self.basis
        noise_model = cfg.to_noise_model()
        self._hamiltonian_data = HamiltonianData(
            self.samples_obj,
            self._register,
            self.device,
            noise_model,
            self._get_n_trajectories(noise_model, check_value=True),
        )
        self._current_hamiltonian = next(self._hamiltonians).hamiltonian
        if self.dim == former_dim:
            self.set_initial_state(self._initial_state)
            return
        if self._initial_state != qutip.tensor(
            [
                former_basis[
                    (
                        "u"
                        if self._hamiltonian_data.basis_data.interaction_type
                        == "XY"
                        else "g"
                    )
                ]
                for _ in range(self._hamiltonian_data.n_qudits)
            ]
        ):
            warnings.warn(
                "Current initial state's dimension does not match new"
                " dimensions. Setting it to 'all-ground'."
            )
        self.set_initial_state("all-ground")

    def add_config(self, config: SimConfig) -> None:
        """Updates the current configuration with parameters of another one.

        Warning:
            This method has been deprecated since v1.6.
            Please prefer instantiating a new ``QutipEmulator`` with a custom
            ``noise_model`` instead.

        Mostly useful when dealing with multiple noise types in different
        configurations and wanting to merge these configurations together.
        Adds simulation parameters to noises that weren't available in the
        former SimConfig. Noises specified in both SimConfigs will keep
        former noise parameters.

        Args:
            config: SimConfig to retrieve parameters from.
        """
        warnings.warn(
            "Supplying a 'SimConfig' to QutipEmulator has been deprecated."
            " Please instantiate with a 'NoiseModel' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(config, SimConfig):
            raise ValueError(f"Object {config} is not a valid `SimConfig`")

        not_supported = (
            set(config.noise)
            - config.supported_noises[
                self._hamiltonian_data.basis_data.interaction_type
            ]
        )
        if not_supported:
            v = self._hamiltonian_data.basis_data.interaction_type
            raise NotImplementedError(
                f"Interaction mode '{v}' "
                "does not support simulation of noise types: "
                f"{', '.join(not_supported)}."
            )
        noise_model = config.to_noise_model()
        old_noise_set = set(self._hamiltonian_data.noise_model.noise_types)
        new_noise_set = old_noise_set.union(noise_model.noise_types)
        diff_noise_set = new_noise_set - old_noise_set
        # Create temporary param_dict to add noise parameters:
        param_dict: dict[str, Any] = asdict(self._hamiltonian_data.noise_model)
        relevant_params = NoiseModel._find_relevant_params(
            diff_noise_set,
            noise_model.state_prep_error,
            noise_model.amp_sigma,
            noise_model.laser_waist,
        )
        for param in relevant_params:
            param_dict[param] = getattr(noise_model, param)
        # set config with the new parameters:
        param_dict.pop("noise_types")
        self.set_config(SimConfig.from_noise_model(NoiseModel(**param_dict)))

    def show_config(self, solver_options: bool = False) -> None:
        """Shows current configuration."""
        print(self.config.__str__(solver_options))

    def reset_config(self) -> None:
        """Resets configuration to default."""
        self.set_config(SimConfig())

    @property
    def initial_state(self) -> qutip.Qobj:
        """The initial state of the simulation."""
        return self._initial_state

    def set_initial_state(
        self, state: Union[str, np.ndarray, qutip.Qobj]
    ) -> None:
        """Sets the initial state of the simulation.

        Args:
            state: The initial state.
                Choose between:

                - "all-ground" for all atoms in ground state
                - An ArrayLike with a shape compatible with the system
                - A Qobj object
        """
        self._initial_state: qutip.Qobj
        if isinstance(state, str) and state == "all-ground":
            v = self._hamiltonian_data.basis_data.interaction_type
            self._initial_state = qutip.tensor(
                [
                    self.basis[("u" if v == "XY" else "g")]
                    for _ in range(self._hamiltonian_data.n_qudits)
                ]
            )
        else:
            state = cast(Union[np.ndarray, qutip.Qobj], state)
            shape = state.shape[0]
            legal_shape = (
                self._hamiltonian_data.basis_data.dim
                ** self._hamiltonian_data.n_qudits
            )
            legal_dims = [
                [self._hamiltonian_data.basis_data.dim]
                * self._hamiltonian_data.n_qudits,
                [1] * self._hamiltonian_data.n_qudits,
            ]
            if shape != legal_shape:
                raise ValueError(
                    "Incompatible shape of initial state."
                    + f"Expected {legal_shape}, got {shape}."
                )
            self._initial_state = (
                qutip.Qobj(state, dims=legal_dims).unit().to("CSR")
            )

    @property
    def evaluation_times(self) -> np.ndarray:
        """The times at which the results of this simulation are returned."""
        return np.array(self._eval_times_array)

    def set_evaluation_times(
        self, value: Union[str, ArrayLike, float]
    ) -> None:
        """Sets times at which the results of this simulation are returned.

        Args:
            value: Choose between:

                - "Full": The times are set to be the ones used to define the
                  Hamiltonian to the solver.

                - "Minimal": The times are set to only include initial and
                  final times.

                - An ArrayLike object of times in µs if you wish to only
                  include those specific times.

                - A float to act as a sampling rate for the resulting state.
        """
        if isinstance(value, str):
            if value == "Full":
                eval_times = np.copy(
                    self._noiseless_hamiltonian.sampling_times
                )
            elif value == "Minimal":
                eval_times = np.array([])
            else:
                raise ValueError(
                    "Wrong evaluation time label. It should "
                    "be `Full`, `Minimal`, an array of times or"
                    + " a float between 0 and 1."
                )
        elif isinstance(value, float):
            if value > 1 or value <= 0:
                raise ValueError(
                    "evaluation_times float must be between 0 and 1."
                )
            indices = np.linspace(
                0,
                len(self._noiseless_hamiltonian.sampling_times) - 1,
                int(value * len(self._noiseless_hamiltonian.sampling_times)),
                dtype=int,
            )
            # Note: if `value` is very small `eval_times` is an empty list:
            eval_times = self._noiseless_hamiltonian.sampling_times[indices]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if np.max(value, initial=0) > self._tot_duration * 1e-3:
                raise ValueError(
                    "Provided evaluation-time list extends "
                    "further than sequence duration."
                )
            if np.min(value, initial=0) < 0:
                raise ValueError(
                    "Provided evaluation-time list contains "
                    "negative values."
                )
            eval_times = np.array(value)
        else:
            raise ValueError(
                "Wrong evaluation time label. It should "
                "be `Full`, `Minimal`, an array of times or a "
                + "float between 0 and 1."
            )
        # Ensure 0 and final time are included:
        self._eval_times_array = np.union1d(
            eval_times, [0.0, self._tot_duration * 1e-3]
        )
        self._eval_times_instruction = value

    def build_operator(self, operations: Union[list, tuple]) -> qutip.Qobj:
        """Creates an operator with non-trivial actions on some qubits.

        Takes as argument a list of tuples ``[(operator_1, qubits_1),
        (operator_2, qubits_2)...]``. Returns the operator given by the tensor
        product of {``operator_i`` applied on ``qubits_i``} and Id on the rest.
        ``(operator, 'global')`` returns the sum for all ``j`` of operator
        applied at ``qubit_j`` and identity elsewhere.

        Example for 4 qubits: ``[(Z, [1, 2]), (Y, [3])]`` returns `ZZYI`
        and ``[(X, 'global')]`` returns `XIII + IXII + IIXI + IIIX`

        Args:
            operations: List of tuples `(operator, qubits)`.
                `operator` can be a ``qutip.Quobj`` or a string key for
                ``self.op_matrix``. `qubits` is the list on which operator
                will be applied. The qubits can be passed as their
                index or their label in the register.

        Returns:
            The final operator.
        """
        return self._current_hamiltonian.build_operator(operations)

    def get_hamiltonian(
        self, time: float, noiseless: bool = False
    ) -> qutip.Qobj:
        r"""Get the Hamiltonian created from the sequence at a fixed time.

        Note:
            The whole Hamiltonian is divided by :math:`\hbar`, so its
            units are rad/µs.

        Args:
            time: The specific time at which we want to extract the
                Hamiltonian (in ns).
            noiseless: If True, returns the Hamiltonian without noise.

        Returns:
            A new Qobj for the Hamiltonian with coefficients
            extracted from the effective sequence (determined by
            `self.sampling_rate`) at the specified time.
        """
        if time > self._tot_duration:
            raise ValueError(
                f"Provided time (`time` = {time}) must be "
                "less than or equal to the sequence duration "
                f"({self._tot_duration})."
            )
        if time < 0:
            raise ValueError(
                f"Provided time (`time` = {time}) must be "
                "greater than or equal to 0."
            )

        if noiseless:
            return self._noiseless_hamiltonian._hamiltonian(time / 1000)

        return self._current_hamiltonian._hamiltonian(
            time / 1000
        )  # Creates new Qutip.Qobj

    @staticmethod
    def _get_min_variation(ch_sample: ChannelSamples) -> int:
        """Compute minimum variations of samples.

        This is used as default value for the max_step option.
        """
        end_point = ch_sample.duration - 1
        min_variations: list[int] = []
        for sample in (
            ch_sample.amp.as_array(detach=True),
            ch_sample.det.as_array(detach=True),
        ):
            min_variations.append(
                int(
                    np.min(
                        np.diff(
                            np.nonzero(np.diff(sample)),
                            prepend=-1,
                            append=end_point,
                        )
                    )
                )
            )

        return min(min_variations)

    def _run_solver(
        self,
        hamiltonian: Hamiltonian,
        progress_bar: bool = False,
        mcsolve_ntraj: int = 1,
        **options: Any,
    ) -> CoherentResults:
        """Returns CoherentResults: Object containing evolution results."""
        # Decide if progress bar will be fed to QuTiP solver
        if progress_bar is True:
            options["progress_bar"] = True
        elif (progress_bar is False) or (progress_bar is None):
            options["progress_bar"] = ""
        else:
            raise ValueError("`progress_bar` must be a bool.")

        if not isinstance(self.solver, Solver):
            allowed_str = ", ".join(s.value for s in Solver)
            raise ValueError(
                f"Invalid solver '{self.solver}'. "
                f"Allowed solvers are: {allowed_str}."
            )

        solver_fn: Callable[..., Any] = qutip.sesolve

        if len(hamiltonian.lindblad_data.local_collapse_ops) > 0:
            if self.solver == Solver.DEFAULT:
                solver_fn = (
                    qutip.mcsolve
                    if _has_stochastic_noise(self.noise_model)
                    else qutip.mesolve
                )
            else:
                solver_fn = {
                    Solver.MCSOLVER: qutip.mcsolve,
                    Solver.MESOLVER: qutip.mesolve,
                }[self.solver]

        if solver_fn in (qutip.mesolve, qutip.sesolve):
            options["normalize_output"] = False

        extra_kwargs: dict[str, Any] = {}
        if solver_fn in (qutip.mesolve, qutip.mcsolve):
            extra_kwargs["c_ops"] = hamiltonian._collapse_ops
            if solver_fn is qutip.mcsolve:
                extra_kwargs["ntraj"] = mcsolve_ntraj

        result = solver_fn(
            hamiltonian._hamiltonian,
            self.initial_state,
            self._eval_times_array,
            **extra_kwargs,
            options=options,
        )

        results = [
            QutipResult(
                tuple(self._hamiltonian_data.register.qubits),
                self._meas_basis,
                state,
                self._meas_basis in self.basis_name,
                evaluation_time=t / self._tot_duration * 1e3,
            )
            for state, t in zip(result.states, self._eval_times_array)
        ]

        meas_errors = (
            {
                "epsilon": self.noise_model.p_false_pos,
                "epsilon_prime": self.noise_model.p_false_neg,
            }
            if "SPAM" in self.noise_model.noise_types
            else None
        )

        return CoherentResults(
            results,
            self._hamiltonian_data.n_qudits,
            self.basis_name,
            self._eval_times_array,
            self._meas_basis,
            meas_errors,
        )

    def _validate_options(self, options: Any) -> None:
        options.setdefault(
            "max_step",
            min(
                self._get_min_variation(ch_sample)
                for ch_sample in self.samples_obj.samples_list
            )
            / 1000,
        )

        options.setdefault(
            "nsteps", max(1000, self._tot_duration // options["max_step"])
        )

        if "SPAM" in self.noise_model.noise_types:
            v = self._hamiltonian_data.basis_data.interaction_type
            if (
                self.noise_model.state_prep_error > 0
                and self.initial_state
                != qutip.tensor(
                    [
                        self.basis[("u" if v == "XY" else "g")]
                        for _ in range(self._hamiltonian_data.n_qudits)
                    ]
                )
            ):
                raise NotImplementedError(
                    "Can't combine state preparation errors with an initial "
                    "state different from the ground."
                )

    # Run Simulation Evolution using Qutip
    def run(
        self,
        progress_bar: bool = False,
        **options: Any,
    ) -> SimulationResults:
        """Simulates the sequence using QuTiP's solvers.

        Will return NoisyResults if the noise in the SimConfig requires it.
        Otherwise will return CoherentResults.

        Args:
            progress_bar: If True, the progress bar of QuTiP's
                solver will be shown. If None or False, no text appears.
            options: Given directly to the Qutip Solver. If specified, will
                override SimConfig solver_options. If no `max_step` value is
                provided, an automatic one is calculated from the `Sequence`'s
                schedule (half of the shortest duration among pulses and
                delays).
                Refer to the QuTiP docs_ for an overview of the parameters.

                .. _docs: https://bit.ly/3il9A2u
        """
        self._validate_options(options)

        if not _has_stochastic_noise(self.noise_model):
            print("Emulating Trajectory 1/1")
            # A single run is needed, regardless of self.config.runs
            return self._run_solver(
                self._current_hamiltonian,
                progress_bar,
                mcsolve_ntraj=self.n_trajectories or 1,
                **options,
            )

        # Will return NoisyResults
        total_count = np.array([Counter() for _ in self._eval_times_array])

        for cleanres_noisyseq, reps in self._noisy_runs(
            progress_bar=progress_bar, **options
        ):
            total_count += np.array(
                [
                    cleanres_noisyseq.sample_state(
                        t,
                        n_samples=self.noise_model.samples_per_run * reps,
                    )
                    for t in self._eval_times_array
                ]
            )

        n_measures = (
            cast(int, self.n_trajectories) * self.noise_model.samples_per_run
        )
        results = [
            SampledResult(
                tuple(self._hamiltonian_data.register.qubits),
                self._meas_basis,
                total_count[ind],
                evaluation_time=t / self._tot_duration * 1e3,
            )
            for ind, t in enumerate(self._eval_times_array)
        ]
        return NoisyResults(
            results,
            self._hamiltonian_data.n_qudits,
            self.basis_name,
            self._eval_times_array,
            n_measures,
        )

    def _noisy_runs(
        self, progress_bar: bool, **options: Any
    ) -> Iterator[tuple[SimulationResults, int]]:
        n_trajectories = self.n_trajectories
        traj_nb = 0
        for i, (ham, reps) in enumerate(self._hamiltonians):
            if reps == 1:
                print(f"Emulating Trajectory {traj_nb+1}/{n_trajectories}")
            else:
                print(
                    f"Emulating Trajectories [{traj_nb+1} - {traj_nb+reps}]"
                    f"/{n_trajectories}"
                )
            self._current_hamiltonian = ham
            traj_nb += reps
            # Yield CoherentResults instance from sequence with added noise:
            yield self._run_solver(ham, progress_bar, **options), reps

    def draw(
        self,
        draw_phase_area: bool = False,
        draw_phase_shifts: bool = False,
        draw_phase_curve: bool = False,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the samples of a sequence of operations used for simulation.

        Args:
            draw_phase_area: Whether phase and area values need
                to be shown as text on the plot, defaults to False.
            draw_phase_shifts: Whether phase shift and reference
                information should be added to the plot, defaults to False.
            draw_phase_curve: Draws the changes in phase in its own curve
                (ignored if the phase doesn't change throughout the channel).
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        See Also:
            Sequence.draw(): Draws the sequence in its current state.
        """
        draw_samples(
            self.samples_obj,
            self._register,
            self._sampling_rate,
            draw_phase_area=draw_phase_area,
            draw_phase_shifts=draw_phase_shifts,
            draw_phase_curve=draw_phase_curve,
        )
        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
        with_modulation: bool = False,
        noise_model: NoiseModel | None = None,
        solver: Solver = Solver.DEFAULT,
        n_trajectories: int | None = None,
    ) -> QutipEmulator:
        r"""Simulation of a pulse sequence using QuTiP.

        Args:
            sequence: An instance of a Pulser Sequence that we
                want to simulate.
            sampling_rate: The fraction of samples that we wish to
                extract from the pulse sequence to simulate. Has to be a
                value between 0.05 and 1.0.
            config: Configuration to be used for this simulation. *Deprecated
                since v1.6, use 'noise_model' instead.*
            evaluation_times: Choose between:

                - "Full": The times are set to be the ones used to define the
                  Hamiltonian to the solver.

                - "Minimal": The times are set to only include initial and
                  final times.

                - An ArrayLike object of times in µs if you wish to only
                  include those specific times.

                - A float to act as a sampling rate for the resulting state.
            with_modulation: Whether to simulate the sequence with the
                programmed input or the expected output.
            noise_model: The noise model for the simulation. Replaces and
                should be preferred over 'config'.
            solver: QuTiP solver selection. If the noise model has no collapse
                operators (i.e. no dephasing/relaxation/depolarizing/eff_noise
                terms), the simulation uses qutip.sesolve and the solver
                setting is ignored. If collapse operators are present, then:

                - ``Solver.DEFAULT``: auto-select ``qutip.mcsolve``
                  for stochastic noise, otherwise ``qutip.mesolve``.

                - ``Solver.MCSOLVER``: use the Monte-Carlo
                  solver ``qutip.mcsolve``.

                - ``Solver.MESOLVER``: use the master-equation
                  solver ``qutip.mesolve``.
            n_trajectories: The number of trajectories to average over when the
                emulation includes stochastic noise or is using a Monte Carlo
                solver. If defined, takes precedence over the (now deprecated)
                `noise_model.runs` or `config.runs`.
        """
        if not isinstance(sequence, Sequence):
            raise TypeError(
                "The provided sequence has to be a valid "
                "pulser.Sequence instance."
            )
        if sequence.is_parametrized() or sequence.is_register_mappable():
            raise ValueError(
                "The provided sequence needs to be built to be simulated. Call"
                " `Sequence.build()` with the necessary parameters."
            )
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(
            sequence._schedule[x][-1].tf == 0
            for x in sequence.declared_channels
        ):
            raise ValueError(
                "No instructions given for the channels in the sequence."
            )
        if with_modulation and sequence._slm_mask_targets:
            raise NotImplementedError(
                "Simulation of sequences combining an SLM mask and output "
                "modulation is not supported."
            )
        return cls(
            sampler.sample(
                sequence,
                modulation=with_modulation,
                extended_duration=sequence.get_duration(
                    include_fall_time=with_modulation
                ),
            ),
            sequence.register,
            sequence.device,
            sampling_rate,
            config,
            evaluation_times,
            noise_model=noise_model,
            solver=solver,
            n_trajectories=n_trajectories,
        )
