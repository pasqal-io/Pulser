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
"""Contains the Simulation class, used for simulation of a Sequence."""

from __future__ import annotations

import itertools
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import qutip
from numpy.typing import ArrayLike

import pulser.sampler as sampler
from pulser import Pulse, Sequence
from pulser.register import QubitId
from pulser.sampler.samples import _TargetSlot
from pulser.sequence._seq_drawer import draw_sequence
from pulser_simulation.simconfig import SimConfig
from pulser_simulation.simresults import (
    CoherentResults,
    NoisyResults,
    SimulationResults,
)

SUPPORTED_NOISE = {
    "ising": {
        "dephasing",
        "doppler",
        "amplitude",
        "SPAM",
        "depolarizing",
        "eff_noise",
    },
    "XY": {"SPAM"},
}


class Simulation:
    r"""Simulation of a pulse sequence using QuTiP.

    Args:
        sequence: An instance of a Pulser Sequence that we
            want to simulate.
        sampling_rate: The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
        evaluation_times: Choose between:

            - "Full": The times are set to be the ones used to define the
              Hamiltonian to the solver.

            - "Minimal": The times are set to only include initial and final
              times.

            - An ArrayLike object of times in µs if you wish to only include
              those specific times.

            - A float to act as a sampling rate for the resulting state.
        with_modulation: Whether to simulated the sequence with the programmed
            input or the expected output.
    """

    def __init__(
        self,
        sequence: Sequence,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
        with_modulation: bool = False,
    ) -> None:
        """Instantiates a Simulation object."""
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
        self._seq = sequence
        self._interaction = "XY" if self._seq._in_xy else "ising"
        self._qdict = self._seq.qubit_info
        self._size = len(self._qdict)
        self._modulated = with_modulation
        if self._modulated and sequence._slm_mask_targets:
            raise NotImplementedError(
                "Simulation of sequences combining an SLM mask and output "
                "modulation is not supported."
            )
        self._tot_duration = self._seq.get_duration(
            include_fall_time=self._modulated
        )
        self.samples_obj = sampler.sample(
            self._seq,
            modulation=self._modulated,
            # The samples are extended by 1 to improve the ODE
            # solver convergence
            extended_duration=self._tot_duration + 1,
        )

        # Type hints for attributes defined outside of __init__
        self.basis_name: str
        self._config: SimConfig
        self.op_matrix: dict[str, qutip.Qobj]
        self.basis: dict[str, qutip.Qobj]
        self.dim: int
        self._eval_times_array: np.ndarray

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
        self._sampling_rate = sampling_rate
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}
        self._collapse_ops: list[qutip.Qobj] = []

        self.sampling_times = self._adapt_to_sampling_rate(
            # Include extra time step for final instruction from samples:
            np.arange(self._tot_duration + 1, dtype=np.double)
            / 1000
        )
        self.evaluation_times = evaluation_times  # type: ignore

        self._bad_atoms: dict[Union[str, int], bool] = {}
        self._doppler_detune: dict[Union[str, int], float] = {}
        # Stores the qutip operators used in building the Hamiltonian
        self.operators: dict[str, defaultdict[str, dict]] = {
            addr: defaultdict(dict) for addr in ["Global", "Local"]
        }
        # Sets the config as well as builds the hamiltonian
        self.set_config(config) if config else self.set_config(SimConfig())
        if hasattr(self._seq, "_measurement"):
            self._meas_basis = self._seq._measurement
        else:
            if self.basis_name in {"digital", "all"}:
                self._meas_basis = "digital"
            else:
                self._meas_basis = self.basis_name
        self.initial_state = "all-ground"

    @property
    def config(self) -> SimConfig:
        """The current configuration, as a SimConfig instance."""
        return self._config

    def set_config(self, cfg: SimConfig) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
        """
        if not isinstance(cfg, SimConfig):
            raise ValueError(f"Object {cfg} is not a valid `SimConfig`.")
        not_supported = set(cfg.noise) - SUPPORTED_NOISE[self._interaction]
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._interaction}' does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )
        prev_config = self.config if hasattr(self, "_config") else SimConfig()
        self._config = cfg
        if not ("SPAM" in self.config.noise and self.config.eta > 0):
            self._bad_atoms = {qid: False for qid in self._qid_index}
        if "doppler" not in self.config.noise:
            self._doppler_detune = {qid: 0.0 for qid in self._qid_index}
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()

        kraus_ops = []
        if "dephasing" in self.config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                # Go back to previous config
                self.set_config(prev_config)
                raise NotImplementedError(
                    "Cannot include dephasing noise in digital- or all-basis."
                )
            # Probability of phase (Z) flip:
            # First order in prob
            prob = self.config.dephasing_prob / 2
            n = self._size
            if prob > 0.1 and n > 1:
                warnings.warn(
                    "The dephasing model is a first-order approximation in the"
                    f" dephasing probability. p = {2*prob} is too large for "
                    "realistic results.",
                    stacklevel=2,
                )
            k = np.sqrt(prob * (1 - prob) ** (n - 1))

            self._collapse_ops += [
                np.sqrt((1 - prob) ** n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            kraus_ops.append(k * qutip.sigmaz())

        if "depolarizing" in self.config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                # Go back to previous config
                self.set_config(prev_config)
                raise NotImplementedError(
                    "Cannot include depolarizing "
                    + "noise in digital- or all-basis."
                )
            # Probability of error occurrence

            prob = self.config.depolarizing_prob / 4
            n = self._size
            if prob > 0.1 and n > 1:
                warnings.warn(
                    "The depolarizing model is a first-order approximation"
                    f" in the depolarizing probability. p = {4*prob}"
                    " is too large for realistic results.",
                    stacklevel=2,
                )

            k = np.sqrt((prob) * (1 - 3 * prob) ** (n - 1))
            self._collapse_ops += [
                np.sqrt((1 - 3 * prob) ** n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            kraus_ops.append(k * qutip.sigmax())
            kraus_ops.append(k * qutip.sigmay())
            kraus_ops.append(k * qutip.sigmaz())

        if "eff_noise" in self.config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                # Go back to previous config
                self.set_config(prev_config)
                raise NotImplementedError(
                    "Cannot include general "
                    + "noise in digital- or all-basis."
                )
            # Probability distribution of error occurences
            n = self._size
            m = len(self.config.eff_noise_opers)
            if n > 1:
                for i in range(1, m):
                    prob_i = self.config.eff_noise_probs[i]
                    if prob_i > 0.1:
                        warnings.warn(
                            "The effective noise model is a first-order"
                            " approximation in the noise probability."
                            f"p={prob_i} is large for realistic results.",
                            stacklevel=2,
                        )
                        break
            # Deriving Kraus operators
            prob_id = self.config.eff_noise_probs[0]
            self._collapse_ops += [
                np.sqrt(prob_id**n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            for i in range(1, m):
                k = np.sqrt(
                    self.config.eff_noise_probs[i] * prob_id ** (n - 1)
                )
                k_op = k * self.config.eff_noise_opers[i]
                kraus_ops.append(k_op)

        # Building collapse operators
        for operator in kraus_ops:
            self._collapse_ops += [
                self.build_operator([(operator, [qid])])
                for qid in self._qid_index
            ]

    def add_config(self, config: SimConfig) -> None:
        """Updates the current configuration with parameters of another one.

        Mostly useful when dealing with multiple noise types in different
        configurations and wanting to merge these configurations together.
        Adds simulation parameters to noises that weren't available in the
        former SimConfig. Noises specified in both SimConfigs will keep
        former noise parameters.

        Args:
            config: SimConfig to retrieve parameters from.
        """
        if not isinstance(config, SimConfig):
            raise ValueError(f"Object {config} is not a valid `SimConfig`")

        not_supported = set(config.noise) - SUPPORTED_NOISE[self._interaction]
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._interaction}' does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )

        old_noise_set = set(self.config.noise)
        new_noise_set = old_noise_set.union(config.noise)
        diff_noise_set = new_noise_set - old_noise_set
        # Create temporary param_dict to add noise parameters:
        param_dict: dict[str, Any] = asdict(self._config)
        # remove redundant `spam_dict`:
        del param_dict["spam_dict"]
        # `doppler_sigma` will be recalculated from temperature if needed:
        del param_dict["doppler_sigma"]
        # Begin populating with added noise parameters:
        param_dict["noise"] = tuple(new_noise_set)
        if "SPAM" in diff_noise_set:
            param_dict["eta"] = config.eta
            param_dict["epsilon"] = config.epsilon
            param_dict["epsilon_prime"] = config.epsilon_prime
        if "doppler" in diff_noise_set:
            param_dict["temperature"] = config.temperature
        if "amplitude" in diff_noise_set:
            param_dict["laser_waist"] = config.laser_waist
        if "dephasing" in diff_noise_set:
            param_dict["dephasing_prob"] = config.dephasing_prob
        if "depolarizing" in diff_noise_set:
            param_dict["depolarizing_prob"] = config.depolarizing_prob
        if "eff_noise" in diff_noise_set:
            param_dict["eff_noise_opers"] = config.eff_noise_opers
            param_dict["eff_noise_probs"] = config.eff_noise_probs
        param_dict["temperature"] *= 1.0e6
        # update runs:
        param_dict["runs"] = config.runs
        param_dict["samples_per_run"] = config.samples_per_run

        # set config with the new parameters:
        self.set_config(SimConfig(**param_dict))

    def show_config(self, solver_options: bool = False) -> None:
        """Shows current configuration."""
        print(self._config.__str__(solver_options))

    def reset_config(self) -> None:
        """Resets configuration to default."""
        self.set_config(SimConfig())

    @property
    def initial_state(self) -> qutip.Qobj:
        """The initial state of the simulation.

        Args:
            state: The initial state.
                Choose between:

                - "all-ground" for all atoms in ground state
                - An ArrayLike with a shape compatible with the system
                - A Qobj object
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Union[str, np.ndarray, qutip.Qobj]) -> None:
        """Sets the initial state of the simulation."""
        self._initial_state: qutip.Qobj
        if isinstance(state, str) and state == "all-ground":
            self._initial_state = qutip.tensor(
                [
                    self.basis["d" if self._interaction == "XY" else "g"]
                    for _ in range(self._size)
                ]
            )
        else:
            state = cast(Union[np.ndarray, qutip.Qobj], state)
            shape = state.shape[0]
            legal_shape = self.dim**self._size
            legal_dims = [[self.dim] * self._size, [1] * self._size]
            if shape != legal_shape:
                raise ValueError(
                    "Incompatible shape of initial state."
                    + f"Expected {legal_shape}, got {shape}."
                )
            self._initial_state = qutip.Qobj(state, dims=legal_dims)

    @property
    def evaluation_times(self) -> np.ndarray:
        """The times at which the results of this simulation are returned.

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
        return np.array(self._eval_times_array)

    @evaluation_times.setter
    def evaluation_times(self, value: Union[str, ArrayLike, float]) -> None:
        """Sets times at which the results of this simulation are returned."""
        if isinstance(value, str):
            if value == "Full":
                eval_times = np.copy(self.sampling_times)
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
                    "evaluation_times float must be between 0 " "and 1."
                )
            indices = np.linspace(
                0,
                len(self.sampling_times) - 1,
                int(value * len(self.sampling_times)),
                dtype=int,
            )
            # Note: if `value` is very small `eval_times` is an empty list:
            eval_times = self.sampling_times[indices]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if np.max(value, initial=0) > self._tot_duration / 1000:
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
            eval_times, [0.0, self._tot_duration / 1000]
        )
        self._eval_times_instruction = value

    def draw(
        self,
        draw_phase_area: bool = False,
        draw_interp_pts: bool = False,
        draw_phase_shifts: bool = False,
        draw_phase_curve: bool = False,
        fig_name: str = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the input sequence and the one used by the solver.

        Args:
            draw_phase_area: Whether phase and area values need
                to be shown as text on the plot, defaults to False.
            draw_interp_pts: When the sequence has pulses with waveforms
                of type InterpolatedWaveform, draws the points of interpolation
                on top of the respective waveforms (defaults to False). Can't
                be used if the sequence is modulated.
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
        if draw_interp_pts and self._modulated:
            raise ValueError(
                "Can't draw the interpolation points when the sequence is "
                "modulated; `draw_interp_pts` must be `False`."
            )
        draw_sequence(
            self._seq,
            self._sampling_rate,
            draw_input=not self._modulated,
            draw_modulation=self._modulated,
            draw_phase_area=draw_phase_area,
            draw_interp_pts=draw_interp_pts,
            draw_phase_shifts=draw_phase_shifts,
            draw_phase_curve=draw_phase_curve,
        )
        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""
        local_noises = True
        if set(self.config.noise).issubset(
            {"dephasing", "SPAM", "depolarizing", "eff_noise"}
        ):
            local_noises = "SPAM" in self.config.noise and self.config.eta > 0
        samples = self.samples_obj.to_nested_dict(all_local=local_noises)

        def add_noise(
            slot: _TargetSlot,
            samples_dict: Mapping[QubitId, dict[str, np.ndarray]],
            is_global_pulse: bool,
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            noise_amp_base = max(
                0, np.random.normal(1.0, self.config.amp_sigma)
            )
            for qid in slot.targets:
                if "doppler" in self.config.noise:
                    noise_det = self._doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.config.noise and is_global_pulse:
                    position = self._qdict[qid]
                    r = np.linalg.norm(position)
                    w0 = self.config.laser_waist
                    noise_amp = noise_amp_base * np.exp(-((r / w0) ** 2))
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= noise_amp

        if local_noises:
            for ch, ch_samples in self.samples_obj.channel_samples.items():
                addr = self._seq.declared_channels[ch].addressing
                basis = self._seq.declared_channels[ch].basis
                samples_dict = samples["Local"][basis]
                for slot in ch_samples.slots:
                    add_noise(slot, samples_dict, addr == "Global")
            # Delete samples for badly prepared atoms
            for basis in samples["Local"]:
                for qid in samples["Local"][basis]:
                    if self._bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            samples["Local"][basis][qid][qty] = 0.0
        self.samples = samples

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
        op_list = [self.op_matrix["I"] for j in range(self._size)]

        if not isinstance(operations, list):
            operations = [operations]

        for operator, qubits in operations:
            if qubits == "global":
                return sum(
                    self.build_operator([(operator, [q_id])])
                    for q_id in self._qdict
                )
            else:
                qubits_set = set(qubits)
                if len(qubits_set) < len(qubits):
                    raise ValueError("Duplicate atom ids in argument list.")
                if not qubits_set.issubset(self._qdict.keys()):
                    raise ValueError(
                        "Invalid qubit names: "
                        f"{qubits_set - self._qdict.keys()}"
                    )
                if isinstance(operator, str):
                    try:
                        operator = self.op_matrix[operator]
                    except KeyError:
                        raise ValueError(f"{operator} is not a valid operator")
                for qubit in qubits:
                    k = self._qid_index[qubit]
                    op_list[k] = operator
        return qutip.tensor(op_list)

    def _adapt_to_sampling_rate(self, full_array: np.ndarray) -> np.ndarray:
        """Adapt list to correspond to sampling rate."""
        indices = np.linspace(
            0,
            len(full_array) - 1,
            int(self._sampling_rate * (self._tot_duration + 1)),
            dtype=int,
        )
        return cast(np.ndarray, full_array[indices])

    def _update_noise(self) -> None:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        if "SPAM" in self.config.noise and self.config.eta > 0:
            dist = (
                np.random.uniform(size=len(self._qid_index))
                < self.config.spam_dict["eta"]
            )
            self._bad_atoms = dict(zip(self._qid_index, dist))
        if "doppler" in self.config.noise:
            detune = np.random.normal(
                0, self.config.doppler_sigma, size=len(self._qid_index)
            )
            self._doppler_detune = dict(zip(self._qid_index, detune))

    def _build_basis_and_op_matrices(self) -> None:
        """Determine dimension, basis and projector operators."""
        if self._interaction == "XY":
            self.basis_name = "XY"
            self.dim = 2
            basis = ["u", "d"]
            projectors = ["uu", "du", "ud", "dd"]
        else:
            used_bases = self.samples_obj.used_bases()
            if "digital" not in used_bases:
                self.basis_name = "ground-rydberg"
                self.dim = 2
                basis = ["r", "g"]
                projectors = ["gr", "rr", "gg"]
            elif "ground-rydberg" not in used_bases:
                self.basis_name = "digital"
                self.dim = 2
                basis = ["g", "h"]
                projectors = ["hg", "hh", "gg"]
            else:
                self.basis_name = "all"  # All three states
                self.dim = 3
                basis = ["r", "g", "h"]
                projectors = ["gr", "hg", "rr", "gg", "hh"]

        self.basis = {b: qutip.basis(self.dim, i) for i, b in enumerate(basis)}
        self.op_matrix = {"I": qutip.qeye(self.dim)}

        for proj in projectors:
            self.op_matrix["sigma_" + proj] = (
                self.basis[proj[0]] * self.basis[proj[1]].dag()
            )

    def _construct_hamiltonian(self, update: bool = True) -> None:
        """Constructs the hamiltonian from the Sequence.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.

        Args:
            update: Whether to update the noise parameters.
        """
        if update:
            self._update_noise()
        self._extract_samples()
        if not hasattr(self, "basis_name"):
            self._build_basis_and_op_matrices()

        def make_vdw_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_rr" at each pair.
            The units are given so that the coefficient includes a
            1/hbar factor.
            """
            dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
            U = 0.5 * self._seq._device.interaction_coeff / dist**6
            return U * self.build_operator([("sigma_rr", [q1, q2])])

        def make_xy_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the XY interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_du * sigma_ud" at each pair.
            The units are given so that the coefficient
            includes a 1/hbar factor.
            """
            dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
            coords_dim = len(self._qdict[q1])
            mag_norm = np.linalg.norm(self._seq.magnetic_field[:coords_dim])
            if mag_norm < 1e-8:
                cosine = 0.0
            else:
                cosine = np.dot(
                    (self._qdict[q1] - self._qdict[q2]),
                    self._seq.magnetic_field[:coords_dim],
                ) / (dist * mag_norm)
            U = (
                0.5
                * cast(float, self._seq._device.interaction_coeff_xy)
                * (1 - 3 * cosine**2)
                / dist**3
            )
            return U * self.build_operator(
                [("sigma_du", [q1]), ("sigma_ud", [q2])]
            )

        def make_interaction_term(masked: bool = False) -> qutip.Qobj:
            if masked:
                # Calculate the total number of good, unmasked qubits
                effective_size = self._size - sum(self._bad_atoms.values())
                for q in self._seq._slm_mask_targets:
                    if not self._bad_atoms[q]:
                        effective_size -= 1
                if effective_size < 2:
                    return 0 * self.build_operator([("I", "global")])

            # make interaction term
            dipole_interaction = cast(qutip.Qobj, 0)
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                if (
                    self._bad_atoms[q1]
                    or self._bad_atoms[q2]
                    or (
                        masked
                        and (
                            q1 in self._seq._slm_mask_targets
                            or q2 in self._seq._slm_mask_targets
                        )
                    )
                ):
                    continue

                if self._interaction == "XY":
                    dipole_interaction += make_xy_term(q1, q2)
                else:
                    dipole_interaction += make_vdw_term(q1, q2)
            return dipole_interaction

        def build_coeffs_ops(basis: str, addr: str) -> list[list]:
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == "ground-rydberg":
                op_ids = ["sigma_gr", "sigma_rr"]
            elif basis == "digital":
                op_ids = ["sigma_hg", "sigma_gg"]
            elif basis == "XY":
                op_ids = ["sigma_du", "sigma_dd"]

            terms = []
            if addr == "Global":
                coeffs = [
                    0.5 * samples["amp"] * np.exp(-1j * samples["phase"]),
                    -0.5 * samples["det"],
                ]
                for op_id, coeff in zip(op_ids, coeffs):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] = self.build_operator(
                                [(op_id, "global")]
                            )
                        terms.append(
                            [
                                operators[op_id],
                                self._adapt_to_sampling_rate(coeff),
                            ]
                        )
            elif addr == "Local":
                for q_id, samples_q in samples.items():
                    if q_id not in operators:
                        operators[q_id] = {}
                    coeffs = [
                        0.5
                        * samples_q["amp"]
                        * np.exp(-1j * samples_q["phase"]),
                        -0.5 * samples_q["det"],
                    ]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if np.any(coeff != 0):
                            if op_id not in operators[q_id]:
                                operators[q_id][op_id] = self.build_operator(
                                    [(op_id, [q_id])]
                                )
                            terms.append(
                                [
                                    operators[q_id][op_id],
                                    self._adapt_to_sampling_rate(coeff),
                                ]
                            )
            self.operators[addr][basis] = operators
            return terms

        qobj_list = []
        # Time independent term:
        effective_size = self._size - sum(self._bad_atoms.values())
        if self.basis_name != "digital" and effective_size > 1:
            # Build time-dependent or time-independent interaction term based
            # on whether an SLM mask was defined or not
            if self._seq._slm_mask_time:
                # Build an array of binary coefficients for the interaction
                # term of unmasked qubits
                coeff = np.ones(self._tot_duration)
                coeff[0 : self._seq._slm_mask_time[1]] = 0
                # Build the interaction term for unmasked qubits
                qobj_list = [
                    [
                        make_interaction_term(),
                        self._adapt_to_sampling_rate(coeff),
                    ]
                ]
                # Build the interaction term for masked qubits
                qobj_list += [
                    [
                        make_interaction_term(masked=True),
                        self._adapt_to_sampling_rate(
                            np.logical_not(coeff).astype(int)
                        ),
                    ]
                ]
            else:
                qobj_list = [make_interaction_term()]

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr))

        if not qobj_list:  # If qobj_list ends up empty
            qobj_list = [0 * self.build_operator([("I", "global")])]

        ham = qutip.QobjEvo(qobj_list, tlist=self.sampling_times)
        ham = ham + ham.dag()
        ham.compress()
        self._hamiltonian = ham

    def get_hamiltonian(self, time: float) -> qutip.Qobj:
        """Get the Hamiltonian created from the sequence at a fixed time.

        Args:
            time: The specific time at which we want to extract the
                Hamiltonian (in ns).

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
        return self._hamiltonian(time / 1000)  # Creates new Qutip.Qobj

    # Run Simulation Evolution using Qutip
    def run(
        self,
        progress_bar: Optional[bool] = False,
        **options: Any,
    ) -> SimulationResults:
        """Simulates the sequence using QuTiP's solvers.

        Will return NoisyResults if the noise in the SimConfig requires it.
        Otherwise will return CoherentResults.

        Args:
            progress_bar: If True, the progress bar of QuTiP's
                solver will be shown. If None or False, no text appears.
            options: Used as arguments for qutip.Options(). If specified, will
                override SimConfig solver_options. If no `max_step` value is
                provided, an automatic one is calculated from the `Sequence`'s
                schedule (half of the shortest duration among pulses and
                delays).
                Refer to the QuTiP docs_ for an overview of the parameters.

                .. _docs: https://bit.ly/3il9A2u
        """
        if "max_step" in options.keys():
            solv_ops = qutip.Options(**options)
        else:
            min_pulse_duration = min(
                slot.tf - slot.ti
                for ch_schedule in self._seq._schedule.values()
                for slot in ch_schedule
                if isinstance(slot.type, Pulse)
            )
            auto_max_step = 0.5 * (min_pulse_duration / 1000)
            solv_ops = qutip.Options(max_step=auto_max_step, **options)

        meas_errors: Optional[Mapping[str, float]] = None
        if "SPAM" in self.config.noise:
            meas_errors = {
                k: self.config.spam_dict[k]
                for k in ("epsilon", "epsilon_prime")
            }
            if self.config.eta > 0 and self.initial_state != qutip.tensor(
                [self.basis["g"] for _ in range(self._size)]
            ):
                raise NotImplementedError(
                    "Can't combine state preparation errors with an initial "
                    "state different from the ground."
                )

        def _run_solver() -> CoherentResults:
            """Returns CoherentResults: Object containing evolution results."""
            # Decide if progress bar will be fed to QuTiP solver
            p_bar: Optional[bool]
            if progress_bar is True:
                p_bar = True
            elif (progress_bar is False) or (progress_bar is None):
                p_bar = None
            else:
                raise ValueError("`progress_bar` must be a bool.")

            if (
                "dephasing" in self.config.noise
                or "depolarizing" in self.config.noise
                or "eff_noise" in self.config.noise
            ):
                result = qutip.mesolve(
                    self._hamiltonian,
                    self.initial_state,
                    self._eval_times_array,
                    self._collapse_ops,
                    progress_bar=p_bar,
                    options=solv_ops,
                )
            else:
                result = qutip.sesolve(
                    self._hamiltonian,
                    self.initial_state,
                    self._eval_times_array,
                    progress_bar=p_bar,
                    options=solv_ops,
                )
            return CoherentResults(
                result.states,
                self._size,
                self.basis_name,
                self._eval_times_array,
                self._meas_basis,
                meas_errors,
            )

        # Check if noises ask for averaging over multiple runs:
        if set(self.config.noise).issubset(
            {"dephasing", "SPAM", "depolarizing", "eff_noise"}
        ):
            # If there is "SPAM", the preparation errors must be zero
            if "SPAM" not in self.config.noise or self.config.eta == 0:
                return _run_solver()

            else:
                # Stores the different initial configurations and frequency
                initial_configs = Counter(
                    "".join(
                        (
                            np.random.uniform(size=len(self._qid_index))
                            < self.config.eta
                        )
                        .astype(int)
                        .astype(str)  # Turns bool->int->str
                    )
                    for _ in range(self.config.runs)
                ).most_common()
                loop_runs = len(initial_configs)
                update_ham = False
        else:
            loop_runs = self.config.runs
            update_ham = True

        # Will return NoisyResults
        time_indices = range(len(self._eval_times_array))
        total_count = np.array([Counter() for _ in time_indices])
        # We run the system multiple times
        for i in range(loop_runs):
            if not update_ham:
                initial_state, reps = initial_configs[i]
                # We load the initial state manually
                self._bad_atoms = dict(
                    zip(
                        self._qid_index,
                        np.array(list(initial_state)).astype(bool),
                    )
                )
            else:
                reps = 1
            # At each run, new random noise: new Hamiltonian
            self._construct_hamiltonian(update=update_ham)
            # Get CoherentResults instance from sequence with added noise:
            cleanres_noisyseq = _run_solver()
            # Extract statistics at eval time:
            total_count += np.array(
                [
                    cleanres_noisyseq.sample_state(
                        t, n_samples=self.config.samples_per_run * reps
                    )
                    for t in self._eval_times_array
                ]
            )
        n_measures = self.config.runs * self.config.samples_per_run
        total_run_prob = [
            Counter({k: v / n_measures for k, v in total_count[t].items()})
            for t in time_indices
        ]
        return NoisyResults(
            total_run_prob,
            self._size,
            self.basis_name,
            self._eval_times_array,
            n_measures,
        )
