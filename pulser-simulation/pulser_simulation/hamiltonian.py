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
"""Defines the Hamiltonian class."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Mapping
from typing import Union, cast

import numpy as np
import qutip

from pulser.channels.base_channel import STATES_RANK, States
from pulser.devices._device_datacls import BaseDevice
from pulser.noise_model import NoiseModel
from pulser.register.base_register import QubitId, BaseRegister
from pulser.sampler.noisy_sampler import BaseHamiltonian
from pulser.sampler.samples import SequenceSamples, _PulseTargetSlot
from pulser_simulation.simconfig import SUPPORTED_NOISES, doppler_sigma


class Hamiltonian(BaseHamiltonian):
    r"""Generates Hamiltonian from a sampled sequence and noise.

    Args:
        samples_obj: A sampled sequence whose ChannelSamples have same
            duration.
        register: A Register associating coordinates to qubit ids.
        device: The device specifications.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
    """

    def __init__(
        self,
        samples_obj: SequenceSamples,
        register: BaseRegister,
        device: BaseDevice,
        sampling_rate: float,
        config: NoiseModel,
    ) -> None:
        """Instantiates a Hamiltonian object."""
        super().__init__(
            samples_obj, register, device, config, assign_config=False
        )
        self._sampling_rate = sampling_rate

        # Type hints for attributes defined outside of __init__
        self.op_matrix: dict[str, qutip.Qobj]
        self.basis: dict[States, qutip.Qobj]

        # Compute sampling times
        self._duration = self.samples_obj.max_duration
        self.sampling_times = self._adapt_to_sampling_rate(
            # Include extra time step for final instruction from samples:
            np.arange(self._duration, dtype=np.double)
            / 1000
        )

        # Stores the qutip operators used in building the Hamiltonian
        self._collapse_ops: list[qutip.Qobj] = []

        self.set_config(config)

    def _adapt_to_sampling_rate(self, full_array: np.ndarray) -> np.ndarray:
        """Adapt list to correspond to sampling rate."""
        indices = np.linspace(
            0,
            len(full_array) - 1,
            int(self._sampling_rate * self._duration),
            dtype=int,
        )
        return cast(np.ndarray, full_array[indices])

    @property
    def config(self) -> NoiseModel:
        """The current configuration, as a NoiseModel instance."""
        return self._config

    def _build_collapse_operators(
        self,
    ) -> None:
        """Build the multi-qudit collapse operators."""
        self._collapse_ops = []
        for coeff, collapse_op in self.local_collapse_operators:
            if isinstance(collapse_op, str):
                if collapse_op not in self.op_matrix:
                    self._depolarizing_pauli_2ds[collapse_op]

                try:
                    operator = coeff * self.op_matrix[collapse_op]
                except KeyError as e1:
                    try:
                        operator = sum(
                            [
                                coeff * proj_coeff * self.op_matrix[proj_op]
                                for (
                                    proj_coeff,
                                    proj_op,
                                ) in self._depolarizing_pauli_2ds[collapse_op]
                            ]
                        )
                    except KeyError as e2:
                        raise KeyError(
                            f"Invalid local collapse operator {collapse_op}."
                        ) from (e1, e2)

            else:
                assert isinstance(collapse_op, np.ndarray)
                operator = coeff * qutip.Qobj(collapse_op).full()
            self._collapse_ops += [
                self._build_operator([(operator, [qid])], self.op_matrix)
                for qid in self._qid_index
            ]

    def set_config(self, cfg: NoiseModel) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
        """
        self._check_config(cfg)
        not_supported = (
            set(cfg.noise_types) - SUPPORTED_NOISES[self.interaction_type]
        )
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self.interaction_type}' does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )
        update_basis = self._update_basis(cfg)
        super().set_config(cfg, construct_hamiltonian=False)
        if update_basis:
            basis, op_matrix = self._get_basis_op_matrices(self.eigenbasis)
            self.basis = basis
            self.op_matrix = op_matrix
            assert set(self.op_matrix_names) == set(self.op_matrix.keys())
        self._build_collapse_operators()
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()

    @staticmethod
    @functools.cache
    def _finite_waist_amp_fraction(
        coords: tuple[float, ...],
        propagation_dir: tuple[float, float, float],
        laser_waist: float,
    ) -> float:
        pos_vec = np.zeros(3, dtype=float)
        pos_vec[: len(coords)] = np.array(coords, dtype=float)
        u_vec = np.array(propagation_dir, dtype=float)
        u_vec = u_vec / np.linalg.norm(u_vec)
        # Given a line crossing the origin with normalized direction vector
        # u_vec, the closest point along said line and an arbitrary point
        # pos_vec is at k*u_vec, where
        k = np.dot(pos_vec, u_vec)
        # The distance between pos_vec and the line is then given by
        dist = np.linalg.norm(pos_vec - k * u_vec)
        # We assume the Rayleigh length of the gaussian beam is very large,
        # resulting in a negligble drop in amplitude along the propagation
        # direction. Therefore, the drop in amplitude at a given position
        # is dictated solely by its distance to the optical axis (as dictated
        # by the propagation direction), ie
        return float(np.exp(-((dist / laser_waist) ** 2)))

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""
        local_noises = True
        if set(self.config.noise_types).issubset(
            {
                "dephasing",
                "relaxation",
                "SPAM",
                "depolarizing",
                "eff_noise",
                "leakage",
            }
        ):
            local_noises = (
                "SPAM" in self.config.noise_types
                and self.config.state_prep_error > 0
            )
        samples = self.samples_obj.to_nested_dict(all_local=local_noises)

        def add_noise(
            slot: _PulseTargetSlot,
            samples_dict: Mapping[QubitId, dict[str, np.ndarray]],
            is_global_pulse: bool,
            amp_fluctuation: float,
            det_fluctuation: float,
            propagation_dir: tuple | None,
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            for qid in slot.targets:
                if "doppler" in self.config.noise_types:
                    noise_det = self._doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.config.noise_types:
                    amp_fraction = amp_fluctuation
                    if self.config.laser_waist is not None and is_global_pulse:
                        # Default to an optical axis along y
                        prop_dir = propagation_dir or (0.0, 1.0, 0.0)
                        amp_fraction *= self._finite_waist_amp_fraction(
                            tuple(self._qdict[qid]),
                            tuple(prop_dir),
                            self.config.laser_waist,
                        )
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= amp_fraction
                if "detuning" in self.config.noise_types:
                    samples_dict[qid]["det"][
                        slot.ti : slot.tf
                    ] += det_fluctuation

        if local_noises:
            for ch, ch_samples in self.samples_obj.channel_samples.items():
                _ch_obj = self.samples_obj._ch_objs[ch]
                samples_dict = samples["Local"][_ch_obj.basis]
                ch_amp_fluctuation = max(
                    0, np.random.normal(1.0, self.config.amp_sigma)
                )
                ch_det_fluctuation = (
                    np.random.normal(0.0, self.config.detuning_sigma)
                    if self.config.detuning_sigma
                    else 0.0
                )
                for slot in ch_samples.slots:
                    add_noise(
                        slot,
                        samples_dict,
                        _ch_obj.addressing == "Global",
                        amp_fluctuation=ch_amp_fluctuation,
                        det_fluctuation=ch_det_fluctuation,
                        propagation_dir=_ch_obj.propagation_dir,
                    )
            # Delete samples for badly prepared atoms
            for basis in samples["Local"]:
                for qid in samples["Local"][basis]:
                    if self._bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            samples["Local"][basis][qid][qty] = 0.0
        self.samples = samples

    def _build_operator(
        self, operations: Union[list, tuple], op_matrix: dict[str, qutip.Qobj]
    ) -> qutip.Qobj:
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
        op_list = [op_matrix["I"] for j in range(self._size)]

        if not isinstance(operations, list):
            operations = [operations]

        for operator, qubits in operations:
            if qubits == "global":
                return sum(
                    self._build_operator([(operator, [q_id])], op_matrix)
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
                elif isinstance(operator, qutip.Qobj):
                    operator = operator.to("CSR")
                else:
                    operator = qutip.Qobj(operator).to("CSR")
                for qubit in qubits:
                    k = self._qid_index[qubit]
                    op_list[k] = operator
        return qutip.tensor(op_list)

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
        return self._build_operator(operations, self.op_matrix)

    def _update_noise(self) -> None:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        if (
            "SPAM" in self.config.noise_types
            and self.config.state_prep_error > 0
        ):
            dist = (
                np.random.uniform(size=len(self._qid_index))
                < self.config.state_prep_error
            )
            self._bad_atoms = dict(zip(self._qid_index, dist))
        if "doppler" in self.config.noise_types:
            temp = self.config.temperature * 1e-6
            detune = np.random.normal(
                0, doppler_sigma(temp), size=len(self._qid_index)
            )
            self._doppler_detune = dict(zip(self._qid_index, detune))

    def _get_eigenbasis(self, with_leakage: bool) -> list[States]:
        eigenbasis = self.samples_obj.eigenbasis
        if with_leakage:
            eigenbasis.append("x")
        return [state for state in STATES_RANK if state in eigenbasis]

    @staticmethod
    def _get_basis_op_matrices(
        eigenbasis: list[States],
    ) -> tuple[dict[States, qutip.Qobj], dict[str, qutip.Qobj]]:
        """Determine basis and projector operators."""
        dim = len(eigenbasis)
        with qutip.CoreOptions(default_dtype="CSR"):
            basis = {b: qutip.basis(dim, i) for i, b in enumerate(eigenbasis)}
            op_matrix = {"I": qutip.qeye(dim)}
        for proj0 in eigenbasis:
            for proj1 in eigenbasis:
                proj_name = "sigma_" + proj0 + proj1
                op_matrix[proj_name] = basis[proj0] * basis[proj1].dag()
        return basis, op_matrix

    def _construct_hamiltonian(self, update: bool = True) -> None:
        """Constructs the hamiltonian from the sampled Sequence and noise.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.

        Warning:
            The refreshed noise parameters (when update=True) are only those
            that change from shot to shot (ie doppler and state preparation).
            Amplitude fluctuations change from pulse to pulse and are always
            applied in `_extract_samples()`.

        Args:
            update: Whether to update the noise parameters.
        """
        if update:
            self._update_noise()
        self._extract_samples()

        def make_vdw_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_rr" at each pair.
            The units are given so that the coefficient includes a
            1/hbar factor.
            """
            dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
            U = 0.5 * self._device.interaction_coeff / dist**6
            return U * self.build_operator([("sigma_rr", [q1, q2])])

        def make_xy_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the XY interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_ud * sigma_du" at each pair.
            The units are given so that the coefficient
            includes a 1/hbar factor.
            """
            diff_vector = np.zeros(3, dtype=float)
            diff_vector[: len(self._qdict[q1])] = (
                self._qdict[q1] - self._qdict[q2]
            )
            dist = np.linalg.norm(diff_vector)
            mag_field = cast(np.ndarray, self.samples_obj._magnetic_field)
            mag_norm = np.linalg.norm(mag_field)
            assert mag_norm > 0, "There must be a magnetic field in XY mode."
            cosine = np.dot(
                diff_vector,
                mag_field,
            ) / (dist * mag_norm)
            U = (
                cast(float, self._device.interaction_coeff_xy)
                * (1 - 3 * cosine**2)
                / dist**3
            )
            return U * self.build_operator(
                [("sigma_ud", [q1]), ("sigma_du", [q2])]
            )

        def make_interaction_term(masked: bool = False) -> qutip.Qobj:
            if masked:
                # Calculate the total number of good, unmasked qubits
                effective_size = self._size - sum(self._bad_atoms.values())
                for q in self.samples_obj._slm_mask.targets:
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
                        and self._interaction == "XY"
                        and (
                            q1 in self.samples_obj._slm_mask.targets
                            or q2 in self.samples_obj._slm_mask.targets
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
                op_ids = ["sigma_du", "sigma_uu"]

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
        if "digital" not in self.basis_name and effective_size > 1:
            # Build time-dependent or time-independent interaction term based
            # on whether an SLM mask was defined or not
            if (
                self.samples_obj._slm_mask.end > 0
                and self._interaction == "XY"
            ):
                # Build an array of binary coefficients for the interaction
                # term of unmasked qubits
                coeff = np.ones(self._duration - 1)
                coeff[0 : self.samples_obj._slm_mask.end] = 0
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
                            np.logical_not(coeff).astype(int),
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
