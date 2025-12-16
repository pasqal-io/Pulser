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

import itertools
from collections import defaultdict
from typing import Union, cast

import numpy as np
import qutip

from pulser._hamiltonian_data import BasisData, LindbladData, NoiseTrajectory
from pulser.channels.base_channel import States
from pulser.register import QubitId
from pulser.register.base_register import BaseRegister
from pulser.sampler.samples import SequenceSamples


class Hamiltonian:
    r"""Generates Hamiltonian from a sampled sequence and noise.

    Args:
        samples: A sampled sequence whose ChannelSamples have same
            duration.
        register: A Register associating coordinates to qubit ids.
        device: The device specifications.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
    """

    def __init__(
        self,
        samples: SequenceSamples,
        noise_trajectory: NoiseTrajectory,
        basis_data: BasisData,
        lindblad_data: LindbladData,
        sampling_rate: float,
    ) -> None:
        """Instantiates a Hamiltonian object."""
        self.samples = samples
        self.noise_trajectory = noise_trajectory
        self._sampling_rate = sampling_rate
        self._qid_index = {
            qid: i for i, qid in enumerate(self._register.qubits)
        }
        self.basis_data = basis_data
        self.lindblad_data = lindblad_data

        # Type hints for attributes defined outside of __init__
        self.op_matrix: dict[str, qutip.Qobj]
        self.basis: dict[States, qutip.Qobj]

        # Compute sampling times
        self._duration = self.samples.max_duration
        self.sampling_times = self._adapt_to_sampling_rate(
            # Include extra time step for final instruction from samples:
            np.arange(self._duration, dtype=np.double)
            / 1000
        )

        # Stores the qutip operators used in building the Hamiltonian
        self._collapse_ops: list[qutip.Qobj] = []
        self._operators: dict[str, defaultdict[str, dict]] = {
            addr: defaultdict(dict) for addr in ["Global", "Local"]
        }

        self._set_config()

    @property
    def _register(self) -> BaseRegister:
        return self.noise_trajectory.register

    def _adapt_to_sampling_rate(self, full_array: np.ndarray) -> np.ndarray:
        """Adapt list to correspond to sampling rate."""
        indices = np.linspace(
            0,
            len(full_array) - 1,
            int(self._sampling_rate * self._duration),
            dtype=int,
        )
        return full_array[indices]

    def _build_collapse_operators(
        self,
    ) -> None:
        """Build the multi-qudit collapse operators."""
        self._collapse_ops = []
        for coeff, collapse_op in self.lindblad_data.local_collapse_ops:
            if isinstance(collapse_op, str):
                if collapse_op not in self.op_matrix:
                    operator = sum(
                        [
                            coeff * proj_coeff * self.op_matrix[proj_op]
                            for (
                                proj_coeff,
                                proj_op,
                            ) in self.lindblad_data.depolarizing_pauli_2ds[
                                collapse_op
                            ]
                        ]
                    )
                else:
                    operator = coeff * (self.op_matrix)[collapse_op]
            else:
                assert isinstance(collapse_op, np.ndarray)
                operator = coeff * qutip.Qobj(collapse_op).full()
            self._collapse_ops += [
                self._build_operator([(operator, [qid])], self.op_matrix)
                for qid in self._register.qubit_ids
            ]

    def _set_config(self) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
            from_init: this is only meant to be used in Hamiltonian.__init__
        """
        basis, op_matrix = self._get_basis_op_matrices(
            self.basis_data.eigenbasis
        )
        self.basis = basis
        self.op_matrix = op_matrix
        assert set(self.lindblad_data.op_matrix_names) == set(
            self.op_matrix.keys()
        )
        self._build_collapse_operators()
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()

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
        op_list = [op_matrix["I"] for j in range(self.n_qudits)]

        if not isinstance(operations, list):
            operations = [operations]

        for operator, qubits in operations:
            if qubits == "global":
                return sum(
                    self._build_operator([(operator, [q_id])], op_matrix)
                    for q_id in self._register.qubits
                )
            else:
                qubits_set = set(qubits)
                if len(qubits_set) < len(qubits):
                    raise ValueError("Duplicate atom ids in argument list.")
                if not qubits_set.issubset(self._register.qubits.keys()):
                    v = qubits_set
                    v -= self._register.qubits.keys()
                    raise ValueError("Invalid qubit names: " f"{v}")
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

    @property
    def n_qudits(self) -> int:
        """Number of qudits in the Register."""
        return len(self._register.qubit_ids)

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

    def _construct_hamiltonian(self) -> None:
        """Constructs the hamiltonian from the sampled Sequence and noise.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.

        Warning:
            The refreshed noise parameters (when update=True) are only those
            that change from shot to shot (ie doppler, state preparation,
            and register).
            Amplitude fluctuations change from pulse to pulse and are always
            applied in `_extract_samples()`.
        """

        def make_vdw_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_rr" at each pair.
            The units are given so that the coefficient includes a
            1/hbar factor.
            """
            U = (
                0.5
                * self.noise_trajectory.interaction_matrix.as_array(
                    detach=True
                )[self._qid_index[q1], self._qid_index[q2]]
            )
            return U * self.build_operator([("sigma_rr", [q1, q2])])

        def make_xy_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the XY interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_ud * sigma_du" at each pair.
            The units are given so that the coefficient
            includes a 1/hbar factor.
            """
            U = self.noise_trajectory.interaction_matrix.as_array(detach=True)[
                self._qid_index[q1], self._qid_index[q2]
            ]
            return U * self.build_operator(
                [("sigma_ud", [q1]), ("sigma_du", [q2])]
            )

        def make_interaction_term(masked: bool = False) -> qutip.Qobj:
            if masked:
                # Calculate the total number of good, unmasked qubits
                effective_size = self.n_qudits - sum(
                    self.noise_trajectory.bad_atoms.values()
                )
                for q in self.samples._slm_mask.targets:
                    if not self.noise_trajectory.bad_atoms[q]:
                        effective_size -= 1
                if effective_size < 2:
                    return 0 * self.build_operator([("I", "global")])

            # make interaction term
            dipole_interaction = cast(qutip.Qobj, 0)
            for q1, q2 in itertools.combinations(
                self._register.qubits.keys(), r=2
            ):
                if (
                    self.noise_trajectory.bad_atoms[q1]
                    or self.noise_trajectory.bad_atoms[q2]
                    or (
                        masked
                        and self.basis_data.interaction_type == "XY"
                        and (
                            q1 in self.samples._slm_mask.targets
                            or q2 in self.samples._slm_mask.targets
                        )
                    )
                ):
                    continue

                if self.basis_data.interaction_type == "XY":
                    dipole_interaction += make_xy_term(q1, q2)
                else:
                    dipole_interaction += make_vdw_term(q1, q2)
            return dipole_interaction

        def build_coeffs_ops(
            basis: str, addr: str, samples_dict: dict
        ) -> list[list]:
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = samples_dict[addr][basis]
            operators = self._operators[addr][basis]
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
            self._operators[addr][basis] = operators
            return terms

        qobj_list = []
        # Time independent term:
        effective_size = self.n_qudits - sum(
            self.noise_trajectory.bad_atoms.values()
        )
        if "digital" not in self.basis_data.basis_name and effective_size > 1:
            # Build time-dependent or time-independent interaction term based
            # on whether an SLM mask was defined or not
            if (
                self.samples._slm_mask.end > 0
                and self.basis_data.interaction_type == "XY"
            ):
                # Build an array of binary coefficients for the interaction
                # term of unmasked qubits
                coeff = np.ones(self._duration - 1)
                coeff[0 : self.samples._slm_mask.end] = 0
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
        d = self.samples.to_nested_dict()
        for addr in d:
            for basis in d[addr]:
                if d[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr, d))

        if not qobj_list:  # If qobj_list ends up empty
            qobj_list = [0 * self.build_operator([("I", "global")])]

        ham = qutip.QobjEvo(qobj_list, tlist=self.sampling_times)
        ham = ham + ham.dag()
        ham.compress()
        self._hamiltonian = ham
