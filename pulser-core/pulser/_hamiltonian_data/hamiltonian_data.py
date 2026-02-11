# Copyright 2025 Pulser Development Team
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
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from typing import Iterator, List, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

import pulser.math as pm
from pulser._hamiltonian_data.basis_data import BasisData
from pulser._hamiltonian_data.lindblad_data import LindbladData
from pulser._hamiltonian_data.noise_trajectory import NoiseTrajectory
from pulser.channels import Microwave, Raman, Rydberg
from pulser.channels.base_channel import STATES_RANK, Channel, States
from pulser.devices._device_datacls import COORD_PRECISION, BaseDevice
from pulser.noise_model import NoiseModel
from pulser.noise_model import _doppler_sigma as doppler_sigma
from pulser.noise_model import _register_sigma_xy_z
from pulser.register import Register3D
from pulser.register.base_register import BaseRegister, QubitId
from pulser.sampler import sampler
from pulser.sampler.samples import (
    ChannelSamples,
    SequenceSamples,
    _PulseTargetSlot,
)
from pulser.sequence import Sequence


class TrajectoryWithReps(NamedTuple):
    """A NoiseTrajectory and the number of times it should be simulated."""

    trajectory: NoiseTrajectory
    reps: int


class SamplesWithReps(NamedTuple):
    """A trajectory, samples and how often they should be simulated."""

    trajectory: NoiseTrajectory
    samples: SequenceSamples
    reps: int


SUPPORTED_NOISES: dict = {
    "ising": {
        "amplitude",
        "detuning",
        "dephasing",
        "relaxation",
        "depolarizing",
        "doppler",
        "eff_noise",
        "SPAM",
        "leakage",
        "register",
    },
    "XY": {
        "dephasing",
        "depolarizing",
        "eff_noise",
        "SPAM",
        "leakage",
        "register",
    },
}


def has_shot_to_shot_except_spam(noise_model: NoiseModel) -> bool:
    """Whether the noise model contains stochastic noise, excepting SPAM.

    Arg:
        noise_model: The noise model to check.
    """
    return (
        "doppler" in noise_model.noise_types
        or (
            "amplitude" in noise_model.noise_types
            and noise_model.amp_sigma != 0.0
        )
        or "detuning" in noise_model.noise_types
        or "register" in noise_model.noise_types
    )


def _noisy_register(
    q_dict: dict[QubitId, pm.AbstractArray], noise_model: NoiseModel
) -> Register3D:
    """Add Gaussian noise to the positions of the register."""
    register_sigma_xy, register_sigma_z = _register_sigma_xy_z(
        noise_model.temperature,
        noise_model.trap_waist,
        cast(float, noise_model.trap_depth),
    )
    atoms = list(q_dict.keys())
    num_atoms = len(atoms)
    positions = list(q_dict.values())
    pos = positions[0]
    if len(pos) == 2:
        positions = [pm.concatenate((p, [0.0])) for p in positions]
    narr_xy = np.random.normal(0, register_sigma_xy, (num_atoms, 2))
    narr_z = np.random.normal(0, register_sigma_z, num_atoms)
    narr = np.column_stack((narr_xy, narr_z))
    return Register3D(
        {k: pos + noise for (k, pos, noise) in zip(atoms, positions, narr)}
    )


def _generate_detuning_fluctuations(
    noise_model: NoiseModel,
    det_cst_term: float,
    phases: np.ndarray,
    times: ArrayLike,
) -> np.ndarray:
    """Compute δ_hf(t) + δ_σ.

    Generates the high-frequency time-dependent component together
    with a constant offset of the detuning fluctuations.

    Args:
        noise_model (NoiseModel): class containing noise parameters
        times (ArrayLike): array of sample times (in µs).

    Notes
    -----
    High frequency term uses Gaussian stochastic noise with power
        spectral density `psd`:
        δ_hf(t) = Σ_k sqrt(2 * Δf_k * psd_k) * cos(2π(f_k * t + φ_k))
        where φ_k ~ U[0, 1) (uniform random phase),
        Δf_k = freqs[k+1] - freqs[k].
        The last (freqs[-1], psd[-1]) is unused.
    """
    det_hf = np.zeros_like(times)

    if noise_model.detuning_hf_psd:
        t = np.asarray(times) * 1e-3  # ns -> microsec
        freqs = np.asarray(noise_model.detuning_hf_omegas)[1:]
        psd = np.asarray(noise_model.detuning_hf_psd)[1:]
        df = np.diff(noise_model.detuning_hf_omegas)
        amp = np.sqrt(2.0 * df * psd)
        arg = freqs[:, None] * t[None, :] + phases[:, None]
        det_hf = (amp[:, None] * np.cos(arg)).sum(axis=0)
    return det_cst_term + det_hf


def _distances(register: BaseRegister) -> pm.AbstractArray:
    r"""Distances between each qubits (in :math:`\mu m`)."""
    positions = list(register.qubits.values())
    if not positions[0].is_tensor:
        return pm.AbstractArray(
            np.round(
                cast(
                    np.ndarray,
                    cdist(positions, positions, metric="euclidean"),
                ),
                COORD_PRECISION,
            ),
        )
    else:
        ten = pm.torch.stack(
            [cast(pm.torch.Tensor, x._array) for x in positions]
        )
        return pm.AbstractArray(pm.torch.cdist(ten, ten))


class HamiltonianData:
    r"""Information that can be used to generate an Hamiltonian.

    Takes information defining the noiseless case, and a noise model.
    Creates a noise trajectory from this info, and allows the user
    to query it for noisy data.

    Args:
        samples: The noiseless sequence samples.
        device: The device specifications.
        register: The noiseless register.
        noise_model: NoiseModel to be used to generate noise.
        n_trajectories: The number of noise trajectories to sample.
            Defaults to 1.
    """

    def __init__(
        self,
        samples: SequenceSamples,
        register: BaseRegister,
        device: BaseDevice,
        noise_model: NoiseModel,
        n_trajectories: int | None,
    ) -> None:
        """Instantiates a Hamiltonian object."""
        # Initializing the samples obj
        if not isinstance(samples, SequenceSamples):
            raise TypeError(
                "The provided sequence has to be a valid "
                "SequenceSamples instance."
            )
        if samples.max_duration == 0:
            raise ValueError("SequenceSamples is empty.")
        # Check compatibility of register and device
        if not isinstance(device, BaseDevice):
            raise TypeError("The device must be a Device or BaseDevice.")
        self._device = device
        self.device.validate_register(register)
        self._register = register
        # Check compatibility of samples and device:
        if samples._slm_mask.end > 0 and not self.device.supports_slm_mask:
            raise ValueError(
                "Samples use SLM mask but device does not have one."
            )
        if not samples.used_bases <= self.device.supported_bases:
            raise ValueError(
                "Bases used in samples should be supported by device."
            )
        # Check compatibility of masked samples and register
        if not samples._slm_mask.targets <= set(self.register.qubits.keys()):
            raise ValueError(
                "The ids of qubits targeted in SLM mask"
                " should be defined in register."
            )

        self._samples = self._delocalize_samples(samples)

        self._size = len(self.register.qubits)
        self._qid_index = {
            qid: i for i, qid in enumerate(self.register.qubits)
        }

        self._noise_model = noise_model
        self._check_noise_model(noise_model)
        if n_trajectories is None:
            n_trajectories = 1

        self.local_noises = True
        if set(self.noise_model.noise_types).issubset(
            {
                "dephasing",
                "relaxation",
                "SPAM",
                "depolarizing",
                "eff_noise",
                "leakage",
            }
        ):
            self.local_noises = (
                "SPAM" in self.noise_model.noise_types
                and self.noise_model.state_prep_error > 0
            )
        self.noise_trajectories = self._create_noise_trajectories(
            n_trajectories
        )

    def _delocalize_samples(self, samples: SequenceSamples) -> SequenceSamples:
        samples_list = []
        for ch, ch_samples in samples.channel_samples.items():
            if samples._ch_objs[ch].addressing == "Local":
                # Check that targets of Local Channels are defined
                # in register
                if not set().union(
                    *(slot.targets for slot in ch_samples.slots)
                ) <= set(self.register.qubits.keys()):
                    raise ValueError(
                        "The ids of qubits targeted in Local channels"
                        " should be defined in register."
                    )
                samples_list.append(ch_samples)
            else:
                # Replace targets of Global channels by qubits of register
                samples_list.append(
                    replace(
                        ch_samples,
                        slots=[
                            replace(
                                slot, targets=set(self.register.qubits.keys())
                            )
                            for slot in ch_samples.slots
                        ],
                    )
                )
        return replace(samples, samples_list=samples_list)

    @property
    def basis_data(self) -> BasisData:
        """Get the BasisData defining this Hamiltonian."""
        interaction: Literal["XY", "ising"] = (
            "XY" if self.samples._in_xy else "ising"
        )
        basis_name = self._get_basis_name(self.noise_model.with_leakage)
        eigenbasis = self._get_eigenbasis(self.noise_model.with_leakage)
        return BasisData(
            dim=len(eigenbasis),
            basis_name=basis_name,
            eigenbasis=eigenbasis,
            interaction_type=interaction,
        )

    @property
    def lindblad_data(self) -> LindbladData:
        """Get the LindbladData defining this Hamiltonian."""
        basis_data = self.basis_data
        op_matrix_names = self._get_projectors(basis_data.eigenbasis)
        local_collapse_ops, paulis = self._build_local_collapse_operators(
            self.noise_model,
            basis_data.basis_name,
            basis_data.eigenbasis,
            op_matrix_names,
        )
        return LindbladData(
            op_matrix_names=op_matrix_names,
            local_collapse_ops=local_collapse_ops,
            depolarizing_pauli_2ds=paulis,
        )

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence,
        *,
        with_modulation: bool = False,
        noise_model: NoiseModel | None = None,
        n_trajectories: int | None = None,
    ) -> HamiltonianData:
        r"""Simulation of a pulse sequence using QuTiP.

        Args:
            sequence: An instance of a Pulser Sequence that we
                want to simulate.
            with_modulation: Whether to simulate the sequence with the
                programmed input or the expected output.
            noise_model: The noise model for the simulation. Replaces and
                should be preferred over 'noise_model'.
            n_trajectories: The number of noise trajectories to sample.
                Defaults to 1.
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
            noise_model or NoiseModel(),
            n_trajectories,
        )

    @functools.cached_property
    def n_qudits(self) -> int:
        """Number of qudits in the Register."""
        return self._size

    @property
    def samples(self) -> SequenceSamples:
        """The samples without noise."""
        return self._samples

    def _sample_with_trajectory(
        self, traj: NoiseTrajectory
    ) -> SequenceSamples:

        samples = self._samples.to_nested_dict(all_local=self.local_noises)

        def add_noise(
            slot: _PulseTargetSlot,
            samples_dict: Mapping[QubitId, dict[str, np.ndarray]],
            is_global_pulse: bool,
            amp_fluctuation: float,
            det_fluctuation: np.ndarray,
            propagation_dir: tuple | None,
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            for qid in slot.targets:
                if "doppler" in self.noise_model.noise_types:
                    noise_det = traj.doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.noise_model.noise_types:
                    amp_fraction = amp_fluctuation
                    if (
                        self.noise_model.laser_waist is not None
                        and is_global_pulse
                    ):
                        # Default to an optical axis along y
                        prop_dir = propagation_dir or (0.0, 1.0, 0.0)
                        amp_fraction *= self._finite_waist_amp_fraction(
                            tuple(traj.register.qubits[qid].as_array()),
                            tuple(prop_dir),
                            self.noise_model.laser_waist,
                        )
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= amp_fraction
                if "detuning" in self.noise_model.noise_types:
                    t_window = slice(slot.ti, slot.tf)
                    samples_dict[qid]["det"][t_window] += det_fluctuation[
                        t_window
                    ]

        if self.local_noises:
            for ch, ch_samples in self._samples.channel_samples.items():
                _ch_obj = self._samples._ch_objs[ch]
                samples_dict = samples["Local"][_ch_obj.basis]
                for slot in ch_samples.slots:
                    det_fluctuation = _generate_detuning_fluctuations(
                        self._noise_model,
                        traj.det_fluctuations[ch],
                        traj.det_phases[ch],
                        np.arange(0, self.samples.max_duration, 1),
                    )
                    add_noise(
                        slot,
                        samples_dict,
                        _ch_obj.addressing == "Global",
                        amp_fluctuation=traj.amp_fluctuations[ch],
                        det_fluctuation=det_fluctuation,
                        propagation_dir=_ch_obj.propagation_dir,
                    )
            channels = []
            samples_list = []
            ch_objs = {}
            # Set amplitude, detuning, phase to 0
            # over all sequence for badly prepared atoms
            for basis in samples["Local"]:
                if basis == "XY":
                    type: Channel = Microwave  # type: ignore
                elif basis == "ground-rydberg":
                    type: Channel = Rydberg  # type: ignore
                else:
                    type: Channel = Raman  # type: ignore
                qids = samples["Local"][basis].keys()
                basis_channels = list(f"{x}_{basis}" for x in qids)
                channels += basis_channels
                for qid, ch in zip(qids, basis_channels):
                    vals = samples["Local"][basis][qid]
                    if traj.bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            vals[qty] *= 0.0
                    samples_list.append(
                        ChannelSamples(
                            **vals,
                            slots=[
                                _PulseTargetSlot(
                                    ti=0, tf=len(vals["amp"]), targets={qid}
                                )
                            ],
                        )
                    )
                    ch_objs[ch] = type.Local(
                        max_abs_detuning=None, max_amp=None
                    )

            return SequenceSamples(
                _basis_ref=self._samples._basis_ref,
                _slm_mask=self._samples._slm_mask,
                _magnetic_field=self._samples._magnetic_field,
                _measurement=self._samples._measurement,
                channels=channels,
                samples_list=samples_list,
                _ch_objs=ch_objs,
            )
        else:
            return self._samples

    @property
    def noisy_samples(
        self,
    ) -> Iterator[SamplesWithReps]:
        """The noiseless samples modified by the noise trajectory."""
        # TODO: store mutiple trajectories and return iterator
        for traj, reps in self.noise_trajectories:
            yield SamplesWithReps(
                traj, self._sample_with_trajectory(traj), reps
            )

    @property
    def register(self) -> BaseRegister:
        """The noiseless register used."""
        return self._register

    @property
    def device(self) -> BaseDevice:
        """The device used."""
        return self._device

    @property
    def noise_model(self) -> NoiseModel:
        """The current NoiseModel used."""
        return self._noise_model

    def _interaction_matrix(
        self, register: BaseRegister
    ) -> "pm.torch.Tensor" | np.ndarray:
        r"""C6/C3 Interactions between the qudits (in :math:`rad/\mu s`).

        Uses data from self to determine the type of interaction and C6/C3.

        Args:
            register: The register for which to compute the interactions.

        Returns:
            The pairwise interaction coefficients, taking into account
            Device specs and the Sequence type.
        """
        # SLM mask is not included, because it's time-dependent
        d = _distances(register)
        interactions = pm.zeros_like(d)._array
        if self.basis_data.interaction_type == "XY":
            positions = list(register.qubits.values())
            assert self.samples._magnetic_field is not None
            assert self._device.interaction_coeff_xy is not None
            mag_arr = pm.AbstractArray(self.samples._magnetic_field)
            mag_norm = pm.norm(mag_arr)
            assert mag_norm > 0, "There must be a magnetic field in XY mode."
            for i in range(self.n_qudits):
                for j in range(i + 1, self.n_qudits):
                    diff = positions[i] - positions[j]
                    if len(diff) == 2:
                        diff = pm.hstack(
                            [diff, pm.AbstractArray(np.array(0.0))]
                        )
                    cosine = pm.dot(diff, mag_arr) / (pm.norm(diff) * mag_norm)
                    interactions[[i, j], [j, i]] = (
                        self._device.interaction_coeff_xy  # type: ignore
                        * (1 - 3 * cosine._array**2)
                        / d._array[i, j] ** 3
                    )
        else:
            for i in range(self.n_qudits):
                for j in range(i + 1, self.n_qudits):
                    interactions[[i, j], [j, i]] = (
                        self._device.interaction_coeff / d._array[i, j] ** 6
                    )
        return interactions

    @property
    def noisy_interaction_matrices(self) -> list[pm.AbstractArray]:
        """Get the noisy interaction matrix for each noise trajectory."""
        return [x[0].interaction_matrix for x in self.noise_trajectories]

    def _noisy_interaction_matrix(
        self, register: BaseRegister, bad_atoms: dict
    ) -> pm.AbstractArray:
        r"""C6/C3 Interactions between the qudits (in :math:`rad/\mu s`).

        Masks out missing qudits from the interaction.
        Uses data from self to determine the type of interaction and C6/C3.

        Args:
            register: The register for which to compute the interactions.
            bad_atoms: Which qudits are missing from the register.

        Returns:
            The pairwise interaction coefficients, taking into account
            Device specs, the Sequence type and missing atoms.
        """
        mask = [False for _ in range(self.n_qudits)]
        for ind, value in enumerate(bad_atoms.values()):
            mask[ind] = True if value else False  # convert to python bool
        imat = self._interaction_matrix(register)
        if isinstance(imat, np.ndarray):
            arr = np.array(mask)
            mask2 = arr.reshape(1, -1) | arr.reshape(-1, 1)
            mat = imat.copy()
            mat[mask2] = 0.0
            return pm.AbstractArray(mat)
        else:
            ten = pm.torch.tensor(mask, dtype=pm.torch.bool)
            mask3 = ten.reshape(1, -1) | ten.reshape(-1, 1)
            mat2 = imat.clone()
            mat2[mask3] = 0.0
            return pm.AbstractArray(mat2)

    def _build_local_collapse_operators(
        self,
        noise_model: NoiseModel,
        basis_name: str,
        eigenbasis: list[States],
        op_matrix: list[str],
    ) -> tuple[
        list[tuple[int | float | complex, str | np.ndarray]],
        dict[str, list[tuple[int | complex, str]]],
    ]:

        local_collapse_ops: list[
            tuple[int | float | complex, str | np.ndarray]
        ] = []
        depolarizing_pauli_2ds: dict[str, list[tuple[int | complex, str]]] = {}
        # TODO: Check that the relevant dephasing parameter is > 0.
        if "dephasing" in noise_model.noise_types:
            dephasing_rates = {
                "d": noise_model.dephasing_rate,
                "r": noise_model.dephasing_rate,
                "h": noise_model.hyperfine_dephasing_rate,
            }
            for state in eigenbasis:
                if state in dephasing_rates:
                    coeff = np.sqrt(2 * dephasing_rates[state])
                    op = f"sigma_{state}{state}"
                    assert op in op_matrix
                    local_collapse_ops.append((coeff, op))

        if "relaxation" in noise_model.noise_types:
            coeff = np.sqrt(noise_model.relaxation_rate)
            op = "sigma_gr"

            if op not in op_matrix:
                raise ValueError(
                    "'relaxation' noise requires addressing of the"
                    " 'ground-rydberg' basis."
                )
            local_collapse_ops.append((coeff, op))

        if "depolarizing" in noise_model.noise_types:
            if "all" in basis_name:
                # Go back to previous noise_model
                raise NotImplementedError(
                    "Cannot include depolarizing noise in all-basis."
                )
            # NOTE: These operators only make sense when basis != "all"
            b, a = eigenbasis[:2]
            depolarizing_pauli_2ds["x"] = [
                (1, f"sigma_{a}{b}"),
                (1, f"sigma_{b}{a}"),
            ]
            depolarizing_pauli_2ds["y"] = [
                (1j, f"sigma_{a}{b}"),
                (-1j, f"sigma_{b}{a}"),
            ]
            depolarizing_pauli_2ds["z"] = [
                (1, f"sigma_{b}{b}"),
                (-1, f"sigma_{a}{a}"),
            ]
            coeff = np.sqrt(noise_model.depolarizing_rate / 4)
            for pauli_label in depolarizing_pauli_2ds.keys():
                local_collapse_ops.append((coeff, pauli_label))

        if "eff_noise" in noise_model.noise_types:
            for id_, rate in enumerate(noise_model.eff_noise_rates):
                operator = noise_model.eff_noise_opers[id_]
                # This supports the case where the operators are given as Qobj
                # (even though they shouldn't per the NoiseModel signature)
                try:
                    operator = operator.full()  # type: ignore
                except AttributeError:
                    pass
                operator = np.array(operator)

                basis_dim = len(eigenbasis)
                op_shape = (basis_dim, basis_dim)
                if operator.shape != op_shape:
                    raise ValueError(
                        "Incompatible shape for effective noise operator n°"
                        f"{id_}. Operator {operator} should be of shape "
                        f"{op_shape}."
                    )
                local_collapse_ops.append((np.sqrt(rate), operator))
        # Building collapse operators
        return local_collapse_ops, depolarizing_pauli_2ds

    def _check_noise_model(self, noise_model: NoiseModel) -> None:
        """Checks that the provided noise_model is a NoiseModel."""
        if not isinstance(noise_model, NoiseModel):
            raise ValueError(
                f"Object {noise_model} is not a valid `NoiseModel`."
            )
        not_supported = (
            set(noise_model.noise_types)
            - SUPPORTED_NOISES[self.basis_data.interaction_type]
        )
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self.basis_data.interaction_type}' "
                "does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )

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

    def _create_noise_trajectories(
        self, ntrajs: int
    ) -> List[TrajectoryWithReps]:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        noise_trajectories: list[TrajectoryWithReps] = []
        amp_fluctuations: dict[str, float] = {}
        det_fluctuations: dict[str, float] = {}
        det_phases: dict[str, np.ndarray] = {}
        if not has_shot_to_shot_except_spam(self.noise_model):
            initial_configs = Counter(
                "".join(
                    (
                        np.random.uniform(size=len(self._qid_index))
                        < self.noise_model.state_prep_error
                    )
                    .astype(int)
                    .astype(str)  # Turns bool->int->str
                )
                for _ in range(ntrajs)
            ).most_common()

            doppler_detune = {qid: 0.0 for qid in self._qid_index}
            for ch in self._samples.channel_samples:
                assert self.noise_model.amp_sigma == 0.0
                amp_fluctuations[ch] = 1.0
                det_fluctuations[ch] = 0.0
                det_phases[ch] = np.array(0.0)
            for bool_string, n in initial_configs:
                bad_atoms = dict(
                    zip(self._qid_index, map(lambda x: x == "1", bool_string))
                )
                noise_trajectories.append(
                    TrajectoryWithReps(
                        NoiseTrajectory(
                            bad_atoms,
                            doppler_detune,
                            amp_fluctuations,
                            det_fluctuations,
                            det_phases,
                            self._register,
                            self._noisy_interaction_matrix(
                                self._register, bad_atoms
                            ),
                        ),
                        n,
                    )
                )
        else:
            for _ in range(ntrajs):
                amp_fluctuations = {}
                det_fluctuations = {}
                det_phases = {}
                register: BaseRegister = self._register
                if (
                    "SPAM" in self.noise_model.noise_types
                    and self.noise_model.state_prep_error > 0
                ):
                    dist = (
                        np.random.uniform(size=len(self._qid_index))
                        < self.noise_model.state_prep_error
                    )
                    bad_atoms = dict(zip(self._qid_index, dist))
                else:
                    bad_atoms = {qid: False for qid in self._qid_index}
                if "doppler" in self.noise_model.noise_types:
                    temp = self.noise_model.temperature * 1e-6
                    detune = np.random.normal(
                        0, doppler_sigma(temp), size=len(self._qid_index)
                    )
                    doppler_detune = dict(zip(self._qid_index, detune))
                else:
                    doppler_detune = {qid: 0.0 for qid in self._qid_index}
                for ch in self._samples.channel_samples:
                    amp_fluctuations[ch] = max(
                        0, np.random.normal(1.0, self.noise_model.amp_sigma)
                    )
                    det_fluctuations[ch] = (
                        np.random.normal(0.0, self.noise_model.detuning_sigma)
                        if self.noise_model.detuning_sigma
                        else 0.0
                    )
                    if self._noise_model.detuning_hf_omegas:
                        det_phases[ch] = np.random.uniform(
                            0.0,
                            2 * np.pi,
                            size=len(self._noise_model.detuning_hf_omegas) - 1,
                        )
                    else:
                        det_phases[ch] = np.array(0.0)
                if "register" in self._noise_model.noise_types:
                    register = _noisy_register(
                        self.register.qubits, self._noise_model
                    )
                noise_trajectories.append(
                    TrajectoryWithReps(
                        NoiseTrajectory(
                            bad_atoms,
                            doppler_detune,
                            amp_fluctuations,
                            det_fluctuations,
                            det_phases,
                            register,
                            self._noisy_interaction_matrix(
                                register, bad_atoms
                            ),
                        ),
                        1,
                    )
                )
        return noise_trajectories

    def _get_basis_name(self, with_leakage: bool) -> str:
        if len(self._samples.used_bases) == 0:
            if self._samples._in_xy:
                basis_name = "XY"
            else:
                basis_name = "ground-rydberg"
        elif len(self._samples.used_bases) == 1:
            basis_name = list(self._samples.used_bases)[0]
        else:
            basis_name = "all"  # All three rydberg states
        if with_leakage:
            basis_name += "_with_error"
        return basis_name

    def _get_eigenbasis(self, with_leakage: bool) -> list[States]:
        eigenbasis = self._samples.eigenbasis
        if with_leakage:
            eigenbasis.append("x")
        return [state for state in STATES_RANK if state in eigenbasis]

    @staticmethod
    def _get_projectors(
        eigenbasis: list[States],
    ) -> list[str]:
        """Determine projector operators."""
        op_matrix_names = ["I"]
        for proj0 in eigenbasis:
            for proj1 in eigenbasis:
                proj_name = "sigma_" + proj0 + proj1
                op_matrix_names.append(proj_name)
        return op_matrix_names
