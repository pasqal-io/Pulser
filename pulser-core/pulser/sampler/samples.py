"""Dataclasses for storing and processing the samples."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Optional, cast, get_args

import numpy as np

import pulser.math as pm
from pulser.channels.base_channel import (
    EIGENSTATES,
    Channel,
    States,
    get_states_from_bases,
)
from pulser.channels.eom import BaseEOM
from pulser.register import QubitId
from pulser.register.weight_maps import DetuningMap
from pulser.sequence._basis_ref import _QubitRef

if TYPE_CHECKING:
    from pulser.sequence._schedule import _EOMSettings, _TimeSlot

"""Literal constants for addressing."""
_GLOBAL = "Global"
_LOCAL = "Local"
_AMP = "amp"
_DET = "det"
_PHASE = "phase"


def _prepare_dict(N: int, in_xy: bool = False) -> dict:
    """Constructs empty dict of size N.

    Usually N is the duration of seq.
    """

    def new_qty_dict() -> dict:
        return {
            _AMP: pm.AbstractArray(np.zeros(N)),
            _DET: pm.AbstractArray(np.zeros(N)),
            _PHASE: pm.AbstractArray(np.zeros(N)),
        }

    def new_qdict() -> dict:
        return defaultdict(new_qty_dict)

    if in_xy:
        return {
            _GLOBAL: {"XY": new_qty_dict()},
            _LOCAL: {"XY": new_qdict()},
        }
    else:
        return {
            _GLOBAL: defaultdict(new_qty_dict),
            _LOCAL: defaultdict(new_qdict),
        }


def _default_to_regular(d: dict | defaultdict) -> dict:
    """Helper function to convert defaultdicts to regular dicts."""
    if isinstance(d, dict):
        d = {k: _default_to_regular(v) for k, v in d.items()}
    return d


@dataclass
class _PulseTargetSlot:
    """Auxiliary class to store target information.

    Recopy of the sequence._TimeSlot but without the unrelevant `type` field,
    unrelevant at the sample level.

    NOTE: While it store targets, targets themselves are insufficient to
    conclude on the addressing of the samples. Additional info is needed:
    compare against a known register or the original sequence information.
    """

    ti: int
    tf: int
    targets: set[QubitId]


@dataclass
class _SlmMask:
    """Auxiliary class to store the SLM mask configuration."""

    targets: set[QubitId] = field(default_factory=set)
    end: int = 0


@dataclass
class ChannelSamples:
    """Gathers samples of a channel."""

    amp: pm.AbstractArray
    det: pm.AbstractArray
    phase: pm.AbstractArray
    slots: list[_PulseTargetSlot] = field(default_factory=list)
    eom_blocks: list[_EOMSettings] = field(default_factory=list)
    eom_start_buffers: list[tuple[int, int]] = field(default_factory=list)
    eom_end_buffers: list[tuple[int, int]] = field(default_factory=list)
    target_time_slots: list[_TimeSlot] = field(default_factory=list)
    _centered_phase: pm.AbstractArray | None = None

    def __post_init__(self) -> None:
        assert (
            len(self.amp)
            == len(self.det)
            == len(self.phase)
            == len(self.centered_phase)
        )
        self.duration = len(self.amp)

        for t in self.slots:
            assert t.ti < t.tf  # well ordered slots
        for t1, t2 in zip(self.slots, self.slots[1:]):
            assert t1.tf <= t2.ti  # no overlaps on a given channel

    @property
    def initial_targets(self) -> set[QubitId]:
        """Returns the initial targets."""
        return (
            self.target_time_slots[0].targets
            if self.target_time_slots
            else set()
        )

    @property
    def centered_phase(self) -> pm.AbstractArray:
        """The phase samples centered in ]-π, π]."""
        if self._centered_phase is not None:
            return self._centered_phase
        phase_ = self.phase.copy() % (2 * np.pi)
        phase_[phase_ > np.pi] -= 2 * np.pi
        return phase_

    @property
    def phase_modulation(self) -> pm.AbstractArray:
        r"""The phase modulation samples (in rad).

        Constructed by combining the integral of the detuning samples with the
        phase offset samples according to

        .. math:: \phi(t) = \phi_c(t) - \sum_{k=0}^{t} \delta(k)
        """
        return self.centered_phase - pm.cumsum(self.det * 1e-3)

    def extend_duration(self, new_duration: int) -> ChannelSamples:
        """Extends the duration of the samples.

        Pads the amplitude and detuning samples with zeros and the phase with
        its last value (or zero if empty).

        Args:
            new_duration: The new duration for the samples (in ns).
                Must be greater than or equal to the current duration.

        Returns:
            The extended channel samples.
        """
        extension = new_duration - self.duration
        if extension < 0:
            raise ValueError("Can't extend samples to a lower duration.")

        new_amp = pm.pad(self.amp, (0, extension))
        # When in EOM mode, we need to keep the detuning at detuning_off
        if self.eom_blocks and self.eom_blocks[-1].tf is None:
            final_detuning = float(self.eom_blocks[-1].detuning_off)
        else:
            final_detuning = 0.0
        new_detuning = pm.pad(
            self.det,
            (0, extension),
            mode="constant",
            constant_values=final_detuning,
        )
        new_phase = pm.pad(
            self.phase,
            (0, extension),
            mode="edge" if self.phase.size > 0 else "constant",
        )
        _new_centered_phase = None
        if self._centered_phase is not None:
            _new_centered_phase = pm.pad(
                self._centered_phase,
                (0, extension),
                mode="edge" if self._centered_phase.size > 0 else "constant",
            )

        return replace(
            self,
            amp=new_amp,
            det=new_detuning,
            phase=new_phase,
            _centered_phase=_new_centered_phase,
        )

    def is_empty(self) -> bool:
        """Whether the channel is effectively empty.

        The channel is considered empty if all amplitude and detuning
        samples are zero.
        """
        return (
            np.count_nonzero(self.amp.as_array(detach=True))
            + np.count_nonzero(self.det.as_array(detach=True))
            == 0
        )

    def _generate_std_samples(self) -> ChannelSamples:
        new_samples = {
            key: getattr(self, key).copy() for key in ("amp", "det")
        }
        for block in self.eom_blocks:
            region = slice(block.ti, block.tf)
            new_samples["amp"][region] = 0
            # For modulation purposes, the detuning on the standard
            # samples is kept at 'detuning_off', which permits a smooth
            # transition to/from the EOM modulated samples
            new_samples["det"][region] = block.detuning_off

        return replace(self, **new_samples)

    def get_eom_mode_intervals(self) -> list[tuple[int, int]]:
        """Returns EOM mode intervals."""
        return [
            (
                block.ti,
                block.tf if block.tf is not None else self.duration,
            )
            for block in self.eom_blocks
        ]

    def in_eom_mode(self, slot: _TimeSlot | _PulseTargetSlot) -> bool:
        """States if a time slot is inside an EOM mode block."""
        return any(
            start <= slot.ti < end
            for start, end in self.get_eom_mode_intervals()
        )

    def modulate(
        self, channel_obj: Channel, max_duration: Optional[int] = None
    ) -> ChannelSamples:
        """Modulates the samples for a given channel.

        It assumes that the detuning and phase start at their initial values
        and are kept at their final values.

        Args:
            channel_obj: The channel object for which to modulate the samples.
            max_duration: The maximum duration of the modulation samples. If
                defined, truncates them to have a duration less than or equal
                to the given value.

        Returns:
            The modulated channel samples.
        """

        def masked(
            samples: pm.AbstractArray,
            mask: np.ndarray,
            keep_end_values: bool = False,
        ) -> pm.AbstractArray:
            new_samples = samples.copy()
            # Extend the mask to fit the size of the samples
            mask = np.pad(mask, (0, len(new_samples) - len(mask)), mode="edge")
            if keep_end_values:
                # Extracts the contiguous masked regions as [ti, tf] pairs
                masked_regions = (
                    np.flatnonzero(
                        np.diff(
                            np.r_[
                                np.int8(0), (~mask).view(np.int8), np.int8(0)
                            ]
                        )
                    )
                    .reshape(-1, 2)
                    .tolist()
                )
                for reg in masked_regions:
                    if not (delta := reg[1] - reg[0]):
                        # Should never happen, added as a precaution
                        continue  # pragma: no cover
                    # Set the masked region to the final sample value
                    new_samples[reg[0] : reg[1]] = samples[reg[1] - 1]
                    if reg[0] > 0:
                        # If not starting from 0, set the first half of
                        # the region to the first sample value
                        new_samples[reg[0] : reg[0] + delta // 2] = samples[
                            reg[0]
                        ]
            else:
                new_samples[~mask] = 0
            return new_samples

        new_samples: dict[str, pm.AbstractArray] = {}

        eom_samples: dict[str, pm.AbstractArray] = {
            key: getattr(self, key).copy() for key in ("amp", "det")
        }

        if self.eom_blocks:
            std_samples = self._generate_std_samples()
            # Note: self.duration already includes the fall time
            eom_mask = np.zeros(self.duration, dtype=bool)
            # Extension of the EOM mask outside of the EOM interval
            eom_mask_ext = eom_mask.copy()
            eom_fall_time = 2 * cast(BaseEOM, channel_obj.eom_config).rise_time
            for block in self.eom_blocks:
                # If block.tf is None, uses the full duration as the tf
                end = block.tf or self.duration
                eom_mask[block.ti : end] = True
                # Extends EOM masks to include fall time
                ext_end = end + eom_fall_time
                eom_mask_ext[end:ext_end] = True

            # We need 'eom_mask_ext' on its own, but we can already add it
            # to the 'eom_mask'
            eom_mask = eom_mask + eom_mask_ext

            eom_buffers_mask = np.zeros_like(eom_mask, dtype=bool)
            for start, end in itertools.chain(
                self.eom_start_buffers, self.eom_end_buffers
            ):
                eom_buffers_mask[start:end] = True
            eom_buffers_mask = eom_buffers_mask & ~eom_mask_ext
            buffer_ch_obj = replace(
                channel_obj,
                mod_bandwidth=channel_obj._eom_buffer_mod_bandwidth,
            )

            if block.tf is None:
                # The sequence finishes in EOM mode, so 'end' was already
                # including the fall time (unlike when it is disabled).
                # For modulation, we make the detuning during the last
                # fall time to be kept at 'detuning_off'
                eom_samples["det"][-eom_fall_time:] = block.detuning_off

            for key in ("amp", "det"):
                # First, we modulated the pre-filtered standard samples, then
                # we mask them to include only the parts outside the EOM mask
                # This ensures smooth transitions between EOM and STD samples
                key_samples = getattr(std_samples, key)
                modulated_std = channel_obj.modulate(
                    key_samples, keep_ends=key == "det"
                )
                if key == "det":
                    std_mask = ~(eom_mask + eom_buffers_mask)
                    # Adjusted detuning modulation during EOM buffers
                    modulated_buffer = buffer_ch_obj.modulate(
                        # Makes detuning constant before and after EOM blocks
                        # for a smooth transition
                        masked(key_samples, ~std_mask, keep_end_values=True),
                        keep_ends=True,
                    )
                else:
                    std_mask = ~eom_mask
                    modulated_buffer = pm.AbstractArray(modulated_std) * 0.0

                std = masked(modulated_std, std_mask)
                buffers = masked(
                    modulated_buffer[: len(std)], eom_buffers_mask
                )

                # At the end of an EOM block, the EOM(s) are switched back
                # to the OFF configuration, so the detuning should go quickly
                # back to `detuning_off`.
                # However, the applied detuning and the lightshift are
                # simultaneously being ramped to zero, so the fast ramp doesn't
                # reach `detuning_off` but rather a modified detuning value
                # (closer to zero). Then, the detuning goes slowly
                # to zero (as dictacted by the standard modulation bandwidth).
                # To mimick this effect, we substitute the detuning at the end
                # of each block by the standard modulated detuning during the
                # transition period, so the EOM modulation is superimposed on
                # the standard modulation
                if key == "det":
                    samples_ = eom_samples[key]
                    samples_[eom_mask_ext] = modulated_std[
                        : len(eom_mask_ext)
                    ][eom_mask_ext]
                    # Starts out in EOM mode, so we prepend 'detuning_off'
                    # such that the modulation starts off from that value
                    # We then remove the extra value after modulation
                    if eom_mask[0]:
                        samples_ = pm.pad(
                            samples_,
                            (1, 0),
                            "constant",
                            constant_values=float(
                                self.eom_blocks[0].detuning_off
                            ),
                        )
                    # Finally, the modified EOM samples are modulated
                    modulated_eom = channel_obj.modulate(
                        samples_, eom=True, keep_ends=True
                    )[(1 if eom_mask[0] else 0) :]
                else:
                    modulated_eom = channel_obj.modulate(
                        eom_samples[key], eom=True
                    )

                # filtered to include only the parts inside the EOM mask
                eom = masked(modulated_eom, eom_mask)

                # 'std', 'eom' and 'buffers' are then summed, but before the
                # short arrays are extended so that they are of the same length
                sample_arrs = [std, eom, buffers]
                sample_arrs.sort(key=len)
                # Extend shortest arrays to match the longest before summing
                new_samples[key] = sample_arrs[-1]
                for arr in sample_arrs[:-1]:
                    arr = pm.pad(
                        arr,
                        (0, sample_arrs[-1].size - arr.size),
                    )
                    new_samples[key] = new_samples[key] + arr

        else:
            new_samples["amp"] = channel_obj.modulate(self.amp)
            new_samples["det"] = channel_obj.modulate(self.det, keep_ends=True)

        new_len_ = len(new_samples["amp"])
        new_samples["phase"] = pm.pad(
            self.phase,
            (0, new_len_ - len(self.phase)),
            mode="edge",
        )
        new_samples["_centered_phase"] = pm.pad(
            self.centered_phase,
            (0, new_len_ - len(self.centered_phase)),
            mode="edge",
        )
        for key in new_samples:
            new_samples[key] = new_samples[key].astype(float)[
                slice(0, max_duration)
            ]
        return replace(self, **new_samples)


@dataclass
class DMMSamples(ChannelSamples):
    """Gathers samples of a DMM channel."""

    # TODO: Make these arguments KW_ONLY once python >= 3.10
    # Although these shouldn't have a default, in this way we can
    # subclass ChannelSamples
    detuning_map: DetuningMap | None = None
    qubits: dict[QubitId, pm.AbstractArray] = field(default_factory=dict)


_SamplesType = Literal["abstract", "array", "tensor"]


@dataclass
class SequenceSamples:
    """Gather samples for each channel in a sequence."""

    channels: list[str]
    samples_list: list[ChannelSamples]
    _ch_objs: dict[str, Channel]
    _basis_ref: dict[str, dict[QubitId, _QubitRef]] = field(
        default_factory=dict
    )
    _slm_mask: _SlmMask = field(default_factory=_SlmMask)
    _magnetic_field: np.ndarray | None = None
    _measurement: str | None = None

    @property
    def channel_samples(self) -> dict[str, ChannelSamples]:
        """Mapping between the channel name and its samples."""
        return dict(zip(self.channels, self.samples_list))

    @property
    def max_duration(self) -> int:
        """The maximum duration among the channel samples."""
        return max(samples.duration for samples in self.samples_list)

    @property
    def used_bases(self) -> set[str]:
        """The bases with non-zero pulses."""
        return {
            ch_obj.basis
            for ch_obj, ch_samples in zip(
                self._ch_objs.values(), self.samples_list
            )
            if not ch_samples.is_empty()
        }

    @property
    def eigenbasis(self) -> list[States]:
        """The basis of eigenstates used for simulation."""
        if len(self.used_bases) == 0:
            return EIGENSTATES["XY" if self._in_xy else "ground-rydberg"]
        return get_states_from_bases(self.used_bases)

    @property
    def _in_xy(self) -> bool:
        """Checks if the sequence is in XY mode."""
        bases = {ch_obj.basis for ch_obj in self._ch_objs.values()}
        in_xy = False
        if "XY" in bases:
            assert bases == {"XY"}
            in_xy = True
        return in_xy

    def extend_duration(self, new_duration: int) -> SequenceSamples:
        """Extend the duration of each samples to a new duration."""
        return replace(
            self,
            samples_list=[
                sample.extend_duration(new_duration)
                for sample in self.samples_list
            ],
        )

    def to_nested_dict(
        self,
        all_local: bool = False,
        samples_type: _SamplesType = "array",
    ) -> dict:
        """Format in the nested dictionary form.

        This is the format expected by `pulser_simulation.QutipEmulator()`.

        Args:
            all_local: Forces all samples to be distributed by their
                individual targets, even when applied by a global channel.
            samples_type: The array type to return the samples in. Can be
                "array" (the default), "tensor" or "abstract".

        Returns:
            A nested dictionary splitting the samples according to their
            addressing ('Global' or 'Local'), the targeted basis
            and, in the 'Local' case, the targeted qubit.
        """
        _samples_type_options = get_args(_SamplesType)
        if samples_type not in _samples_type_options:
            raise ValueError(
                f"'samples_type' must be one of {_samples_type_options!r}, "
                f"not {samples_type!r}."
            )

        d = _prepare_dict(self.max_duration, in_xy=self._in_xy)
        for chname, samples in zip(self.channels, self.samples_list):
            cs = (
                samples.extend_duration(self.max_duration)
                if samples.duration != self.max_duration
                else samples
            )
            addr = self._ch_objs[chname].addressing
            basis = self._ch_objs[chname].basis
            is_dmm = isinstance(samples, DMMSamples)
            in_xy = basis == "XY"
            if is_dmm:
                samples = cast(DMMSamples, samples)
                det_map = cast(DetuningMap, samples.detuning_map)
                det_weight_map = defaultdict(
                    int, det_map.get_qubit_weight_map(samples.qubits)
                )
            else:
                det_weight_map = defaultdict(lambda: 1.0)
            if addr == _GLOBAL and not all_local and not is_dmm:
                start_t = self._slm_mask.end if in_xy else 0
                d[_GLOBAL][basis][_AMP][start_t:] += cs.amp[start_t:]
                d[_GLOBAL][basis][_DET][start_t:] += cs.det[start_t:]
                d[_GLOBAL][basis][_PHASE][start_t:] += cs.phase[start_t:]
                if start_t == 0:
                    # Prevents lines below from running unnecessarily
                    continue
                unmasked_targets = cs.slots[0].targets - self._slm_mask.targets
                for t in unmasked_targets:
                    d[_LOCAL][basis][t][_AMP][:start_t] += cs.amp[:start_t]
                    d[_LOCAL][basis][t][_DET][:start_t] += cs.det[:start_t]
                    d[_LOCAL][basis][t][_PHASE][:start_t] += cs.phase[:start_t]
            else:
                if not cs.slots:
                    # Fill the defaultdict entries to not return an empty dict
                    for t in cs.initial_targets:
                        d[_LOCAL][basis][t]
                for s in cs.slots:
                    for t in s.targets:
                        ti = s.ti
                        if in_xy and t in self._slm_mask.targets:
                            ti = max(ti, self._slm_mask.end)
                        times = slice(ti, s.tf)
                        d[_LOCAL][basis][t][_AMP][times] += cs.amp[times]
                        d[_LOCAL][basis][t][_DET][times] += (
                            cs.det[times] * det_weight_map[t]
                        )
                        d[_LOCAL][basis][t][_PHASE][times] += cs.phase[times]

        regular_dict = _default_to_regular(d)

        def cast_arrays(arr_dict: dict) -> dict:
            for k in arr_dict:
                if isinstance(arr_dict[k], dict):
                    arr_dict[k] = cast_arrays(arr_dict[k])
                    continue
                assert isinstance(arr := arr_dict[k], pm.AbstractArray)
                arr_dict[k] = (
                    arr.as_tensor()
                    if samples_type == "tensor"
                    else arr.as_array(detach=True)
                )
            return arr_dict

        if samples_type != "abstract":
            regular_dict = cast_arrays(regular_dict)

        return regular_dict

    def __repr__(self) -> str:
        blocks = [
            f"{chname}:\n{cs!r}"
            for chname, cs in zip(self.channels, self.samples_list)
        ]
        return "\n\n".join(blocks)


# This is just to preserve backwards compatibility after the renaming of
# _TargetSlot to _PulseTargetSlot
_TargetSlot = _PulseTargetSlot
