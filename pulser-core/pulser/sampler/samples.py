"""Dataclasses for storing and processing the samples."""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from pulser.channels.base_channel import Channel
from pulser.channels.eom import BaseEOM
from pulser.register import QubitId
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
            _AMP: np.zeros(N),
            _DET: np.zeros(N),
            _PHASE: np.zeros(N),
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

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray
    slots: list[_PulseTargetSlot] = field(default_factory=list)
    eom_blocks: list[_EOMSettings] = field(default_factory=list)
    eom_start_buffers: list[tuple[int, int]] = field(default_factory=list)
    eom_end_buffers: list[tuple[int, int]] = field(default_factory=list)
    target_time_slots: list[_TimeSlot] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert len(self.amp) == len(self.det) == len(self.phase)
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

        new_amp = np.pad(self.amp, (0, extension))
        # When in EOM mode, we need to keep the detuning at detuning_off
        if self.eom_blocks and self.eom_blocks[-1].tf is None:
            final_detuning = self.eom_blocks[-1].detuning_off
        else:
            final_detuning = 0.0
        new_detuning = np.pad(
            self.det,
            (0, extension),
            constant_values=(final_detuning,),
            mode="constant",
        )
        new_phase = np.pad(
            self.phase,
            (0, extension),
            mode="edge" if self.phase.size > 0 else "constant",
        )
        return replace(self, amp=new_amp, det=new_detuning, phase=new_phase)

    def is_empty(self) -> bool:
        """Whether the channel is effectively empty.

        The channel is considered empty if all amplitude and detuning
        samples are zero.
        """
        return np.count_nonzero(self.amp) + np.count_nonzero(self.det) == 0

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

        def masked(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
            new_samples = samples.copy()
            # Extend the mask to fit the size of the samples
            mask = np.pad(mask, (0, len(new_samples) - len(mask)), mode="edge")
            new_samples[~mask] = 0
            return new_samples

        new_samples: dict[str, np.ndarray] = {}

        eom_samples = {
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
                        key_samples, keep_ends=True
                    )
                else:
                    std_mask = ~eom_mask
                    modulated_buffer = np.zeros_like(modulated_std)

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
                        samples_ = np.insert(
                            samples_,
                            0,
                            self.eom_blocks[0].detuning_off,
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
                    arr = np.pad(
                        arr,
                        (0, sample_arrs[-1].size - arr.size),
                    )
                    new_samples[key] = new_samples[key] + arr

        else:
            new_samples["amp"] = channel_obj.modulate(self.amp)
            new_samples["det"] = channel_obj.modulate(self.det, keep_ends=True)

        new_samples["phase"] = channel_obj.modulate(self.phase, keep_ends=True)
        for key in new_samples:
            new_samples[key] = new_samples[key][slice(0, max_duration)]
        return replace(self, **new_samples)


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

    def to_nested_dict(self, all_local: bool = False) -> dict:
        """Format in the nested dictionary form.

        This is the format expected by `pulser_simulation.Simulation()`.

        Args:
            all_local: Forces all samples to be distributed by their
                individual targets, even when applied by a global channel.

        Returns:
            A nested dictionary splitting the samples according to their
            addressing ('Global' or 'Local'), the targeted basis
            and, in the 'Local' case, the targeted qubit.
        """
        d = _prepare_dict(self.max_duration, in_xy=self._in_xy)
        for chname, samples in zip(self.channels, self.samples_list):
            cs = (
                samples.extend_duration(self.max_duration)
                if samples.duration != self.max_duration
                else samples
            )
            addr = self._ch_objs[chname].addressing
            basis = self._ch_objs[chname].basis
            if addr == _GLOBAL and not all_local:
                start_t = self._slm_mask.end
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
                        if t in self._slm_mask.targets:
                            ti = max(ti, self._slm_mask.end)
                        times = slice(ti, s.tf)
                        d[_LOCAL][basis][t][_AMP][times] += cs.amp[times]
                        d[_LOCAL][basis][t][_DET][times] += cs.det[times]
                        d[_LOCAL][basis][t][_PHASE][times] += cs.phase[times]

        return _default_to_regular(d)

    def __repr__(self) -> str:
        blocks = [
            f"{chname}:\n{cs!r}"
            for chname, cs in zip(self.channels, self.samples_list)
        ]
        return "\n\n".join(blocks)


# This is just to preserve backwards compatibility after the renaming of
# _TargetSlot to _PulseTarget slot
_TargetSlot = _PulseTargetSlot
