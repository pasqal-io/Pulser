"""Dataclasses for storing and processing the samples."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from pulser.channels.base_channel import Channel
from pulser.register import QubitId

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
class _TargetSlot:
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
    slots: list[_TargetSlot] = field(default_factory=list)
    # (t_start, t_end) of each EOM mode block
    eom_intervals: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert len(self.amp) == len(self.det) == len(self.phase)
        self.duration = len(self.amp)

        for t in self.slots:
            assert t.ti < t.tf  # well ordered slots
        for t1, t2 in zip(self.slots, self.slots[1:]):
            assert t1.tf <= t2.ti  # no overlaps on a given channel

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
        new_detuning = np.pad(self.det, (0, extension))
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

    def modulate(
        self, channel_obj: Channel, max_duration: Optional[int] = None
    ) -> ChannelSamples:
        """Modulates the samples for a given channel.

        It assumes that the phase starts at its initial value and is kept at
        its final value. The same could potentially be done for the detuning,
        but it's not as safe of an assumption so it's not done for now.

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
            new_samples[~mask] = 0
            return new_samples

        new_samples: dict[str, np.ndarray] = {}

        if self.eom_intervals:
            eom_mask = np.zeros(self.duration, dtype=bool)
            for start, end in self.eom_intervals:
                end = min(end, self.duration)  # This is defensive
                eom_mask[np.arange(start, end)] = True

            for key in ("amp", "det"):
                samples = getattr(self, key)
                std = channel_obj.modulate(masked(samples, ~eom_mask))
                eom = channel_obj.modulate(masked(samples, eom_mask), eom=True)
                sample_arrs = [std, eom]
                sample_arrs.sort(key=len)
                # Extend shortest array to match the longest
                sample_arrs[0] = np.concatenate(
                    (
                        sample_arrs[0],
                        np.zeros(sample_arrs[1].size - sample_arrs[0].size),
                    )
                )
                new_samples[key] = sample_arrs[0] + sample_arrs[1]

        else:
            new_samples["amp"] = channel_obj.modulate(self.amp)
            new_samples["det"] = channel_obj.modulate(self.det)

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
    _slm_mask: _SlmMask = field(default_factory=_SlmMask)

    @property
    def channel_samples(self) -> dict[str, ChannelSamples]:
        """Mapping between the channel name and its samples."""
        return dict(zip(self.channels, self.samples_list))

    @property
    def max_duration(self) -> int:
        """The maximum duration among the channel samples."""
        return max(samples.duration for samples in self.samples_list)

    def used_bases(self) -> set[str]:
        """The bases with non-zero pulses."""
        return {
            ch_obj.basis
            for ch_obj, ch_samples in zip(
                self._ch_objs.values(), self.samples_list
            )
            if not ch_samples.is_empty()
        }

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
        bases = {ch_obj.basis for ch_obj in self._ch_objs.values()}
        in_xy = False
        if "XY" in bases:
            assert bases == {"XY"}
            in_xy = True
        d = _prepare_dict(self.max_duration, in_xy=in_xy)
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
