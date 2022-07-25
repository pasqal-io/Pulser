"""Contains dataclasses for samples and some helper functions."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pulser.register import QubitId
from pulser.channels import Channel

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
class ChannelSamples:
    """Gathers samples of a channel."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray
    slots: list[_TargetSlot] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert len(self.amp) == len(self.det) == len(self.phase)
        self.duration = len(self.amp)

        for t in self.slots:
            assert t.ti < t.tf  # well ordered slots
        for t1, t2 in zip(self.slots, self.slots[1:]):
            assert t1.tf <= t2.ti  # no overlaps on a given channel

    def extend_duration(self, new_duration: int) -> ChannelSamples:
        """Extends the duration of the samples.

        Args:
            new_duration: The new duration for the samples (in ns).
                Must be greater than or equal to the current duration.

        Returns:
            The extend channel samples.
        """
        extension = new_duration - len(self.amp)
        if new_duration < self.duration:
            raise ValueError("Can't extend samples to a lower duration.")

        new_amp = np.pad(self.amp, (0, extension))
        new_detuning = np.pad(self.det, (0, extension))
        new_phase = np.pad(self.phase, (0, extension), mode="edge")
        return ChannelSamples(new_amp, new_detuning, new_phase, self.slots)

    def modulate(self, channel_obj: Channel) -> ChannelSamples:
        """Modulates the samples for a given channel.

        Args:
            channel_obj: The channel object for which to modulate the samples.

        Returns:
            The modulated channel samples.
        """
        if not isinstance(channel_obj, Channel):
            raise TypeError("'channel_obj' must be a Channel instance.")

        new_amp = channel_obj.modulate(self.amp)
        new_detuning = channel_obj.modulate(self.det, keep_ends=True)
        new_phase = channel_obj.modulate(self.phase, keep_ends=True)
        return ChannelSamples(new_amp, new_detuning, new_phase, self.slots)


@dataclass
class SequenceSamples:
    """Gather samples of a sequence with useful info."""

    channels: list[str]
    channel_samples: list[ChannelSamples]
    _ch_objs: dict[str, Channel]

    @property
    def duration(self) -> int:
        """The maximum duration among the channel samples."""
        return max(samples.duration for samples in self.channel_samples)

    def to_nested_dict(self) -> dict:
        """Format in the nested dictionary form.

        This is the format expected by `pulser_simulation.Simulation()`.
        """
        bases = {ch_obj.basis for ch_obj in self._ch_objs.values()}
        in_xy = False
        if "XY" in bases:
            assert bases == {"XY"}
            in_xy = True
        d = _prepare_dict(self.duration, in_xy=in_xy)
        for chname, samples in zip(self.channels, self.channel_samples):
            cs = samples.extend_duration(self.duration)
            addr = self._ch_objs[chname].addressing
            basis = self._ch_objs[chname].basis
            if addr == _GLOBAL:
                d[_GLOBAL][basis][_AMP] += cs.amp
                d[_GLOBAL][basis][_DET] += cs.det
                d[_GLOBAL][basis][_PHASE] += cs.phase
            else:
                for s in cs.slots:
                    for t in s.targets:
                        times = slice(s.ti, s.tf)
                        d[_LOCAL][basis][t][_AMP][times] += cs.amp[times]
                        d[_LOCAL][basis][t][_DET][times] += cs.det[times]
                        d[_LOCAL][basis][t][_PHASE][times] += cs.phase[times]

        return _default_to_regular(d)

    def __repr__(self) -> str:
        blocks = [
            f"{chname}:\n{cs!r}"
            for chname, cs in zip(self.channels, self.channel_samples)
        ]
        return "\n\n".join(blocks)
