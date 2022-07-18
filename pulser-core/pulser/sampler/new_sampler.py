"""New version of the sequence sampler."""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from pulser.pulse import Pulse
from pulser.sequence import QubitId, Sequence


"""Literal constants for addressing."""
_GLOBAL = "Global"
_LOCAL = "Local"


def _prepare_dict(N: int, in_xy: bool = False) -> dict:
    """Constructs empty dict of size N.

    Usually N is the duration of seq.
    """

    def new_qty_dict() -> dict:
        return {
            "amp": np.zeros(N),
            "det": np.zeros(N),
            "phase": np.zeros(N),
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
    if isinstance(d, defaultdict):
        d = {k: _default_to_regular(v) for k, v in d.items()}
    return d


class SequenceSamples:
    """Gather samples of a sequence with useful info."""

    def __init__(self, seq: Sequence) -> None:
        """Construct samples of a Sequence."""
        self._duration = seq.get_duration()
        self.channel = seq.declared_channels
        self._bases = {
            ch_name: self.channel[ch_name].basis for ch_name in self.channel
        }
        self._addrs = {
            ch_name: seq._channels[ch_name].addressing
            for ch_name in self.channel
        }
        self.channel_samples = {
            chname: _sample_channel(chname, seq) for chname in self.channel
        }

    def to_nested_dict(self) -> dict:
        """Format in the nested dictionary form.

        This is the format expected by `pulser.simulation`.
        """
        d = _prepare_dict(self._duration)
        for chname, cs in self.channel_samples.items():
            addr = self._addrs[chname]
            basis = self._bases[chname]
            if addr == _GLOBAL:
                d[_GLOBAL][basis]["amp"] += cs.amp
                d[_GLOBAL][basis]["det"] += cs.det
                d[_GLOBAL][basis]["phase"] += cs.phase
            else:
                for s in cs.slots:
                    for t in s.targets:
                        d[_LOCAL][basis][t]["amp"][s.ti : s.tf] += cs.amp
                        d[_LOCAL][basis][t]["det"][s.ti : s.tf] += cs.det
                        d[_LOCAL][basis][t]["phase"][s.ti : s.tf] += cs.phase

        return _default_to_regular(d)

    def __repr__(self) -> str:
        s = ""
        for chname, cs in self.channel_samples.items():
            s += chname + ":\n"
            s += cs.__repr__()
            s += "\n\n"
        s += "\n"
        return s


def sample(seq: Sequence) -> SequenceSamples:
    """Construct samples of a Sequence.

    Alternatively, use the SequenceSample constructor.
    """
    return SequenceSamples(seq)


@dataclass
class _TimeSlot:
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

    slots: list[_TimeSlot] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert len(self.amp) == len(self.det) == len(self.phase)

        for t in self.slots:
            assert t.ti < t.tf  # well ordered slots
        for t1, t2 in itertools.pairwise(self.slots):
            assert t1.tf <= t2.ti  # no overlaps on a given channel


def _sample_channel(chname: str, seq: Sequence) -> ChannelSamples:
    N = seq.get_duration()
    # Keep only pulse slots
    channel_slots = [
        s for s in seq._schedule[chname] if isinstance(s.type, Pulse)
    ]
    amp, det, phase = np.zeros(N), np.zeros(N), np.zeros(N)
    slots: list[_TimeSlot] = []

    for s in channel_slots:
        pulse = cast(Pulse, s.type)
        amp[s.ti : s.tf] += pulse.amplitude.samples
        det[s.ti : s.tf] += pulse.detuning.samples
        phase[s.ti : s.tf] += pulse.phase
        slots.append(_TimeSlot(s.ti, s.tf, s.targets))

    return ChannelSamples(amp, det, phase, slots)
