"""Contains dataclasses for samples and some helper functions."""
from __future__ import annotations

import itertools
import typing
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

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


def _pairwise(iterable: typing.Iterable) -> typing.Iterable:
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _default_to_regular(d: dict | defaultdict) -> dict:
    """Helper function to convert defaultdicts to regular dicts."""
    if isinstance(d, defaultdict):
        d = {k: _default_to_regular(v) for k, v in d.items()}
    return d


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
        for t1, t2 in _pairwise(self.slots):
            assert t1.tf <= t2.ti  # no overlaps on a given channel


@dataclass
class SequenceSamples:
    """Gather samples of a sequence with useful info."""

    channels: list[str]
    channel_samples: dict[str, ChannelSamples]
    _addrs: dict[str, str]
    _bases: dict[str, str]
    _duration: int

    def to_nested_dict(self) -> dict:
        """Format in the nested dictionary form.

        This is the format expected by `pulser.simulation`.
        """
        d = _prepare_dict(self._duration)
        for chname, cs in self.channel_samples.items():
            addr = self._addrs[chname]
            basis = self._bases[chname]
            if addr == _GLOBAL:
                d[_GLOBAL][basis][_AMP] += cs.amp
                d[_GLOBAL][basis][_DET] += cs.det
                d[_GLOBAL][basis][_PHASE] += cs.phase
            else:
                for s in cs.slots:
                    for t in s.targets:
                        d[_LOCAL][basis][t][_AMP][s.ti : s.tf] += cs.amp
                        d[_LOCAL][basis][t][_DET][s.ti : s.tf] += cs.det
                        d[_LOCAL][basis][t][_PHASE][s.ti : s.tf] += cs.phase

        return _default_to_regular(d)

    def __repr__(self) -> str:
        s = ""
        for chname, cs in self.channel_samples.items():
            s += chname + ":\n"
            s += cs.__repr__()
            s += "\n\n"
        s += "\n"
        return s
