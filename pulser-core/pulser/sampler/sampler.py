"""New version of the sequence sampler."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pulser.sampler.samples import SequenceSamples

if TYPE_CHECKING:  # pragma: no cover
    from pulser import Sequence


def sample(seq: Sequence, modulation: bool = False) -> SequenceSamples:
    """Construct samples of a Sequence.

    KNOWN BUG: Does not support modulation despite having a keyword for it .
    """
    return SequenceSamples(
        list(seq.declared_channels.keys()),
        [ch_schedule.get_samples() for ch_schedule in seq._schedule.values()],
        seq.declared_channels,
        seq.get_duration(),
    )
