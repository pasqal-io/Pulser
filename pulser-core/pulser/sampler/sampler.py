"""Defines the main function for sequence sampling."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pulser.sampler.samples import SequenceSamples

if TYPE_CHECKING:  # pragma: no cover
    from pulser import Sequence


def sample(seq: Sequence, modulation: bool = False) -> SequenceSamples:
    """Construct samples of a Sequence.

    Args:
        seq: The sequence to sample.
        modulation: Whether to modulate the samples.
    """
    return SequenceSamples(
        list(seq.declared_channels.keys()),
        [
            ch_schedule.get_samples(modulated=modulation)
            for ch_schedule in seq._schedule.values()
        ],
        seq.declared_channels,
    )
