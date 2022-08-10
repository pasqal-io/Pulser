"""The main function for sequence sampling."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pulser.sampler.samples import SequenceSamples, _SlmMask

if TYPE_CHECKING:  # pragma: no cover
    from pulser import Sequence


def sample(
    seq: Sequence,
    modulation: bool = False,
    extended_duration: Optional[int] = None,
) -> SequenceSamples:
    """Construct samples of a Sequence.

    Args:
        seq: The sequence to sample.
        modulation: Whether to modulate the samples.
        extended_duration: If defined, extends the samples duration to the
            desired value.
    """
    samples_list = [
        ch_schedule.get_samples(modulated=modulation)
        for ch_schedule in seq._schedule.values()
    ]
    if extended_duration:
        samples_list = [
            cs.extend_duration(extended_duration) for cs in samples_list
        ]
    optionals = {}
    if seq._slm_mask_targets and seq._slm_mask_time:
        optionals["_slm_mask"] = _SlmMask(
            seq._slm_mask_targets,
            seq._slm_mask_time[1],
        )
    return SequenceSamples(
        list(seq.declared_channels.keys()),
        samples_list,
        seq.declared_channels,
        **optionals,
    )
