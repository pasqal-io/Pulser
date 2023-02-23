"""The main function for sequence sampling."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pulser.sampler.samples import SequenceSamples, _SlmMask

if TYPE_CHECKING:
    from pulser import Sequence

IGNORE_DETUNED_DELAY_PHASE = True


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
    if seq.is_parametrized():
        raise NotImplementedError("Parametrized sequences can't be sampled.")

    samples_list = []
    for ch_schedule in seq._schedule.values():
        samples = ch_schedule.get_samples(IGNORE_DETUNED_DELAY_PHASE)
        if extended_duration:
            samples = samples.extend_duration(extended_duration)
        if modulation:
            samples = samples.modulate(
                ch_schedule.channel_obj,
                max_duration=extended_duration
                or ch_schedule.get_duration(include_fall_time=True),
            )
        samples_list.append(samples)

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
