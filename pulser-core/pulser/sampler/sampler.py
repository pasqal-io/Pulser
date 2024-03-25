"""The main function for sequence sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

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
        kwargs: dict[str, Any] = dict(
            ignore_detuned_delay_phase=IGNORE_DETUNED_DELAY_PHASE
        )
        if hasattr(ch_schedule, "detuning_map"):
            if seq.is_register_mappable():
                raise NotImplementedError(
                    "Sequences with a DMM channel can't be sampled while "
                    "their register is mappable."
                )
            kwargs["qubits"] = seq.register.qubits
        samples = ch_schedule.get_samples(**kwargs)
        if extended_duration:
            samples = samples.extend_duration(extended_duration)
        if modulation:
            samples = samples.modulate(
                ch_schedule.channel_obj,
                max_duration=extended_duration
                or ch_schedule.get_duration(include_fall_time=True),
            )
        samples_list.append(samples)

    optionals: dict = dict()
    if seq._slm_mask_targets and seq._slm_mask_time:
        optionals["_slm_mask"] = _SlmMask(
            seq._slm_mask_targets,
            seq._slm_mask_time[1],
        )
    if seq._in_xy:
        optionals["_magnetic_field"] = seq.magnetic_field
    if hasattr(seq, "_measurement"):
        # Has attribute measurement because sequence can't be parametrized
        optionals["_measurement"] = seq._measurement

    return SequenceSamples(
        list(seq.declared_channels.keys()),
        samples_list,
        seq.declared_channels,
        seq._basis_ref,
        **optionals,
    )
