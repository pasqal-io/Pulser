"""Exposes the sample() functions.

It contains many helpers.
"""
from __future__ import annotations

import enum
import itertools
from typing import Callable, List, Optional, cast

import numpy as np

import pulser.sampler.noises as noises
from pulser.channels import Channel
from pulser.pulse import Pulse
from pulser.sampler.noises import NoiseModel
from pulser.sampler.samples import QubitSamples, Samples
from pulser.sequence import QubitId, Sequence, _TimeSlot


def sample(
    seq: Sequence,
    modulation: bool = False,
    common_noises: Optional[list[NoiseModel]] = None,
    global_noises: Optional[list[NoiseModel]] = None,
) -> dict:
    """Samples the given Sequence and returns a nested dictionary.

    Args:
        seq (Sequence): a pulser.Sequence instance.
        modulation (bool): a flag to account for the modulation of AOM/EOM
            before sampling.
        common_noises (Optional[list[LocalNoise]]): a list of the noise sources
            for all channels.
        global_noises (Optional[list[LocalNoise]]): a list of the noise sources
            for global channels.

    Returns:
        A nested dictionnary of the samples of the amplitude, detuning and
        phase at every nanoseconds for all channels.
    """
    if common_noises is None:
        common_noises = []
    if global_noises is None:
        global_noises = []
    # The noises are applied in the reversed order of the list

    d = _prepare_dict(seq, seq.get_duration())
    if modulation:
        max_rt = max([ch.rise_time for ch in seq.declared_channels.values()])
        d = _prepare_dict(seq, seq.get_duration() + 2 * max_rt)

    for ch_name in seq.declared_channels:
        addr = seq.declared_channels[ch_name].addressing
        basis = seq.declared_channels[ch_name].basis

        ls: list[QubitSamples] = []

        if addr == "Global":
            # 1. determine if the global channel decay to a local one
            # 2. extract samples
            # 3. modulate
            # 4. apply noises/SLM
            # 5. write samples

            decay = (
                len(seq._slm_mask_targets) > 0
                or len(global_noises) > 0
                or len(common_noises) > 0
            )

            gs = _sample_global_channel(seq, ch_name)
            if modulation:
                gs = _modulate_global(seq.declared_channels[ch_name], gs)

            if not decay:
                _write_global_samples(d, basis, gs)
            else:
                qs = seq._qids - seq._slm_mask_targets
                ls = [QubitSamples.from_global(q, gs) for q in qs]
                ls = noises.apply(ls, common_noises + global_noises)
                _write_local_samples(d, basis, ls)

        elif addr == "Local":
            # 1. determine if modulation
            # 2. extract samples (depends on modulation)
            # 3. apply noises/SLM
            # 4. write samples
            if modulation:
                # gatering the samples for consecutive pulses with same targets
                # that can be safely modulated.
                ls = _sample_local_channel(
                    seq, ch_name, _group_between_retarget
                )
                ls = _modulate_local(seq.declared_channels[ch_name], ls)
            else:
                ls = _sample_local_channel(seq, ch_name, _regular)
            qs = seq._qids - seq._slm_mask_targets
            ls = [s for s in ls if s.qubit in qs]
            ls = noises.apply(ls, common_noises)
            _write_local_samples(d, basis, ls)

    return d


def _write_global_samples(d: dict, basis: str, samples: Samples) -> None:
    d["Global"][basis]["amp"] += samples.amp
    d["Global"][basis]["det"] += samples.det
    d["Global"][basis]["phase"] += samples.phase


def _write_local_samples(
    d: dict, basis: str, samples: list[QubitSamples]
) -> None:
    for s in samples:
        d["Local"][basis][s.qubit]["amp"] += s.amp
        d["Local"][basis][s.qubit]["det"] += s.det
        d["Local"][basis][s.qubit]["phase"] += s.phase


def _prepare_dict(seq: Sequence, N: int) -> dict:
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
        return {qubit: new_qty_dict() for qubit in seq._qids}

    if seq._in_xy:
        return {
            "Global": {"XY", new_qty_dict()},
            "Local": {"XY": new_qdict()},
        }
    else:
        return {
            "Global": {
                basis: new_qty_dict()
                for basis in ["ground-rydberg", "digital"]
            },
            "Local": {
                basis: new_qdict() for basis in ["ground-rydberg", "digital"]
            },
        }


def _sample_global_channel(seq: Sequence, ch_name: str) -> Samples:
    """Compute Samples for a global channel."""
    if seq.declared_channels[ch_name].addressing != "Global":
        raise ValueError(f"{ch_name} is no a global channel")
    slots = seq._schedule[ch_name]
    return _sample_slots(seq.get_duration(), *slots)


def _sample_local_channel(
    seq: Sequence, ch_name: str, strategy: TimeSlotExtractionStrategy
) -> list[QubitSamples]:
    """Compute Samples for a local channel."""
    if seq.declared_channels[ch_name].addressing != "Local":
        raise ValueError(f"{ch_name} is no a local channel")

    return strategy(seq.get_duration(), seq._schedule[ch_name])


def _sample_slots(N: int, *slots: _TimeSlot) -> Samples:
    """Gather samples of a list of _TimeSlot in a single Samples instance.

    Args:
        N (int): the size of the samples arrays.
        *slots (tuple[_TimeSlots]): the _TimeSlots to sample

    Returns:
        A Samples instance.
    """
    samples = Samples(np.zeros(N), np.zeros(N), np.zeros(N))
    for s in slots:
        if type(s.type) is str:
            continue
        pulse = cast(Pulse, s.type)
        samples.amp[s.ti : s.tf] += pulse.amplitude.samples
        samples.det[s.ti : s.tf] += pulse.detuning.samples
        samples.phase[s.ti : s.tf] += pulse.phase

    return samples


TimeSlotExtractionStrategy = Callable[
    [int, List[_TimeSlot]], List[QubitSamples]
]
"""Extraction strategy of _TimeSlot's of a Channel.

This strategy type is used mostly for the necessity to extract samples
differently when taking into account the modulation of AOM/EOM. Despite there
are only two cases, whether it's necessary to modulate a local channel or not,
this pattern can accomodate for future needs.
"""


def _regular(N: int, ts: list[_TimeSlot]) -> list[QubitSamples]:
    """No grouping performed.

    Fallback on the extraction procedure for a Global-like channel, and create
    QubitSamples for each targeted qubit from the result.
    """
    return [
        QubitSamples.from_global(q, _sample_slots(N, slot))
        for slot in ts
        for q in slot.targets
    ]


def _group_between_retarget(N: int, ts: list[_TimeSlot]) -> list[QubitSamples]:
    qs: list[QubitSamples] = []
    """Group a list of _TimeSlot by consecutive Pulse between retarget ops."""
    grouped_slots = _consecutive_slots_between_retargets(ts)
    for targets, group in grouped_slots:
        ss = [
            QubitSamples.from_global(q, _sample_slots(N, *group))
            for q in targets
        ]
        qs.extend(ss)
    return qs


class _GroupType(enum.Enum):
    PULSE_AND_DELAYS = "pulses_and_delays"
    TARGET = "target"
    OTHER = "other"


def _key_func(x: _TimeSlot) -> _GroupType:
    if isinstance(x.type, Pulse) or x.type == "delay":
        return _GroupType.PULSE_AND_DELAYS
    else:
        return _GroupType.OTHER


def _consecutive_slots_between_retargets(
    ts: list[_TimeSlot],
) -> list[tuple[list[QubitId], list[_TimeSlot]]]:
    """Filter and group _TimeSlots together.

    Group the input slots by group of consecutive Pulses and delays between two
    target operations.

    Returns:
        A list of tuples (a, b) where a is the list of common targeted qubits
        and b is a list of consecutive _TimeSlot of type Pulse or "delay". All
        "target" _TimeSlots are discarded.
    """
    grouped_slots: list = []

    for key, group in itertools.groupby(ts, _key_func):
        g = list(group)
        if key != _GroupType.PULSE_AND_DELAYS:
            continue
        grouped_slots.append((g[0].targets, g))

    return grouped_slots


def _modulate_global(ch: Channel, samples: Samples) -> Samples:
    """Modulate global samples according to the hardware specs.

    Additional parameters will probably be needed (keep_end, etc).
    """
    return Samples(
        amp=ch.modulate(samples.amp),
        det=ch.modulate(samples.det),
        phase=ch.modulate(samples.phase),
    )


def _modulate_local(
    ch: Channel, samples: list[QubitSamples]
) -> list[QubitSamples]:
    """Modulate local samples according to the hardware specs.

    Additional parameters will probably be needed (keep_end, etc).
    """
    modulated_samples: list[QubitSamples] = []
    for s in samples:
        modulated_samples.append(
            QubitSamples(
                amp=ch.modulate(s.amp),
                det=ch.modulate(s.det),
                phase=ch.modulate(s.phase),
                qubit=s.qubit,
            )
        )
    return modulated_samples
