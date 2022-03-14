"""Expose the sample() functions.

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
from pulser.sampler.samples import GlobalSamples, QubitSamples
from pulser.sequence import QubitId, Sequence, _TimeSlot


def sample(
    seq: Sequence,
    modulation: bool = False,
    local_noises: Optional[list[NoiseModel]] = None,
    global_noises: Optional[list[NoiseModel]] = None,
) -> dict:
    """Samples the given Sequence and returns a nested dictionary.

    Args:
        seq (Sequence): a pulser.Sequence instance.
        local_noise (Optional[list[LocalNoise]]): a list of the noise sources
            to account for.
        modulation (bool): a flag to account for the modulation of AOM/EOM
            before sampling.

    Returns:
        A nested dictionnary of the samples of the amplitude, detuning and
        phase at every nanoseconds for all channels.
    """
    if local_noises is None:
        local_noises = []
    if global_noises is None:
        global_noises = []
    # The noises are applied in the reversed order of the list

    d = _prepare_dict(seq, seq.get_duration())
    for ch_name in seq.declared_channels:
        addr = seq.declared_channels[ch_name].addressing
        basis = seq.declared_channels[ch_name].basis

        samples: list[QubitSamples] = []

        if addr == "Global":
            # 1. determine if decay
            # 2. extract samples
            # 3. modulate
            # 4. apply noises/SLM
            # 5. write samples

            decay = len(seq._slm_mask_targets) > 0 or len(global_noises) > 0

            ch_samples = _sample_global_channel(seq, ch_name)
            if modulation:
                ch_samples = _modulate_global(
                    seq.declared_channels[ch_name], ch_samples
                )

            if not decay:
                _write_global_samples(d, basis, ch_samples)
            else:
                qs = seq._qids - seq._slm_mask_targets
                samples = [QubitSamples.from_global(q, ch_samples) for q in qs]
                samples = noises.apply(samples, global_noises)
                _write_local_samples(d, basis, samples)

        elif addr == "Local":
            # 1. determine if modulation
            # 2. extract samples (depends on modulation)
            # 3. apply noises/SLM
            # 4. write samples
            if modulation:
                # gatering the samples of consecutive pulses with same targets
                # that can be safely modulated.
                samples = _sample_local_channel(
                    seq, ch_name, _group_between_retarget
                )
                samples = _modulate_local(
                    seq.declared_channels[ch_name], samples
                )
            else:
                samples = _sample_local_channel(seq, ch_name, _regular)
            qs = seq._qids - seq._slm_mask_targets
            samples = [s for s in samples if s.qubit in qs]
            samples = noises.apply(samples, local_noises)
            _write_local_samples(d, basis, samples)

    return d


def _write_global_samples(d: dict, basis: str, samples: GlobalSamples) -> None:
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

    Usually N is the duration of seq, but we allow for a longer one, in case of
    modulation for example.
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


def _sample_global_channel(seq: Sequence, ch_name: str) -> GlobalSamples:
    if ch_name not in seq.declared_channels:
        raise ValueError(f"{ch_name} is not declared in the given Sequence")
    slots = seq._schedule[ch_name]
    return _sample_slots(seq.get_duration(), *slots)


def _sample_local_channel(
    seq: Sequence, ch_name: str, strategy: TimeSlotExtractionStrategy
) -> list[QubitSamples]:
    if ch_name not in seq.declared_channels:
        raise ValueError(f"{ch_name} is not declared in the given Sequence")
    if seq.declared_channels[ch_name].addressing != "Local":
        raise ValueError(f"{ch_name} is no a local channel")

    return strategy(seq.get_duration(), seq._schedule[ch_name])


def _sample_slots(N: int, *slots: _TimeSlot) -> GlobalSamples:
    samples = GlobalSamples(np.zeros(N), np.zeros(N), np.zeros(N))
    for s in slots:
        if type(s.type) is str:
            continue
        pulse = cast(Pulse, s.type)
        samples.amp[s.ti : s.tf] += pulse.amplitude.samples
        samples.det[s.ti : s.tf] += pulse.detuning.samples
        samples.phase[s.ti : s.tf] += pulse.phase

    return samples


# This strategy type is used mostly for the necessity to extract samples
# differently when taking into account the modulation of AOM/EOM. Still it's
# nice to keep modularity here, to accomodate for future needs.
TimeSlotExtractionStrategy = Callable[
    [int, List[_TimeSlot]], List[QubitSamples]
]


def _regular(N: int, ts: list[_TimeSlot]) -> list[QubitSamples]:
    return [
        QubitSamples.from_global(q, _sample_slots(N, slot))
        for slot in ts
        for q in slot.targets
    ]


def _group_between_retarget(N: int, ts: list[_TimeSlot]) -> list[QubitSamples]:
    qs: list[QubitSamples] = []
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

    def key_func(x: _TimeSlot) -> _GroupType:
        if isinstance(x.type, Pulse) or x.type == "delay":
            return _GroupType.PULSE_AND_DELAYS
        else:
            return _GroupType.OTHER

    for key, group in itertools.groupby(ts, key_func):
        g = list(group)
        if key != _GroupType.PULSE_AND_DELAYS:
            continue
        grouped_slots.append((g[0].targets, g))

    return grouped_slots


def _modulate_global(ch: Channel, samples: GlobalSamples) -> GlobalSamples:
    """Modulate global samples according to the hardware specs.

    Additional parameters will probably be needed (keep_end, etc).
    """
    return GlobalSamples(
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
                phase=s.phase * np.ones(len(ch.modulate(s.amp))),
                qubit=s.qubit,
            )
        )
    return modulated_samples
