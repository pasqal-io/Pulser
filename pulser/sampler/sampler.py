"""Exposes the sample() functions.

It contains many helpers.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Callable, List, Optional, cast

import numpy as np

import pulser.simulation.noises as noises
from pulser.channels import Channel
from pulser.pulse import Pulse
from pulser.sampler.samples import QubitSamples
from pulser.sequence import Sequence, _TimeSlot


def sample(
    seq: Sequence,
    modulation: bool = False,
    common_noises: Optional[list[noises.NoiseModel]] = None,
    global_noises: Optional[list[noises.NoiseModel]] = None,
) -> dict:
    """Samples the given Sequence and returns a nested dictionary.

    It is intended to be used like the json.dumps() function.

    Args:
        seq (Sequence): A pulser.Sequence instance.
        modulation (bool): Flag to account for the modulation of AOM/EOM
            before sampling.
        common_noises (Optional[list[LocalNoise]]): A list of the noise sources
            for all channels.
        global_noises (Optional[list[LocalNoise]]): A list of the noise sources
            for global channels.

    Returns:
        A nested dictionnary of the samples of the amplitude, detuning and
        phase at every nanoseconds for all channels.
    """
    if common_noises is None:
        common_noises = []
    if global_noises is None:
        global_noises = []

    # 1. determine if the global channel decay to a local one
    # 2. extract samples
    # 3. modulate
    # 4. apply noises/SLM
    # 5. write samples

    samples: dict[str, list[QubitSamples]] = {}
    addrs: dict[str, str] = {}

    for ch_name, ch in seq.declared_channels.items():
        s: list[QubitSamples]

        addr = seq.declared_channels[ch_name].addressing

        ch_noises = list(common_noises)

        if addr == "Global":
            decay = (
                len(seq._slm_mask_targets) > 0
                or len(global_noises) > 0
                or len(common_noises) > 0
            )
            if decay:
                addr = "Decayed"
                ch_noises.extend(global_noises)

        addrs[ch_name] = addr

        strategy = _group_between_retargets if modulation else _regular
        s = _sample_channel(seq, ch_name, strategy)
        if modulation:
            s = _modulate(ch, s)

        s = noises.apply(s, ch_noises)

        unmasked_qubits = seq._qids - seq._slm_mask_targets
        s = [x for x in s if x.qubit in unmasked_qubits]  # SLM

        samples[ch_name] = s

    # format the samples in the simulation dict form
    d = _write_dict(seq, samples, addrs)

    return d


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
            "Global": {"XY": new_qty_dict()},
            "Local": {"XY": new_qdict()},
        }
    else:
        return {
            "Global": defaultdict(new_qty_dict),
            "Local": defaultdict(new_qdict),
        }


def _write_dict(
    seq: Sequence,
    samples: dict[str, list[QubitSamples]],
    addrs: dict[str, str],
) -> dict:
    """Export the given samples to a nested dictionary."""
    # Get the duration
    if not _same_duration(samples):  # pragma: no cover
        raise ValueError("All the samples do not share the same duration.")
    N = list(samples.values())[0][0].amp.size

    d = _prepare_dict(seq, N)

    for ch_name, some_samples in samples.items():
        basis = seq.declared_channels[ch_name].basis
        addr = addrs[ch_name]
        if addr == "Global":
            # Take samples on only one qubit and write them
            a_qubit = next(iter(seq._qids))
            to_write = [x for x in some_samples if x.qubit == a_qubit]
            for s in to_write:
                d["Global"][basis]["amp"] += s.amp
                d["Global"][basis]["det"] += s.det
                d["Global"][basis]["phase"] += s.phase
        else:
            for s in some_samples:
                d["Local"][basis][s.qubit]["amp"] += s.amp
                d["Local"][basis][s.qubit]["det"] += s.det
                d["Local"][basis][s.qubit]["phase"] += s.phase
    return d


def _same_duration(samples: dict[str, list[QubitSamples]]) -> bool:
    durations: list[int] = []
    flatten_samples: list[QubitSamples] = []
    for some_samples in samples.values():
        flatten_samples.extend(some_samples)
    for s in flatten_samples:
        durations.extend((s.amp.size, s.det.size, s.phase.size))
    return durations.count(durations[0]) == len(durations)


def _sample_channel(
    seq: Sequence, ch_name: str, strategy: TimeSlotExtractionStrategy
) -> list[QubitSamples]:
    """Compute a list of QubitSamples for a channel."""
    qs: list[QubitSamples] = []
    grouped_slots = strategy(seq._schedule[ch_name])

    for group in grouped_slots:
        ss = _sample_slots(seq.get_duration(), *group)
        qs.extend(ss)
    return qs


def _sample_slots(N: int, *slots: _TimeSlot) -> list[QubitSamples]:
    """Gather samples of a list of _TimeSlot in a single Samples instance."""
    # Same target in one group, guaranteed by the strategy (this seems
    # weird, it's not enforced by the structure,bad design?)
    qubits = slots[0].targets
    amp, det, phase = np.zeros(N), np.zeros(N), np.zeros(N)
    for s in slots:
        if type(s.type) is str:  # pragma: no cover
            continue
        pulse = cast(Pulse, s.type)
        amp[s.ti : s.tf] += pulse.amplitude.samples
        det[s.ti : s.tf] += pulse.detuning.samples
        phase[s.ti : s.tf] += pulse.phase
    qs = [
        QubitSamples(
            amp=amp.copy(), det=det.copy(), phase=phase.copy(), qubit=q
        )
        for q in qubits
    ]
    return qs


TimeSlotExtractionStrategy = Callable[[List[_TimeSlot]], List[List[_TimeSlot]]]
"""Extraction strategy of _TimeSlot's of a Channel.

It's an alias for functions that returns a list of lists of _TimeSlots.
_TimeSlots in the same group MUST share the same targets.

NOTE:
    This strategy type is used mostly for the necessity to extract samples
    differently when taking into account the modulation of AOM/EOM. Despite
    there are only two cases, whether it's necessary to modulate a local
    channel or not, this pattern can accomodate for future needs.
"""


def _regular(ts: list[_TimeSlot]) -> list[list[_TimeSlot]]:
    """No grouping performed, return only the pulses."""
    return [[x] for x in ts if isinstance(x.type, Pulse)]


def _group_between_retargets(
    ts: list[_TimeSlot],
) -> list[list[_TimeSlot]]:
    """Filter and group _TimeSlots together.

    Group the input slots by groups of successive Pulses and delays between
    two target operations. Consider the following sequence consisting of pulses
    A B C D E F, targeting different qubits:

    .---A---B------.---C--D--E---.----F--
    ^              ^             ^
    |              |             |
    target q0   target q1     target q0

    It will group the pulses' _TimeSlot's in batches (A B), (C D E) and (F),
    returning the following list of list of _TimeSlot instances:

    [[A, B], [C, D, E], [F]]

    Args:
        ts (list[_TimeSlot]): A list of TimeSlot from a Sequence schedule.

    Returns:
        A list of list of _TimeSlot. _TimeSlot instances are successive and
        share the same targets. They are of type either Pulse or "delay", all
        "target" ones are discarded.
    """
    TO_KEEP = "pulses_and_delays"

    def key_func(x: _TimeSlot) -> str:
        if isinstance(x.type, Pulse) or x.type == "delay":
            return TO_KEEP
        else:
            return "other"

    grouped_slots: list[list[_TimeSlot]] = []

    for key, group in itertools.groupby(ts, key_func):
        g = list(group)
        if key != TO_KEEP:
            continue
        grouped_slots.append(g)

    return grouped_slots


def _modulate(ch: Channel, samples: list[QubitSamples]) -> list[QubitSamples]:
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
