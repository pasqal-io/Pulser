# Copyright 2022 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import pulser
import pulser_simulation
from pulser.channels.dmm import DMM
from pulser.devices import Device, MockDevice
from pulser.pulse import Pulse
from pulser.sampler import sample
from pulser.sequence._seq_drawer import draw_samples
from pulser.waveforms import BlackmanWaveform, RampWaveform

# Helpers


def assert_same_samples_as_sim(seq: pulser.Sequence) -> None:
    """Check against the legacy sample extraction in the simulation module."""
    got = sample(seq).to_nested_dict()
    want = pulser_simulation.QutipEmulator.from_sequence(seq).samples.copy()

    def truncate_samples(samples_dict):
        for key, value in samples_dict.items():
            if isinstance(value, dict):
                if value:  # Dictionary is not empty
                    samples_dict[key] = truncate_samples(value)
            else:
                samples_dict[key] = value[:-1]
        return samples_dict

    assert_nested_dict_equality(got, truncate_samples(want))


def assert_nested_dict_equality(got: dict, want: dict) -> None:
    for basis in want["Global"]:
        for qty in want["Global"][basis]:
            np.testing.assert_array_equal(
                got["Global"][basis][qty],
                want["Global"][basis][qty],
            )
    for basis in want["Local"]:
        for qubit in want["Local"][basis]:
            for qty in want["Local"][basis][qubit]:
                np.testing.assert_array_equal(
                    got["Local"][basis][qubit][qty],
                    want["Local"][basis][qubit][qty],
                )


# Tests


def test_init_error(seq_rydberg):
    var = seq_rydberg.declare_variable("var")
    seq_rydberg.delay(var, "ch0")
    assert seq_rydberg.is_parametrized()
    with pytest.raises(
        NotImplementedError, match="Parametrized sequences can't be sampled."
    ):
        sample(seq_rydberg)


@pytest.mark.parametrize("local_only", [True, False])
def test_delay_only(local_only):
    seq_ = pulser.Sequence(pulser.Register({"q0": (0, 0)}), MockDevice)
    seq_.declare_channel("ch0", "rydberg_global")
    seq_.delay(16, "ch0")
    samples = sample(seq_)
    assert samples.channel_samples["ch0"].initial_targets == {"q0"}

    qty_dict = {
        "amp": np.zeros(16),
        "det": np.zeros(16),
        "phase": np.zeros(16),
    }
    if local_only:
        expected = {
            "Local": {"ground-rydberg": {"q0": qty_dict}},
            "Global": dict(),
        }
    else:
        expected = {"Global": {"ground-rydberg": qty_dict}, "Local": dict()}

    assert_nested_dict_equality(
        samples.to_nested_dict(all_local=local_only), expected
    )


def test_one_pulse_sampling():
    """Test the sample function on a one-pulse sequence."""
    reg = pulser.Register.square(1, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    N = 1000
    amp_wf = BlackmanWaveform(N, np.pi)
    det_wf = RampWaveform(N, -np.pi / 2, np.pi / 2)
    phase = 1.234
    seq.add(Pulse(amp_wf, det_wf, phase), "ch0")
    seq.measure()

    got = sample(seq).to_nested_dict()["Global"]["ground-rydberg"]
    want = (amp_wf.samples, det_wf.samples, np.ones(N) * phase)
    for i, key in enumerate(["amp", "det", "phase"]):
        np.testing.assert_array_equal(got[key], want[i])


def test_table_sequence(seqs):
    """A table-driven test designed to be extended easily."""
    for seq in seqs:
        assert_same_samples_as_sim(seq)


def test_inXY() -> None:
    """Test sequence in XY mode."""
    pulse = Pulse(
        BlackmanWaveform(200, np.pi / 2),
        RampWaveform(200, -np.pi / 2, np.pi / 2),
        0.0,
    )
    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "mw_global")
    seq.add(pulse, "ch0")
    seq.measure(basis="XY")

    assert_same_samples_as_sim(seq)


def test_modulation(mod_seq: pulser.Sequence) -> None:
    """Test sampling for modulated channels."""
    N = mod_seq.get_duration()
    chan = mod_seq.declared_channels["ch0"]
    blackman = np.clip(np.blackman(N), 0, np.inf)
    input = (np.pi / 2) / (np.sum(blackman) / N) * blackman

    want_amp = chan.modulate(input)
    mod_samples = sample(mod_seq, modulation=True)
    got_amp = mod_samples.to_nested_dict()["Global"]["ground-rydberg"]["amp"]
    np.testing.assert_array_equal(got_amp, want_amp)

    want_det = chan.modulate(np.ones(N), keep_ends=True)
    got_det = mod_samples.to_nested_dict()["Global"]["ground-rydberg"]["det"]
    np.testing.assert_array_equal(got_det, want_det)

    want_phase = np.ones(mod_seq.get_duration(include_fall_time=True))
    got_phase = mod_samples.to_nested_dict()["Global"]["ground-rydberg"][
        "phase"
    ]
    np.testing.assert_array_equal(got_phase, want_phase)

    input_samples = sample(mod_seq)
    input_ch_samples = input_samples.channel_samples["ch0"]
    output_ch_samples = mod_samples.channel_samples["ch0"]

    for qty in ("amp", "det", "phase"):
        np.testing.assert_array_equal(
            getattr(input_ch_samples.modulate(chan), qty),
            getattr(output_ch_samples, qty),
        )


def test_modulation_local(mod_device):
    seq = pulser.Sequence(pulser.Register.square(2), mod_device)
    seq.declare_channel("ch0", "rydberg_local", initial_target=0)
    ch_obj = seq.declared_channels["ch0"]
    pulse1 = Pulse.ConstantPulse(500, 1, -1, 0)
    pulse2 = Pulse.ConstantPulse(200, 2.5, 0, 0)
    partial_fall = pulse1.fall_time(ch_obj) // 3
    seq.add(pulse1, "ch0")
    seq.delay(partial_fall, "ch0")
    seq.add(pulse2, "ch0")
    seq.target(1, "ch0")
    seq.add(pulse1, "ch0")

    input_samples = sample(seq)
    output_samples = sample(seq, modulation=True)
    assert input_samples.max_duration == seq.get_duration()
    assert output_samples.max_duration == seq.get_duration(
        include_fall_time=True
    )
    out_ch_samples = output_samples.channel_samples["ch0"]
    # The target slots account for fall time in both cases
    assert input_samples.channel_samples["ch0"].slots == out_ch_samples.slots

    # Check that the target slots account for fall time
    out_slots = out_ch_samples.slots
    # The first slot should extend to the second
    assert out_slots[0].tf == pulse1.duration + partial_fall
    assert out_slots[0].tf == out_slots[1].ti
    # The next slots should fully account for fall time
    for slot, pulse in zip(out_slots[1:], (pulse2, pulse1)):
        assert slot.tf - slot.ti == pulse.duration + pulse.fall_time(ch_obj)

    # Check that the samples are fully extracted to the nested dict
    samples_dict = output_samples.to_nested_dict()
    for qty in ("amp", "det", "phase"):
        combined = sum(
            samples_dict["Local"]["ground-rydberg"][t][qty] for t in range(2)
        )
        np.testing.assert_array_equal(getattr(out_ch_samples, qty), combined)


@pytest.mark.parametrize("disable_eom", [True, False])
def test_eom_modulation(mod_device, disable_eom):
    seq = pulser.Sequence(pulser.Register.square(2), mod_device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.enable_eom_mode("ch0", amp_on=1, detuning_on=0.0)
    seq.add_eom_pulse("ch0", 100, 0.0)
    seq.delay(200, "ch0")
    seq.add_eom_pulse("ch0", 100, 0.0)
    end_of_eom = seq.get_duration()
    if disable_eom:
        seq.disable_eom_mode("ch0")
        seq.add(Pulse.ConstantPulse(500, 1, 0, 0), "ch0")

    full_duration = seq.get_duration(include_fall_time=True)
    eom_mask = np.zeros(full_duration, dtype=bool)
    eom_mask[:end_of_eom] = True
    ext_eom_mask = np.zeros_like(eom_mask)
    eom_config = seq.declared_channels["ch0"].eom_config
    ext_eom_mask[end_of_eom : end_of_eom + 2 * eom_config.rise_time] = True

    det_off = seq._schedule["ch0"].eom_blocks[-1].detuning_off

    input_samples = sample(
        seq, extended_duration=full_duration
    ).channel_samples["ch0"]
    assert input_samples.in_eom_mode(input_samples.slots[-1]) == (
        not disable_eom
    )
    mod_samples = sample(seq, modulation=True, extended_duration=full_duration)
    chan = seq.declared_channels["ch0"]
    for qty in ("amp", "det"):
        samples = getattr(input_samples, qty)
        aom_input = samples.copy()
        aom_input[eom_mask] = det_off if qty == "det" else 0.0
        aom_output = chan.modulate(
            aom_input, eom=False, keep_ends=(qty == "det")
        )[:full_duration]

        eom_input = samples.copy()
        eom_input[ext_eom_mask] = aom_output[ext_eom_mask]
        if qty == "det":
            if not disable_eom:
                eom_input[end_of_eom:] = det_off
            eom_input = np.insert(eom_input, 0, det_off)
            eom_output = chan.modulate(eom_input, eom=True, keep_ends=True)[1:]
        else:
            eom_output = chan.modulate(eom_input, eom=True)
        eom_output = eom_output[:full_duration]

        aom_output[eom_mask + ext_eom_mask] = 0.0
        eom_output[~(eom_mask + ext_eom_mask)] = 0.0

        want = eom_output + aom_output

        # Check that modulation through sample() = sample() + modulation
        got = getattr(mod_samples.channel_samples["ch0"], qty)
        alt_got = getattr(input_samples.modulate(chan, full_duration), qty)
        np.testing.assert_array_equal(got, alt_got)

        np.testing.assert_allclose(want, got, atol=1e-10)


@pytest.fixture
def seq_with_SLM() -> pulser.Sequence:
    q_dict = {
        "batman": np.array([-4.0, 0.0]),  # sometimes masked
        "superman": np.array([4.0, 0.0]),  # always unmasked
    }

    reg = pulser.Register(q_dict)
    seq = pulser.Sequence(reg, replace(MockDevice, dmm_objects=(DMM(),)))

    seq.declare_channel("ch0", "rydberg_global")
    seq.config_slm_mask(["batman"])

    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(200, np.pi / 2), 0.0, 0.0),
        "ch0",
    )
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(200, np.pi / 2), 0.0, 0.0),
        "ch0",
    )
    seq.measure()
    return seq


def test_SLM_samples(seq_with_SLM):
    pulse = Pulse.ConstantDetuning(BlackmanWaveform(200, np.pi / 2), 0.0, 0.0)
    a_samples = pulse.amplitude.samples

    def z() -> np.ndarray:
        return np.zeros(seq_with_SLM.get_duration())

    want: dict = {
        "Global": {"ground-rydberg": {"amp": z(), "det": z(), "phase": z()}},
        "Local": {
            "ground-rydberg": {
                "superman": {"amp": z(), "det": z(), "phase": z()},
            }
        },
    }
    want["Global"]["ground-rydberg"]["amp"][200:400] = a_samples
    want["Local"]["ground-rydberg"]["superman"]["amp"][0:200] = a_samples

    got = sample(seq_with_SLM).to_nested_dict()
    assert_nested_dict_equality(got, want)


def test_SLM_against_simulation(seq_with_SLM):
    assert_same_samples_as_sim(seq_with_SLM)


def test_samples_repr(seq_rydberg):
    samples = sample(seq_rydberg)
    assert repr(samples) == "\n\n".join(
        [
            f"ch0:\n{samples.samples_list[0]!r}",
            f"ch1:\n{samples.samples_list[1]!r}",
        ]
    )


def test_extend_duration(seq_rydberg):
    samples = sample(seq_rydberg)
    short, long = samples.samples_list
    assert short.duration < long.duration
    assert short.extend_duration(short.duration).duration == short.duration
    with pytest.raises(
        ValueError, match="Can't extend samples to a lower duration."
    ):
        long.extend_duration(short.duration)

    extended_short = short.extend_duration(long.duration)
    assert extended_short.duration == long.duration
    for qty in ("amp", "det", "phase"):
        new_qty_samples = getattr(extended_short, qty)
        old_qty_samples = getattr(short, qty)
        np.testing.assert_array_equal(
            new_qty_samples[: short.duration], old_qty_samples
        )
        np.testing.assert_equal(
            new_qty_samples[short.duration :],
            old_qty_samples[-1] if qty == "phase" else 0.0,
        )
    assert extended_short.slots == short.slots


def test_phase_sampling(mod_device):
    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_global")

    dt = 100
    seq.add(Pulse.ConstantPulse(dt, 1, 0, phase=1), "ch0")
    # With 'no-delay', the jump should in between the two pulses
    seq.add(Pulse.ConstantPulse(dt, 1, 0, phase=2), "ch0", protocol="no-delay")
    # With the standard protocol, there shoud be a delay added and then
    # phase jump time is accounted for
    seq.add(Pulse.ConstantPulse(dt, 1, 0, phase=3), "ch0")
    pulse3_start = seq.get_duration() - dt
    # Detuned delay (its phase should be ignored)
    seq.add(
        Pulse.ConstantPulse(1000, 0, 1, phase=0), "ch0", protocol="no-delay"
    )
    end_of_detuned_delay = seq.get_duration()
    # phase jump time happens during the detuned delay
    seq.add(Pulse.ConstantPulse(dt, 1, 0, phase=4), "ch0")
    full_duration = seq.get_duration()
    # Nothing was added between the detuned delay and pulse4
    assert end_of_detuned_delay == full_duration - dt

    ph_jump_time = seq.declared_channels["ch0"].phase_jump_time
    assert ph_jump_time > 0
    expected_phase = np.zeros(full_duration)
    expected_phase[:dt] = 1.0
    transition2_3 = pulse3_start - ph_jump_time
    assert transition2_3 >= 2 * dt  # = End of pulse2
    expected_phase[dt:transition2_3] = 2.0
    # The detuned delay is ignored
    transition3_4 = full_duration - dt - ph_jump_time
    expected_phase[transition2_3:transition3_4] = 3.0
    expected_phase[transition3_4:] = 4.0

    got_phase = sample(seq).channel_samples["ch0"].phase
    np.testing.assert_array_equal(expected_phase, got_phase)


@pytest.mark.parametrize("modulation", [True, False])
@pytest.mark.parametrize("draw_phase_area", [True, False])
@pytest.mark.parametrize("draw_phase_shifts", [True, False])
@pytest.mark.parametrize("draw_phase_curve", [True, False])
def test_draw_samples(
    mod_seq, modulation, draw_phase_area, draw_phase_curve, draw_phase_shifts
):
    sampled_seq = sample(mod_seq, modulation=modulation)
    draw_samples(
        sampled_seq,
        draw_phase_area=draw_phase_area,
        draw_phase_shifts=draw_phase_shifts,
        draw_phase_curve=draw_phase_curve,
    )


# Fixtures


@pytest.fixture
def seqs(seq_rydberg) -> list[pulser.Sequence]:
    seqs: list[pulser.Sequence] = []

    pulse = Pulse(
        BlackmanWaveform(200, np.pi / 2),
        RampWaveform(200, -np.pi / 2, np.pi / 2),
        0.0,
    )

    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "raman_global")
    seq.add(pulse, "ch0")
    seq.measure(basis="digital")
    seqs.append(deepcopy(seq))

    seqs.append(seq_rydberg)

    return seqs


@pytest.fixture
def seq_rydberg() -> pulser.Sequence:
    reg = pulser.Register.from_coordinates(
        np.array([[0.0, 0.0], [2.0, 0.0]]), prefix="q"
    )
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "rydberg_local", initial_target="q0")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(100, np.pi / 8), 0.0, 0.0),
        "ch0",
    )
    seq.delay(20, "ch0")
    seq.add(
        Pulse.ConstantAmplitude(0.0, BlackmanWaveform(100, np.pi / 8), 0.0),
        "ch0",
    )
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(100, np.pi / 8), 0.0, 0.0),
        "ch1",
    )
    seq.target("q1", "ch1")
    seq.add(
        Pulse.ConstantAmplitude(1.0, BlackmanWaveform(100, np.pi / 8), 0.0),
        "ch1",
    )
    seq.target(["q0", "q1"], "ch1")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(100, np.pi / 8), 0.0, 0.0),
        "ch1",
    )
    seq.measure()
    return seq


@pytest.fixture
def mod_seq(mod_device: Device) -> pulser.Sequence:
    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi / 2), 1.0, 1.0),
        "ch0",
    )
    seq.measure()
    return seq
