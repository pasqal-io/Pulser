# Copyright 2026 Pulser Development Team
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
"""Detuning noise must not switch off when a Rydberg drive is idle."""

from dataclasses import replace

import numpy as np
import pytest

from pulser import (
    ConstantWaveform,
    MockDevice,
    NoiseModel,
    Pulse,
    Register,
    Sequence,
)
from pulser._hamiltonian_data import HamiltonianData


@pytest.mark.parametrize("with_modulation", [False, True])
@pytest.mark.parametrize("noise", ["doppler", "dc", "hf", "combined"])
def test_rydberg_detuning_noise_spans_delays(noise, with_modulation):
    """Noise covers delays and the modulation fall time."""
    channel = replace(MockDevice.channels["rydberg_global"], mod_bandwidth=4)
    device = replace(
        MockDevice, channel_ids=("ryd",), channel_objects=(channel,)
    )
    seq = Sequence(Register({"q0": (0, 0)}), device)
    seq.declare_channel("ryd", "ryd")
    seq.delay(100, "ryd")
    seq.add(Pulse.ConstantPulse(200, 1, 2, 0), "ryd")
    seq.delay(400, "ryd")
    seq.add(Pulse.ConstantPulse(200, 1, 2, 0), "ryd")
    seq.delay(10, "ryd")
    params = {}
    if noise in ("doppler", "combined"):
        params["temperature"] = 20
    if noise in ("dc", "combined"):
        params["detuning_sigma"] = 0.5
    if noise in ("hf", "combined"):
        params.update(detuning_hf_psd=(0.1, 0.3), detuning_hf_omegas=(1, 3))
    np.random.seed(127)
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(**params),
        n_trajectories=3,
        with_modulation=with_modulation,
    )
    noiseless = ham.samples.to_nested_dict(all_local=True)["Local"][
        "ground-rydberg"
    ]["q0"]
    duration = seq.get_duration(include_fall_time=with_modulation)
    assert len(noiseless["det"]) == duration
    for samples in ham.noisy_samples:
        traj = samples.trajectory
        actual = samples.samples.to_nested_dict()["Local"]["ground-rydberg"][
            "q0"
        ]
        expected = np.zeros(duration)
        if noise in ("doppler", "combined"):
            expected += traj.doppler_detune["q0"]
        if noise in ("dc", "combined"):
            expected += traj.det_fluctuations["ryd"]
        if noise in ("hf", "combined"):
            # Two PSD points give one cosine: sqrt(2 * d_omega * PSD).
            expected += np.sqrt(2 * 2 * 0.3) * np.cos(
                3 * np.arange(duration) * 1e-3 + traj.det_phases["ryd"][0]
            )
        np.testing.assert_allclose(
            actual["det"] - noiseless["det"], expected, atol=1e-14
        )
        np.testing.assert_array_equal(actual["amp"], noiseless["amp"])


def test_rydberg_delay_matches_zero_amplitude_pulse():
    """A delay and a zero drive have the same noisy detuning."""
    outputs = []
    for zero_pulse in (False, True):
        seq = Sequence(Register({"q0": (0, 0)}), MockDevice)
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
        if zero_pulse:
            seq.add(Pulse.ConstantPulse(1000, 0, 0, 0), "ryd")
        else:
            seq.delay(1000, "ryd")
        seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
        np.random.seed(23)
        ham = HamiltonianData.from_sequence(
            seq,
            noise_model=NoiseModel(temperature=20, detuning_sigma=0.5),
            n_trajectories=1,
        )
        outputs.append(
            next(ham.noisy_samples).samples.to_nested_dict()["Local"][
                "ground-rydberg"
            ]["q0"]
        )
    for quantity in ("amp", "det", "phase"):
        np.testing.assert_array_equal(
            outputs[0][quantity], outputs[1][quantity]
        )


def test_detuning_noise_covers_alignment_delays_once_per_channel():
    """Alignment does not gate or duplicate channel-frequency noise."""
    seq = Sequence(Register({"q0": (0, 0)}), MockDevice)
    seq.declare_channel("first", "rydberg_global")
    seq.declare_channel("second", "rydberg_global")
    seq.add(Pulse.ConstantPulse(600, 1, 0, 0), "first")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "second", protocol="no-delay")
    seq.align("first", "second")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "second")
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(temperature=20, detuning_sigma=0.5),
        n_trajectories=1,
    )
    traj = ham.noise_trajectories[0].trajectory
    det = next(ham.noisy_samples).samples.to_nested_dict()["Local"][
        "ground-rydberg"
    ]["q0"]["det"]
    expected = 2 * traj.doppler_detune["q0"] + sum(
        traj.det_fluctuations.values()
    )
    np.testing.assert_allclose(det, expected)


def test_full_duration_detuning_noise_keeps_bad_atoms_dark():
    """State preparation still zeros all coefficients of an absent atom."""
    seq = Sequence(Register({"q0": (0, 0), "q1": (10, 0)}), MockDevice)
    seq.declare_channel("ryd", "rydberg_global")
    seq.delay(100, "ryd")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(temperature=20, detuning_sigma=0.5),
        n_trajectories=1,
    )
    traj = replace(
        ham.noise_trajectories[0].trajectory,
        bad_atoms={"q0": True, "q1": False},
    )
    local = ham._sample_with_trajectory(traj).to_nested_dict()["Local"][
        "ground-rydberg"
    ]
    for quantity in ("amp", "det", "phase"):
        np.testing.assert_array_equal(local["q0"][quantity], 0)
    np.testing.assert_allclose(
        local["q1"]["det"],
        traj.doppler_detune["q1"] + traj.det_fluctuations["ryd"],
    )


@pytest.mark.parametrize("with_drive", [False, True])
def test_dmm_does_not_add_laser_noise_during_delays(with_drive):
    """A long DMM slot adds no Doppler or laser-frequency noise."""
    reg = Register({"q0": (0, 0), "q1": (10, 0)})
    seq = Sequence(reg, MockDevice)
    if with_drive:
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
    seq.config_detuning_map(
        reg.define_detuning_map({"q0": 1, "q1": 0}), "dmm_0"
    )
    seq.add_dmm_detuning(
        ConstantWaveform(400, -4), "dmm_0", protocol="no-delay"
    )
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(
            temperature=20, detuning_sigma=0.5, dmm_sigma=0.2
        ),
        n_trajectories=1,
    )
    traj = ham.noise_trajectories[0].trajectory
    local = next(ham.noisy_samples).samples.to_nested_dict()["Local"][
        "ground-rydberg"
    ]
    for qid, weight in [("q0", 1), ("q1", 0)]:
        expected = -4 * weight * traj.dmm_det_fluctuation["dmm_0"]
        if with_drive:
            expected += traj.doppler_detune[qid] + traj.det_fluctuations["ryd"]
        np.testing.assert_allclose(local[qid]["det"], expected)


def test_raman_detuning_noise_keeps_its_slot_scope():
    """This ground-Rydberg fix does not extend Raman noise into delays."""
    seq = Sequence(Register({"q0": (0, 0)}), MockDevice)
    seq.declare_channel("raman", "raman_local", initial_target="q0")
    seq.delay(100, "raman")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "raman")
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(temperature=20, detuning_sigma=0.5),
        n_trajectories=1,
    )
    traj = ham.noise_trajectories[0].trajectory
    det = next(ham.noisy_samples).samples.to_nested_dict()["Local"]["digital"][
        "q0"
    ]["det"]
    np.testing.assert_array_equal(det[:100], 0)
    np.testing.assert_allclose(
        det[100:], traj.doppler_detune["q0"] + traj.det_fluctuations["raman"]
    )


def test_local_rydberg_noise_covers_retargeting_and_untargeted_atoms():
    """Frequency noise persists for every atom during retargeting."""
    seq = Sequence(
        Register({"q0": (0, 0), "q1": (10, 0), "q2": (20, 0)}), MockDevice
    )
    seq.declare_channel("ryd", "rydberg_local", initial_target="q0")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
    seq.delay(100, "ryd")
    seq.target("q1", "ryd")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(temperature=20, detuning_sigma=0.5),
        n_trajectories=1,
    )
    traj = ham.noise_trajectories[0].trajectory
    local = next(ham.noisy_samples).samples.to_nested_dict()["Local"][
        "ground-rydberg"
    ]
    assert set(local) == {"q0", "q1", "q2"}
    for qid in local:
        np.testing.assert_allclose(
            local[qid]["det"],
            traj.doppler_detune[qid] + traj.det_fluctuations["ryd"],
        )
    np.testing.assert_array_equal(local["q2"]["amp"], 0)


def test_amplitude_noise_leaves_delays_without_a_drive():
    """Frequency noise does not create a drive during a delay."""
    seq = Sequence(Register({"q0": (0, 0)}), MockDevice)
    seq.declare_channel("ryd", "rydberg_global")
    seq.delay(100, "ryd")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd")
    seq.delay(100, "ryd")
    ham = HamiltonianData.from_sequence(
        seq,
        noise_model=NoiseModel(amp_sigma=0.2, temperature=20),
        n_trajectories=1,
    )
    traj = ham.noise_trajectories[0].trajectory
    amp = next(ham.noisy_samples).samples.to_nested_dict()["Local"][
        "ground-rydberg"
    ]["q0"]["amp"]
    expected = np.zeros(300)
    expected[100:200] = traj.amp_fluctuations["ryd"]
    np.testing.assert_array_equal(amp, expected)
