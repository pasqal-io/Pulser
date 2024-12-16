# Copyright 2020 Pulser Development Team
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

import contextlib
import dataclasses
import itertools
import json
import re
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

import pulser
from pulser import Pulse, Register, Register3D, Sequence
from pulser.channels import Raman, Rydberg
from pulser.channels.dmm import DMM
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices import AnalogDevice, DigitalAnalogDevice, MockDevice
from pulser.devices._device_datacls import Device, VirtualDevice
from pulser.register.base_register import BaseRegister
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import TriangularLatticeLayout
from pulser.sampler import sample
from pulser.sequence.sequence import _TimeSlot
from pulser.waveforms import (
    BlackmanWaveform,
    CompositeWaveform,
    ConstantWaveform,
    InterpolatedWaveform,
    RampWaveform,
)


@pytest.fixture
def reg():
    layout = TriangularLatticeLayout(100, spacing=5)
    return layout.rectangular_register(4, 7, prefix="q")


@pytest.fixture
def det_map(reg: Register):
    return reg.define_detuning_map(
        {"q" + str(i): (1.0 if i in [0, 1, 3, 4] else 0) for i in range(10)}
    )


@pytest.fixture
def device():
    return dataclasses.replace(
        DigitalAnalogDevice,
        dmm_objects=(
            DMM(bottom_detuning=-70, total_bottom_detuning=-700),
            DMM(bottom_detuning=-100, total_bottom_detuning=-1000),
        ),
    )


def test_init(reg, device):
    with pytest.raises(TypeError, match="must be of type 'BaseDevice'"):
        Sequence(reg, Device)

    seq = Sequence(reg, device)
    assert Register(seq.qubit_info) == reg
    assert seq.declared_channels == {}
    assert (
        seq.available_channels.keys()
        == {**device.channels, **device.dmm_channels}.keys()
    )


def test_channel_declaration(reg, device):
    seq = Sequence(reg, device)
    available_channels = set(seq.available_channels)
    assert seq.get_addressed_bases() == ()
    assert seq.get_addressed_states() == []
    with pytest.raises(ValueError, match="Name starting by 'dmm_'"):
        seq.declare_channel("dmm_1_2", "raman")
    seq.declare_channel("ch0", "rydberg_global")
    assert seq.get_addressed_bases() == ("ground-rydberg",)
    assert seq.get_addressed_states() == ["r", "g"]
    seq.declare_channel("ch1", "raman_local")
    assert seq.get_addressed_bases() == ("ground-rydberg", "digital")
    assert seq.get_addressed_states() == ["r", "g", "h"]
    with pytest.raises(ValueError, match="No channel"):
        seq.declare_channel("ch2", "raman")
    with pytest.raises(ValueError, match="not available"):
        seq.declare_channel("ch2", "rydberg_global")
    with pytest.raises(ValueError, match="name is already in use"):
        seq.declare_channel("ch0", "raman_local")

    chs = {"rydberg_global", "raman_local"}
    assert seq._schedule["ch0"][-1] == _TimeSlot(
        "target", -1, 0, set(seq.qubit_info.keys())
    )
    assert set(seq.available_channels) == available_channels - chs

    seq2 = Sequence(reg, MockDevice)
    available_channels = set(seq2.available_channels)
    channel_map = {
        "ch0": "raman_local",
        "ch1": "rydberg_global",
        "ch2": "rydberg_global",
    }
    for channel, channel_id in channel_map.items():
        seq2.declare_channel(channel, channel_id)
    assert set(seq2.available_channels) == (available_channels - {"mw_global"})
    assert set(
        seq2._schedule[channel].channel_id
        for channel in seq2.declared_channels
    ) == set(channel_map.values())
    with pytest.raises(ValueError, match="type 'Microwave' cannot work "):
        seq2.declare_channel("ch3", "mw_global")

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "mw_global")
    assert set(seq2.available_channels) == {"mw_global", "dmm_0"}
    with pytest.raises(
        ValueError,
        match="cannot work simultaneously with the declared 'Microwave'",
    ):
        seq2.declare_channel("ch3", "rydberg_global")
    assert seq2.get_addressed_bases() == ("XY",)
    assert seq2.get_addressed_states() == ["u", "d"]


def test_dmm_declaration(reg, device, det_map):
    seq = Sequence(reg, device)
    available_channels = set(seq.available_channels)
    assert seq.get_addressed_bases() == ()
    seq.config_detuning_map(det_map, "dmm_0")
    assert seq.get_addressed_bases() == ("ground-rydberg",)
    seq.config_detuning_map(det_map, "dmm_1")
    with pytest.raises(ValueError, match="No DMM dmm_2"):
        seq.config_detuning_map(det_map, "dmm_2")
    with pytest.raises(ValueError, match="DMM dmm_0 is not available"):
        seq.config_detuning_map(det_map, "dmm_0")

    chs = {"dmm_0", "dmm_1"}
    assert seq._schedule["dmm_0"][-1] == _TimeSlot(
        "target", -1, 0, set(seq.qubit_info.keys())
    )
    assert set(seq.available_channels) == available_channels - chs

    seq2 = Sequence(reg, MockDevice)
    available_channels = set(seq2.available_channels)
    channel_map = {
        "dmm_0": "dmm_0",
        "dmm_0_1": "dmm_0",
    }
    seq2.config_detuning_map(det_map, "dmm_0")
    # If a DMM was declared but not as an SLM Mask,
    # MW channels are not available
    assert set(seq2.available_channels) == (available_channels - {"mw_global"})
    seq2.config_detuning_map(det_map, "dmm_0")
    assert set(seq2.available_channels) == (available_channels - {"mw_global"})
    assert channel_map.keys() == seq2.declared_channels.keys()
    assert set(
        seq2._schedule[channel].channel_id
        for channel in seq2.declared_channels
    ) == set(channel_map.values())
    with pytest.raises(ValueError, match="type 'Microwave' cannot work "):
        seq2.declare_channel("mw_ch", "mw_global")

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "mw_global")
    # DMM channels are still available,
    # but can only be declared using an SLM Mask
    assert set(seq2.available_channels) == {"mw_global", "dmm_0"}
    with pytest.raises(
        ValueError,
        match="cannot work simultaneously with the declared 'Microwave'",
    ):
        seq2.config_detuning_map(det_map, "dmm_0")


def test_slm_declaration(reg, device, det_map):
    # Definining an SLM on a Device
    seq = Sequence(reg, device)
    available_channels = set(seq.available_channels)
    assert seq.get_addressed_bases() == ()
    with pytest.raises(ValueError, match="No DMM dmm_2 in the device"):
        seq.config_slm_mask(["q0", "q1", "q3", "q4"], "dmm_2")
    seq.config_slm_mask(["q0", "q1", "q3", "q4"])
    assert seq.get_addressed_bases() == tuple()
    with pytest.raises(
        ValueError, match="SLM mask can be configured only once."
    ):
        seq.config_slm_mask(["q0", "q1", "q3", "q4"], "dmm_1")
    # no channel has been declared
    assert len(seq._schedule) == 0
    # dmm_0 no longer appears in available channels
    assert set(seq.available_channels) == available_channels - {"dmm_0"}

    # Configuring a DMM after having configured a SLM with the same DMM
    seq2 = Sequence(reg, MockDevice)
    available_channels = set(seq2.available_channels)
    channel_map = {
        "dmm_0": "dmm_0",
        "dmm_0_1": "dmm_0",
    }
    seq2.config_slm_mask(["q0", "q1", "q3", "q4"])
    assert set(seq2.declared_channels.keys()) == set()
    # If a DMM was declared as an SLM Mask, MW channels are still available
    assert set(seq2.available_channels) == available_channels
    # If other DMM are configured, the MW channel is no longer available
    seq2.config_detuning_map(det_map, "dmm_0")
    assert seq2._slm_mask_dmm == "dmm_0"
    assert set(seq2.available_channels) == (available_channels - {"mw_global"})
    assert channel_map.keys() == seq2.declared_channels.keys()
    assert set(
        seq2._schedule[channel].channel_id
        for channel in seq2.declared_channels
    ) == set(channel_map.values())
    with pytest.raises(ValueError, match="type 'Microwave' cannot work "):
        seq2.declare_channel("mw_ch", "mw_global")

    # Configuring an SLM after having configured a DMM with the same DMM
    seq2 = Sequence(reg, MockDevice)
    seq2.config_detuning_map(det_map, "dmm_0")
    seq2.config_slm_mask(["q0", "q1", "q3", "q4"])
    # Name of DMM implementing SLM has a suffix
    assert seq2._slm_mask_dmm == "dmm_0_1"

    # Configuring a SLM after having declared a microwave channel
    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "mw_global")
    # DMM channels are still available, but can be configured using an SLM Mask
    assert set(seq2.available_channels) == {"mw_global", "dmm_0"}
    assert set(seq2.declared_channels.keys()) == {"ch0"}
    seq2.config_slm_mask(["q0", "q1", "q3", "q4"], "dmm_0")
    assert set(seq2.available_channels) == {"mw_global"}
    assert set(seq2.declared_channels.keys()) == {"ch0"}

    # Declaring a microwave channel after having configured an SLM
    seq2 = Sequence(reg, MockDevice)
    available_channels = set(seq2.available_channels)
    seq2.config_slm_mask(["q0", "q1", "q3", "q4"], "dmm_0")
    # If a DMM was declared as an SLM Mask, all channels are still available
    assert set(seq2.available_channels) == available_channels
    assert set(seq2.declared_channels.keys()) == set()
    # If MW channel is defined, only mw channels are available
    seq2.declare_channel("ch0", "mw_global")
    assert set(seq2.available_channels) == {"mw_global"}
    # DMM is not shown as declared
    assert set(seq2.declared_channels.keys()) == {"ch0"}


def test_magnetic_field(reg):
    seq = Sequence(reg, MockDevice)
    with pytest.raises(
        AttributeError,
        match="only defined when the sequence " "is in 'XY Mode'.",
    ):
        seq.magnetic_field
    seq.declare_channel("ch0", "mw_global")  # seq in XY mode
    # mag field is the default
    assert np.all(seq.magnetic_field == np.array((0.0, 0.0, 30.0)))
    seq.set_magnetic_field(bx=1.0, by=-1.0, bz=0.5)
    assert np.all(seq.magnetic_field == np.array((1.0, -1.0, 0.5)))
    with pytest.raises(ValueError, match="magnitude greater than 0"):
        seq.set_magnetic_field(bz=0.0)
    assert seq._empty_sequence
    seq.add(Pulse.ConstantPulse(100, 1, 1, 0), "ch0")
    assert not seq._empty_sequence
    with pytest.raises(ValueError, match="can only be set on an empty seq"):
        seq.set_magnetic_field(1.0, 0.0, 0.0)

    # Raises an error if a Global channel is declared (not in xy)
    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "rydberg_global")
    with pytest.raises(ValueError, match="can only be set in 'XY Mode'."):
        seq2.set_magnetic_field(1.0, 0.0, 0.0)

    # Same if a dmm channel was configured
    seq2 = Sequence(reg, MockDevice)
    seq2.config_detuning_map(det_map, "dmm_0")  # not in XY mode
    with pytest.raises(ValueError, match="can only be set in 'XY Mode'."):
        seq2.set_magnetic_field(1.0, 0.0, 0.0)

    # Works if a slm mask was configured
    seq3 = Sequence(reg, MockDevice)
    seq3.config_slm_mask(["q0", "q1"], "dmm_0")
    seq3.set_magnetic_field(1.0, 0.0, 0.0)  # sets seq to XY mode
    # dmm_0 doesn't appear because there can only be one in XY mode
    # and the SLM is already configured
    assert set(seq3.available_channels) == {"mw_global"}
    assert list(seq3.declared_channels.keys()) == []
    seq3.declare_channel("ch0", "mw_global")
    assert list(seq3.declared_channels.keys()) == ["ch0"]

    seq3 = Sequence(reg, MockDevice)
    seq3.set_magnetic_field(1.0, 0.0, 0.0)  # sets seq to XY mode
    assert set(seq3.available_channels) == {"mw_global", "dmm_0"}
    seq3.declare_channel("ch0", "mw_global")
    # Does not change to default
    assert np.all(seq3.magnetic_field == np.array((1.0, 0.0, 0.0)))
    var = seq3.declare_variable("var")
    # Sequence is marked as non-empty when parametrized too
    seq3.add(Pulse.ConstantPulse(100, var, 1, 0), "ch0")
    assert seq3.is_parametrized()
    with pytest.raises(ValueError, match="can only be set on an empty seq"):
        seq3.set_magnetic_field()

    seq3_str = seq3._serialize()
    seq3_ = Sequence._deserialize(seq3_str)
    assert seq3_._in_xy
    assert str(seq3) == str(seq3_)
    assert np.all(seq3_.magnetic_field == np.array((1.0, 0.0, 0.0)))


@pytest.fixture
def devices():
    device1 = Device(
        name="test_device1",
        dimensions=2,
        rydberg_level=70,
        max_atom_num=100,
        max_radial_distance=60,
        min_atom_distance=5,
        supports_slm_mask=True,
        channel_objects=(
            Raman.Global(
                2 * np.pi * 20,
                2 * np.pi * 10,
                max_duration=2**26,
            ),
            Raman.Local(
                2 * np.pi * 20,
                2 * np.pi * 10,
                clock_period=1,
                max_duration=2**26,
                max_targets=3,
                mod_bandwidth=4,
            ),
            Rydberg.Global(
                max_abs_detuning=2 * np.pi * 4,
                max_amp=2 * np.pi * 3,
                clock_period=4,
                max_duration=2**26,
            ),
        ),
        dmm_objects=(
            DMM(
                clock_period=4,
                min_duration=16,
                max_duration=2**26,
                # Better than DMM of DigitalAnalogDevice
                bottom_detuning=-2 * np.pi * 40,
                total_bottom_detuning=-2 * np.pi * 4000,
            ),
        ),
    )

    device2 = Device(
        name="test_device2",
        dimensions=2,
        rydberg_level=70,
        max_atom_num=100,
        max_radial_distance=60,
        min_atom_distance=5,
        supports_slm_mask=True,
        channel_ids=("rmn_local", "rydberg_global"),
        channel_objects=(
            Raman.Local(
                2 * np.pi * 20,
                2 * np.pi * 10,
                clock_period=3,
                max_duration=2**26,
                max_targets=5,
                mod_bandwidth=2,
                fixed_retarget_t=2,
            ),
            Rydberg.Global(
                max_abs_detuning=2 * np.pi * 4,
                max_amp=2 * np.pi * 3,
                clock_period=2,
                max_duration=2**26,
            ),
        ),
        dmm_objects=(
            DMM(
                clock_period=4,
                min_duration=16,
                max_duration=2**26,
                bottom_detuning=-2 * np.pi * 20,
                total_bottom_detuning=-2 * np.pi * 2000,
            ),
        ),
    )

    device3 = VirtualDevice(
        name="test_device3",
        dimensions=2,
        rydberg_level=70,
        min_atom_distance=5,
        supports_slm_mask=True,
        channel_ids=(
            "rmn_local1",
            "rmn_local2",
            "rmn_local3",
            "rydberg_global",
        ),
        channel_objects=(
            Raman.Local(
                max_abs_detuning=2 * np.pi * 20,
                max_amp=2 * np.pi * 10,
                min_retarget_interval=220,
                fixed_retarget_t=1,
                max_targets=1,
                mod_bandwidth=2,
                clock_period=3,
                min_duration=16,
                max_duration=2**26,
            ),
            Raman.Local(
                2 * np.pi * 20,
                2 * np.pi * 10,
                clock_period=3,
                max_duration=2**26,
                mod_bandwidth=2,
                fixed_retarget_t=2,
            ),
            Raman.Local(
                0,
                2 * np.pi * 10,
                clock_period=4,
                max_duration=2**26,
            ),
            Rydberg.Global(
                max_abs_detuning=2 * np.pi * 4,
                max_amp=2 * np.pi * 3,
                clock_period=4,
                max_duration=2**26,
            ),
        ),
        dmm_objects=(
            DMM(
                clock_period=4,
                min_duration=16,
                max_duration=2**26,
                bottom_detuning=-2 * np.pi * 20,
                total_bottom_detuning=-2 * np.pi * 2000,
            ),
        ),
    )

    return [device1, device2, device3]


@pytest.fixture
def pulses():
    rise = Pulse.ConstantDetuning(
        RampWaveform(252, 0.0, 2.3 * 2 * np.pi),
        -4 * np.pi,
        0.0,
    )
    sweep = Pulse.ConstantAmplitude(
        2.3 * 2 * np.pi,
        RampWaveform(400, -4 * np.pi, 4 * np.pi),
        1.0,
    )
    fall = Pulse.ConstantDetuning(
        RampWaveform(500, 2.3 * 2 * np.pi, 0.0),
        4 * np.pi,
        0.0,
    )
    return [rise, sweep, fall]


def init_seq(
    reg,
    device,
    channel_name,
    channel_id,
    l_pulses,
    initial_target=None,
    parametrized=False,
    mappable_reg=False,
    config_det_map=False,
    prefer_slm_mask=True,
) -> Sequence:
    register = (
        reg.layout.make_mappable_register(len(reg.qubits))
        if mappable_reg
        else reg
    )
    seq = Sequence(register, device)
    seq.declare_channel(
        channel_name, channel_id, initial_target=initial_target
    )
    if l_pulses is not None:
        for pulse in l_pulses:
            seq.add(pulse, channel_name)
    if parametrized:
        delay = seq.declare_variable("delay", dtype=int)
        seq.delay(delay, channel_name)
    if config_det_map:
        det_map = reg.define_detuning_map(
            {
                "q" + str(i): (1.0 if i in [0, 1, 3, 4] else 0)
                for i in range(10)
            }
        )
        if mappable_reg or not prefer_slm_mask:
            seq.config_detuning_map(detuning_map=det_map, dmm_id="dmm_0")
        else:
            seq.config_slm_mask(["q0"], "dmm_0")
    return seq


def test_ising_mode(
    reg,
    device,
):
    seq = Sequence(reg, device)
    assert not seq._in_ising and not seq._in_xy
    seq.declare_channel("ch0", "rydberg_global")
    assert seq._in_ising and not seq._in_xy
    with pytest.raises(TypeError, match="_in_ising must be a bool."):
        seq._in_ising = 1
    with pytest.raises(ValueError, match="Cannot quit ising."):
        seq._in_ising = False

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "mw_global")
    assert seq2._in_xy and not seq2._in_ising
    with pytest.raises(ValueError, match="Cannot be in ising if in xy."):
        seq2._in_ising = True


@pytest.mark.parametrize("config_det_map", [False, True])
@pytest.mark.parametrize("starts_mappable", [False, True])
@pytest.mark.parametrize("mappable_reg", [False, True])
@pytest.mark.parametrize("parametrized", [False, True])
def test_switch_register(
    reg, mappable_reg, parametrized, starts_mappable, config_det_map
):
    pulse = Pulse.ConstantPulse(1000, 1, -1, 2)
    with_slm_mask = not starts_mappable and not mappable_reg
    seq = init_seq(
        reg,
        DigitalAnalogDevice,
        "raman",
        "raman_local",
        [pulse],
        initial_target="q0",
        parametrized=parametrized,
        mappable_reg=starts_mappable,
        config_det_map=config_det_map,
        prefer_slm_mask=with_slm_mask,
    )

    with pytest.raises(
        ValueError,
        match="given ids have to be qubit ids declared in this sequence's"
        " register",
    ):
        seq.switch_register(Register(dict(q1=(0, 0), qN=(10, 10))))

    seq.declare_channel("ryd", "rydberg_global")
    seq.add(pulse, "ryd", protocol="no-delay")

    if mappable_reg:
        new_reg = TriangularLatticeLayout(10, 5).make_mappable_register(2)
    else:
        new_reg = Register(dict(q0=(0, 0), foo=(10, 10)))

    if config_det_map and not with_slm_mask:
        context_manager = pytest.warns(
            UserWarning, match="configures a detuning map"
        )
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        new_seq = seq.switch_register(new_reg)
    assert seq.declared_variables or not parametrized
    assert seq.declared_variables == new_seq.declared_variables
    assert new_seq.is_parametrized() == parametrized
    assert new_seq.is_register_mappable() == mappable_reg
    assert new_seq._calls[1:] == seq._calls[1:]  # Excludes __init__
    assert new_seq._to_build_calls == seq._to_build_calls

    build_kwargs = {}
    if parametrized:
        build_kwargs["delay"] = 120
    if mappable_reg:
        build_kwargs["qubits"] = {"q0": 1, "q1": 4}
    if build_kwargs:
        new_seq = new_seq.build(**build_kwargs)

    assert isinstance(
        (raman_pulse_slot := new_seq._schedule["raman"][1]).type, Pulse
    )
    assert raman_pulse_slot.type == pulse
    assert raman_pulse_slot.targets == {"q0"}

    assert isinstance(
        (rydberg_pulse_slot := new_seq._schedule["ryd"][1]).type, Pulse
    )
    assert rydberg_pulse_slot.type == pulse
    assert rydberg_pulse_slot.targets == set(new_reg.qubit_ids)

    if config_det_map:
        if with_slm_mask:
            if parametrized:
                seq = seq.build(**build_kwargs)
            assert np.any(reg.qubits["q0"] != new_reg.qubits["q0"])
            assert "dmm_0" in seq.declared_channels
            prev_qubit_wmap = seq._schedule[
                "dmm_0"
            ].detuning_map.get_qubit_weight_map(reg.qubits)
            new_qubit_wmap = new_seq._schedule[
                "dmm_0"
            ].detuning_map.get_qubit_weight_map(new_reg.qubits)
            assert prev_qubit_wmap["q0"] == 1.0
            assert new_qubit_wmap == dict(q0=1.0, foo=0.0)
        elif not parametrized:
            assert (
                seq._schedule["dmm_0"].detuning_map
                == new_seq._schedule["dmm_0"].detuning_map
            )


@pytest.mark.parametrize("mappable_reg", [False, True])
@pytest.mark.parametrize("parametrized", [False, True])
def test_switch_device_down(
    reg, det_map, devices, pulses, mappable_reg, parametrized
):
    phys_Chadoq2 = dataclasses.replace(
        DigitalAnalogDevice,
        dmm_objects=(
            dataclasses.replace(
                DigitalAnalogDevice.dmm_objects[0], total_bottom_detuning=-2000
            ),
        ),
    )
    # Device checkout
    seq = init_seq(
        reg,
        phys_Chadoq2,
        "ising",
        "rydberg_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    with pytest.warns(
        UserWarning,
        match="Switching a sequence to the same device"
        + " returns the sequence unchanged.",
    ):
        seq.switch_device(phys_Chadoq2)

    # From sequence reusing channels to Device without reusable channels
    seq = init_seq(
        reg,
        dataclasses.replace(phys_Chadoq2.to_virtual(), reusable_channels=True),
        "global",
        "rydberg_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    seq.declare_channel("raman", "raman_local", ["q0"])
    seq.declare_channel("raman_1", "raman_local", ["q0"])
    with pytest.raises(
        TypeError,
        match="No match for channel raman_1 with the"
        " right type, basis and addressing.",
    ):
        # Can't find a match for the 2nd raman_local
        seq.switch_device(phys_Chadoq2)

    with pytest.raises(
        TypeError,
        match="No match for channel raman_1 with the"
        " right type, basis and addressing.",
    ):
        # Can't find a match for the 2nd raman_local
        seq.switch_device(phys_Chadoq2, strict=True)

    with pytest.raises(
        ValueError,
        match="No match for channel raman_1 with the" " same clock_period.",
    ):
        # Can't find a match for the 2nd rydberg_local
        seq.switch_device(
            dataclasses.replace(
                phys_Chadoq2,
                channel_objects=(
                    DigitalAnalogDevice.channels["rydberg_global"],
                    dataclasses.replace(
                        DigitalAnalogDevice.channels["raman_local"],
                        clock_period=10,
                    ),
                    DigitalAnalogDevice.channels["raman_local"],
                ),
                channel_ids=(
                    "rydberg_global",
                    "rydberg_local",
                    "rydberg_local1",
                ),
            ),
            strict=True,
        )

    # From sequence reusing DMMs to Device without reusable channels
    seq = init_seq(
        reg,
        dataclasses.replace(phys_Chadoq2.to_virtual(), reusable_channels=True),
        "global",
        "rydberg_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
        config_det_map=True,
    )
    seq.config_detuning_map(det_map, dmm_id="dmm_0")
    assert list(seq.declared_channels.keys()) == [
        "global",
        "dmm_0",
        "dmm_0_1",
    ]

    with pytest.raises(
        TypeError,
        match="No match for channel dmm_0_1 with the"
        " right type, basis and addressing.",
    ):
        # Can't find a match for the 2nd dmm_0
        seq.switch_device(phys_Chadoq2)
    # There is no need to have same bottom detuning to have a strict switch
    dmm_down = dataclasses.replace(
        phys_Chadoq2.dmm_channels["dmm_0"], bottom_detuning=-10
    )
    new_seq = seq.switch_device(
        dataclasses.replace(phys_Chadoq2, dmm_objects=(dmm_down, dmm_down)),
        strict=True,
    )
    assert list(new_seq.declared_channels.keys()) == [
        "global",
        "dmm_0",
        "dmm_1",
    ]
    seq.add_dmm_detuning(ConstantWaveform(100, -20), "dmm_0_1")
    seq.add_dmm_detuning(ConstantWaveform(100, -20), dmm_name="dmm_0_1")
    # Still works with reusable channels
    new_seq = seq.switch_device(
        dataclasses.replace(
            phys_Chadoq2.to_virtual(),
            reusable_channels=True,
            dmm_objects=(dataclasses.replace(dmm_down, bottom_detuning=-20),),
        ),
        strict=True,
    )
    assert list(new_seq.declared_channels.keys()) == [
        "global",
        "dmm_0",
        "dmm_0_1",
    ]
    # Still one compatible configuration
    new_seq = seq.switch_device(
        dataclasses.replace(
            phys_Chadoq2,
            dmm_objects=(phys_Chadoq2.dmm_channels["dmm_0"], dmm_down),
        ),
        strict=True,
    )
    assert list(new_seq.declared_channels.keys()) == [
        "global",
        "dmm_1",
        "dmm_0",
    ]
    # No compatible configuration
    error_msg = (
        "No matching found between declared channels and channels in the "
        "new device that does not modify the samples of the Sequence. "
        "Here is a list of matchings tested and their associated errors: "
        "{(('global', 'rydberg_global'), ('dmm_0', 'dmm_0'), ('dmm_0_1', "
        "'dmm_1')): ('The detunings on some atoms go below the local bottom "
        "detuning of the DMM (-10 rad/µs).',), (('global', 'rydberg_global'), "
        "('dmm_0', 'dmm_1'), ('dmm_0_1', 'dmm_0')): ('The detunings on some "
        "atoms go below the local bottom detuning of the DMM (-10 rad/µs).',)}"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        seq.switch_device(
            dataclasses.replace(
                phys_Chadoq2, dmm_objects=(dmm_down, dmm_down)
            ),
            strict=True,
        )
    dmm_down = dataclasses.replace(
        phys_Chadoq2.dmm_channels["dmm_0"],
        bottom_detuning=-10,
        total_bottom_detuning=-10,
    )
    seq.switch_device(
        dataclasses.replace(
            phys_Chadoq2,
            dmm_objects=(phys_Chadoq2.dmm_channels["dmm_0"], dmm_down),
        ),
        strict=True,
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        seq.switch_device(
            dataclasses.replace(
                phys_Chadoq2, dmm_objects=(dmm_down, dmm_down)
            ),
            strict=True,
        )
    seq_ising = init_seq(
        reg,
        MockDevice,
        "ising",
        "rydberg_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )

    seq_xy = init_seq(
        reg,
        MockDevice,
        "microwave",
        "mw_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    mod_mock = dataclasses.replace(
        MockDevice, rydberg_level=50, interaction_coeff_xy=100.0
    )
    for seq, msg in [
        (seq_ising, "Rydberg level"),
        (seq_xy, "XY interaction coefficient"),
    ]:
        with pytest.raises(
            ValueError,
            match="Strict device match failed because the devices"
            f" have different {msg}s.",
        ):
            seq.switch_device(mod_mock, True)

        with pytest.warns(
            UserWarning,
            match=f"Switching to a device with a different {msg},"
            " check that the expected interactions still hold.",
        ):
            seq.switch_device(mod_mock, False)

    seq = init_seq(
        reg,
        devices[0],
        "ising",
        "raman_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    for dev_ in (
        DigitalAnalogDevice,  # Different Channels type / basis
        devices[1],  # Different addressing channels
    ):
        with pytest.raises(
            TypeError,
            match="No match for channel ising with the"
            + " right type, basis and addressing.",
        ):
            seq.switch_device(dev_)

    # Clock_period not match
    seq = init_seq(
        reg,
        devices[0],
        channel_name="ising",
        channel_id="rydberg_global",
        l_pulses=pulses[:2],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    with pytest.raises(
        ValueError,
        match="No match for channel ising with the same clock_period.",
    ):
        seq.switch_device(devices[1], True)

    seq = init_seq(
        reg,
        devices[2],
        channel_name="digital",
        channel_id="rmn_local1",
        l_pulses=[],
        initial_target=["q0"],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    with pytest.raises(
        ValueError,
        match="No match for channel digital with the same mod_bandwidth.",
    ):
        seq.switch_device(devices[0], True)

    with pytest.raises(
        ValueError,
        match="No match for channel digital"
        + " with the same fixed_retarget_t.",
    ):
        seq.switch_device(devices[1], True)

    seq = init_seq(
        reg,
        devices[2],
        channel_name="digital",
        channel_id="rmn_local3",
        l_pulses=[],
        initial_target=["q0"],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    with pytest.raises(
        ValueError,
        match="No match for channel digital"
        + " with the same min_retarget_interval.",
    ):
        seq.switch_device(DigitalAnalogDevice, True)


@pytest.mark.parametrize("mappable_reg", [False, True])
@pytest.mark.parametrize("trap_id", [20, 38, 50])
@pytest.mark.parametrize("parametrized", [False, True])
@pytest.mark.parametrize("config_det_map", [False, True])
@pytest.mark.parametrize("device_ind, strict", [(1, False), (2, True)])
def test_switch_device_up(
    reg,
    device_ind,
    devices,
    pulses,
    strict,
    mappable_reg,
    trap_id,
    parametrized,
    config_det_map,
):
    # Device checkout
    seq = init_seq(
        reg,
        DigitalAnalogDevice,
        "ising",
        "rydberg_global",
        None,
        parametrized=parametrized,
        mappable_reg=mappable_reg,
        config_det_map=config_det_map,
    )
    with pytest.warns(
        UserWarning,
        match="Switching a sequence to the same device returns the "
        "sequence unchanged",
    ):
        assert (
            seq.switch_device(DigitalAnalogDevice)._device
            == DigitalAnalogDevice
        )
    # Test non-strict mode
    assert "ising" in seq.switch_device(devices[0]).declared_channels

    # Strict: Jump_phase_time & CLock-period criteria
    # Jump_phase_time check 1: phase not null
    mod_wvf = ConstantWaveform(100, -10)
    seq1 = init_seq(
        reg,
        devices[device_ind],
        channel_name="ising",
        channel_id="rydberg_global",
        l_pulses=pulses[:2],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
        config_det_map=config_det_map,
    )
    if config_det_map:
        seq1.add_dmm_detuning(mod_wvf, "dmm_0")
    seq2 = init_seq(
        reg,
        devices[0],
        channel_name="ising",
        channel_id="rydberg_global",
        l_pulses=pulses[:2],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
        config_det_map=config_det_map,
    )
    if config_det_map:
        seq2.add_dmm_detuning(mod_wvf, "dmm_0")
    new_seq = seq1.switch_device(devices[0], strict)
    build_kwargs = {}
    if parametrized:
        build_kwargs["delay"] = 120
    if mappable_reg:
        build_kwargs["qubits"] = {"q0": trap_id}

    if build_kwargs:
        seq1 = seq1.build(**build_kwargs)
        seq2 = seq2.build(**build_kwargs)
        new_seq = new_seq.build(**build_kwargs)
    s1 = sample(new_seq)
    s2 = sample(seq1)
    s3 = sample(seq2)
    nested_s1_glob = s1.to_nested_dict()["Global"]["ground-rydberg"]
    nested_s2_glob = s2.to_nested_dict()["Global"]["ground-rydberg"]
    nested_s3_glob = s3.to_nested_dict()["Global"]["ground-rydberg"]
    if config_det_map:
        nested_s1_loc = s1.to_nested_dict()["Local"]["ground-rydberg"]["q0"]
        nested_s2_loc = s2.to_nested_dict()["Local"]["ground-rydberg"]["q0"]
        nested_s3_loc = s3.to_nested_dict()["Local"]["ground-rydberg"]["q0"]
    # Check if the samples are the same
    for key in ["amp", "det", "phase"]:
        np.testing.assert_array_equal(nested_s1_glob[key], nested_s3_glob[key])
        if strict:
            np.testing.assert_array_equal(
                nested_s1_glob[key], nested_s2_glob[key]
            )
        if config_det_map:
            for nested_s_loc in [
                nested_s1_loc[key],
                nested_s2_loc[key],
                nested_s3_loc[key],
            ]:
                if key != "det":
                    assert np.all(nested_s_loc == 0.0)
                elif mappable_reg:
                    # modulates detuning map on trap ids 0, 1, 3, 4
                    mod_trap_ids = [20, 32, 54, 66]
                    assert np.all(
                        nested_s_loc[:100]
                        == (-10.0 if trap_id in mod_trap_ids else 0)
                    )
                else:
                    # first pulse is covered by SLM Mask
                    np.all(
                        nested_s_loc[:252]
                        == -10 * np.max(pulses[0].amplitude.samples)
                    )
                    # Modulated pulse added afterwards
                    assert np.all(nested_s_loc[252:352] == -10)

    # Channels with the same mod_bandwidth and fixed_retarget_t
    seq = init_seq(
        reg,
        devices[2],
        channel_name="digital",
        channel_id="rmn_local2",
        l_pulses=[],
        initial_target=["q0"],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    assert seq.switch_device(devices[1], True)._device == devices[1]
    assert "digital" in seq.switch_device(devices[1], True).declared_channels


extended_eom = dataclasses.replace(
    cast(RydbergEOM, AnalogDevice.channels["rydberg_global"].eom_config),
    controlled_beams=tuple(RydbergBeam),
    multiple_beam_control=True,
    custom_buffer_time=None,
)
extended_eom_channel = dataclasses.replace(
    AnalogDevice.channels["rydberg_global"], eom_config=extended_eom
)
extended_eom_device = dataclasses.replace(
    AnalogDevice, channel_objects=(extended_eom_channel,)
)


@pytest.mark.parametrize("device", [AnalogDevice, extended_eom_device])
@pytest.mark.parametrize("mappable_reg", [False, True])
@pytest.mark.parametrize("parametrized", [False, True])
@pytest.mark.parametrize(
    "extension_arg", ["amp", "control", "2control", "buffer_time"]
)
def test_switch_device_eom(
    reg, device, mappable_reg, parametrized, extension_arg, patch_plt_show
):
    # Sequence with EOM blocks
    seq = init_seq(
        reg,
        dataclasses.replace(device, max_atom_num=28),
        "rydberg",
        "rydberg_global",
        [],
        parametrized=parametrized,
        mappable_reg=mappable_reg,
    )
    seq.enable_eom_mode("rydberg", amp_on=2.0, detuning_on=0.0)
    seq.add_eom_pulse("rydberg", 100, 0.0)
    seq.delay(200, "rydberg")
    assert seq.is_in_eom_mode("rydberg")

    err_base = "No match for channel rydberg "
    warns_msg = (
        "Switching to a device with a different Rydberg level,"
        " check that the expected interactions still hold."
    )
    with pytest.warns(UserWarning, match=warns_msg), pytest.raises(
        TypeError, match=err_base + "with an EOM configuration."
    ):
        seq.switch_device(DigitalAnalogDevice)

    ch_obj = seq.declared_channels["rydberg"]
    wrong_eom_config = dataclasses.replace(ch_obj.eom_config, mod_bandwidth=20)
    wrong_ch_obj = dataclasses.replace(ch_obj, eom_config=wrong_eom_config)
    wrong_analog = dataclasses.replace(
        device, channel_objects=(wrong_ch_obj,), max_atom_num=28
    )
    if parametrized:
        # Can't switch if the two EOM configurations don't match
        # If the modulation bandwidth is different
        with pytest.raises(
            ValueError, match=err_base + "with the same EOM configuration."
        ):
            seq.switch_device(wrong_analog, strict=True)
        down_eom_configs = {
            # If the amplitude is different
            "amp": dataclasses.replace(
                ch_obj.eom_config, max_limiting_amp=10 * 2 * np.pi
            ),
            # If less controlled beams/the controlled beam is not the same
            "control": dataclasses.replace(
                ch_obj.eom_config,
                controlled_beams=(RydbergBeam.RED,),
                multiple_beam_control=False,
            ),
            # If the multiple_beam_control is not the same
            "2control": dataclasses.replace(
                ch_obj.eom_config,
                controlled_beams=(
                    tuple(RydbergBeam)
                    if device == extended_eom_device
                    else (RydbergBeam.RED,)
                ),
                multiple_beam_control=False,
            ),
            # If the buffer time is different
            "buffer_time": dataclasses.replace(
                ch_obj.eom_config,
                custom_buffer_time=300,
            ),
        }
        wrong_ch_obj = dataclasses.replace(
            ch_obj, eom_config=down_eom_configs[extension_arg]
        )
        wrong_analog = dataclasses.replace(
            device, channel_objects=(wrong_ch_obj,), max_atom_num=28
        )
        with pytest.raises(
            ValueError, match=err_base + "with the same EOM configuration."
        ):
            seq.switch_device(wrong_analog, strict=True)
    else:
        # Can't switch to eom if the modulation bandwidth doesn't match
        with pytest.raises(
            ValueError,
            match=err_base + "with the same mod_bandwidth for the EOM.",
        ):
            seq.switch_device(wrong_analog, strict=True)
    # Can if one Channel has a correct EOM configuration
    new_seq = seq.switch_device(
        dataclasses.replace(
            wrong_analog,
            channel_objects=(wrong_ch_obj, ch_obj),
            channel_ids=("wrong_eom", "good_eom"),
        ),
        strict=True,
    )
    assert new_seq.declared_channels == {"rydberg": ch_obj}
    # Can if eom extends current eom
    up_eom_configs = {
        # Still raises for max_amplitude in parametrized Sequence
        "amp": dataclasses.replace(
            ch_obj.eom_config, max_limiting_amp=40 * 2 * np.pi
        ),
        # With one controlled beam, don't care about multiple_beam_control
        # Raises an error if device is extended_eom_device (less options)
        "control": dataclasses.replace(
            ch_obj.eom_config,
            controlled_beams=(RydbergBeam.BLUE,),
            multiple_beam_control=False,
        ),
        # Using 2 controlled beams
        # Raises an error if device is extended_eom_device (less options)
        "2control": dataclasses.replace(
            ch_obj.eom_config,
            controlled_beams=tuple(RydbergBeam),
            multiple_beam_control=False,
        ),
        # If custom buffer time is None
        # Raises an error if device is extended_eom_device
        "buffer_time": dataclasses.replace(
            ch_obj.eom_config,
            custom_buffer_time=None,
        ),
    }
    up_eom_config = up_eom_configs[extension_arg]
    up_ch_obj = dataclasses.replace(ch_obj, eom_config=up_eom_config)
    up_analog = dataclasses.replace(
        device, channel_objects=(up_ch_obj,), max_atom_num=28
    )
    if (
        (parametrized and extension_arg == "amp")
        or (
            parametrized
            and extension_arg in ["control", "2control"]
            and device == extended_eom_device
        )
        or (
            parametrized
            and extension_arg == "buffer_time"
            and device == AnalogDevice
        )
    ):
        with pytest.raises(
            ValueError,
            match=err_base + "with the same EOM configuration.",
        ):
            seq.switch_device(up_analog, strict=True)
        return
    if device == extended_eom_device:
        if extension_arg in ["control", "2control"]:
            with pytest.raises(
                ValueError,
                match="No match for channel rydberg with an EOM configuration",
            ):
                seq.switch_device(up_analog, strict=True)
            return
        elif extension_arg == "buffer_time":
            with pytest.warns(
                UserWarning, match="Switching a sequence to the same device"
            ):
                up_seq = seq.switch_device(up_analog, strict=True)
        else:
            up_seq = seq.switch_device(up_analog, strict=True)
    else:
        up_seq = seq.switch_device(up_analog, strict=True)
    build_kwargs = {}
    if parametrized:
        build_kwargs["delay"] = 120
    if mappable_reg:
        build_kwargs["qubits"] = {"q0": 0}
    og_eom_block = (
        (seq.build(**build_kwargs) if build_kwargs else seq)
        ._schedule["rydberg"]
        .eom_blocks[0]
    )
    up_eom_block = (
        (up_seq.build(**build_kwargs) if build_kwargs else up_seq)
        ._schedule["rydberg"]
        .eom_blocks[0]
    )
    assert og_eom_block.detuning_on == up_eom_block.detuning_on
    assert og_eom_block.rabi_freq == up_eom_block.rabi_freq
    assert og_eom_block.detuning_off == up_eom_block.detuning_off

    # Some parameters might modify the samples
    mod_eom_config = dataclasses.replace(
        ch_obj.eom_config, max_limiting_amp=5 * 2 * np.pi
    )
    mod_ch_obj = dataclasses.replace(ch_obj, eom_config=mod_eom_config)
    mod_analog = dataclasses.replace(
        device, channel_objects=(mod_ch_obj,), max_atom_num=28
    )
    err_msg = (
        "No matching found between declared channels and channels in "
        "the new device that does not modify the samples of the "
        "Sequence. Here is a list of matchings tested and their "
        "associated errors: {(('rydberg', 'rydberg_global'),): ('No "
        "match for channel rydberg with an EOM configuration that "
        "does not change the samples."
    )
    if parametrized:
        with pytest.raises(
            ValueError,
            match=err_base + "with the same EOM configuration.",
        ):
            seq.switch_device(mod_analog, strict=True)
        return
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        seq.switch_device(mod_analog, strict=True)
    mod_seq = seq.switch_device(mod_analog, strict=False)
    mod_eom_block = (
        (mod_seq.build(**build_kwargs) if build_kwargs else mod_seq)
        ._schedule["rydberg"]
        .eom_blocks[0]
    )
    assert og_eom_block.detuning_on == mod_eom_block.detuning_on
    assert og_eom_block.rabi_freq == mod_eom_block.rabi_freq
    assert og_eom_block.detuning_off != mod_eom_block.detuning_off

    # Test drawing in eom mode
    (seq.build(**build_kwargs) if build_kwargs else seq).draw()


def test_target(reg, device):
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q1")
    seq.declare_channel("ch1", "rydberg_global")

    with pytest.raises(ValueError, match="name of a declared channel"):
        seq.target("q0", "ch2")
    with pytest.raises(ValueError, match="ids have to be qubit ids"):
        seq.target(0, "ch0")
    with pytest.raises(ValueError, match="ids have to be qubit ids"):
        seq.target("0", "ch0")
    with pytest.raises(ValueError, match="Can only choose target of 'Local'"):
        seq.target("q3", "ch1")
    with pytest.raises(ValueError, match="can target at most 1 qubits"):
        seq.target(["q1", "q5"], "ch0")
    with pytest.raises(ValueError, match="Need at least one qubit to target"):
        seq.target([], "ch0")

    assert seq._schedule["ch0"][-1] == _TimeSlot("target", -1, 0, {"q1"})
    seq.target("q4", "ch0")
    retarget_t = seq.declared_channels["ch0"].min_retarget_interval
    assert seq._schedule["ch0"][-1] == _TimeSlot(
        "target", 0, retarget_t, {"q4"}
    )
    seq.target("q4", "ch0")  # targets the same qubit
    seq.target("q20", "ch0")
    assert seq._schedule["ch0"][-1] == _TimeSlot(
        "target", retarget_t, 2 * retarget_t, {"q20"}
    )
    seq.delay(216, "ch0")
    seq.target("q2", "ch0")
    ti = 2 * retarget_t + 216
    tf = ti + 16
    assert seq._schedule["ch0"][-1] == _TimeSlot("target", ti, tf, {"q2"})

    seq.delay(220, "ch0")
    seq.target("q1", "ch0")
    ti = tf + 220
    tf = ti
    assert seq._schedule["ch0"][-1] == _TimeSlot("target", ti, tf, {"q1"})

    seq.delay(100, "ch0")
    seq.target("q10", "ch0")
    ti = tf + 100
    tf = ti + 120
    assert seq._schedule["ch0"][-1] == _TimeSlot("target", ti, tf, {"q10"})

    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("ch0", "raman_local", initial_target={"q1", "q10"})

    # Test unlimited targets with Local channel when 'max_targets=None'
    assert seq2.declared_channels["ch0"].max_targets is None
    seq2.target(set(reg.qubit_ids) - {"q2"}, "ch0")

    seq2.phase_shift(1, "q2")
    with pytest.raises(ValueError, match="qubits with different phase"):
        seq2.target({"q3", "q1", "q2"}, "ch0")


@pytest.mark.parametrize("at_rest", [True, False])
def test_delay(reg, device, at_rest):
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local")
    with pytest.raises(ValueError, match="Use the name of a declared channel"):
        seq.delay(1e3, "ch01")
    with pytest.raises(ValueError, match="channel has no target"):
        seq.delay(100, "ch0")
    seq.target("q19", "ch0")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ch0")
    # At rest will have no effect
    assert seq.declared_channels["ch0"].mod_bandwidth is None
    seq.delay(388, "ch0", at_rest)
    assert seq._last("ch0") == (
        last_slot := _TimeSlot("delay", 100, 488, {"q19"})
    )
    seq.delay(0, "ch0", at_rest)
    # A delay of 0 is not added to the schedule
    assert seq._last("ch0") == last_slot


@pytest.mark.parametrize("delay_duration", [200, 0])
@pytest.mark.parametrize("at_rest", [True, False])
@pytest.mark.parametrize("in_eom", [True, False])
def test_delay_at_rest(in_eom, at_rest, delay_duration):
    seq = Sequence(Register.square(2, 5), AnalogDevice)
    seq.declare_channel("ryd", "rydberg_global")
    assert (ch_obj := seq.declared_channels["ryd"]).mod_bandwidth is not None
    pulse = Pulse.ConstantPulse(100, 1, 0, 0)
    assert pulse.duration == 100
    if in_eom:
        seq.enable_eom_mode("ryd", 1, 0, 0)
        seq.add_eom_pulse("ryd", pulse.duration, 0)
    else:
        seq.add(pulse, "ryd")
    assert (extra_delay := pulse.fall_time(ch_obj, in_eom_mode=in_eom)) > 0
    seq.delay(delay_duration, "ryd", at_rest=at_rest)
    assert seq.get_duration() == pulse.duration + delay_duration + (
        extra_delay * at_rest
    )


def test_delay_min_duration(reg, device):
    # Check that a delay shorter than a channel's minimal duration
    # is automatically extended to that minimal duration
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "rydberg_local")
    seq.target("q0", "ch1")
    pulse0 = Pulse.ConstantPulse(52, 1, 1, 0)
    pulse1 = Pulse.ConstantPulse(180, 1, 1, 0)
    seq.add(pulse1, "ch1")
    seq.add(pulse0, "ch0")
    seq.target("q1", "ch1")
    seq.add(pulse1, "ch1")
    min_duration = seq.declared_channels["ch1"].min_duration
    assert seq._schedule["ch1"][3] == _TimeSlot(
        "delay", 220, 220 + min_duration, {"q1"}
    )


def test_phase(reg, device, det_map):
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    seq.phase_shift(-1, "q0", "q1")
    with pytest.raises(ValueError, match="id of a qubit declared"):
        seq.current_phase_ref(0, "digital")
    with pytest.raises(ValueError, match="targets the given 'basis'"):
        seq.current_phase_ref("q1", "ground-rydberg")
    with pytest.raises(ValueError, match="No declared channel targets"):
        seq.phase_shift(1, "q3", basis="hyperfine")
    assert seq.current_phase_ref("q0", "digital") == 2 * np.pi - 1

    # Phase shifts of 0
    seq.phase_shift(0, "q0")
    seq.phase_shift(-8 * np.pi, "q1")
    assert seq.current_phase_ref("q0", "digital") == 2 * np.pi - 1
    assert seq.current_phase_ref("q1", "digital") == 2 * np.pi - 1

    with pytest.raises(ValueError, match="ids have to be qubit ids"):
        seq.phase_shift(np.pi, "q1", "q4", "q100")

    seq.declare_channel("ch1", "rydberg_global")
    seq.phase_shift(1, *seq._qids, basis="ground-rydberg")
    for q in seq.qubit_info:
        assert seq.current_phase_ref(q, "ground-rydberg") == 1
    seq.phase_shift(1, *seq._qids)
    assert seq.current_phase_ref("q1", "digital") == 0
    assert seq.current_phase_ref("q10", "digital") == 1

    # Check that the phase of DMM pulses is unaffected
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ch1")
    seq.config_detuning_map(det_map, "dmm_0")
    det_wf = RampWaveform(100, -10, -1)
    seq.add_dmm_detuning(det_wf, "dmm_0")
    # We shift the phase of just one qubit, which blocks addition
    # of new pulses on this basis
    seq.phase_shift(1.0, "q0", basis="ground-rydberg")
    with pytest.raises(
        ValueError,
        match="Cannot do a multiple-target pulse on qubits with different "
        "phase references for the same basis.",
    ):
        seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ch1")
    # But it works on the DMM
    seq.add_dmm_detuning(det_wf, "dmm_0")

    seq_samples = sample(seq)
    # The phase on the rydberg channel matches the phase ref
    np.testing.assert_array_equal(
        seq_samples.channel_samples["ch1"].phase,
        seq.current_phase_ref("q1", basis="ground-rydberg"),
    )

    # but the phase in the DMMSamples stays at zero
    np.testing.assert_array_equal(
        sample(seq).channel_samples["dmm_0"].phase, 0.0
    )


def test_align(reg, device):
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    seq.declare_channel("ch1", "rydberg_global")
    with pytest.raises(ValueError, match="names must correspond to declared"):
        seq.align("ch0", "ch1", "ch2")
    with pytest.raises(ValueError, match="more than once"):
        seq.align("ch0", "ch1", "ch0")
    with pytest.raises(ValueError, match="at least two channels"):
        seq.align()
        seq.align("ch1")


@pytest.mark.parametrize("parametrized", [True, False])
def test_measure(reg, parametrized):
    pulse = Pulse.ConstantPulse(500, 2, -10, 0, post_phase_shift=np.pi)
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    t = seq.declare_variable("t", dtype=int)
    seq.delay(t if parametrized else 100, "ch0")
    assert seq.is_parametrized() == parametrized

    assert "XY" in MockDevice.supported_bases
    with pytest.raises(ValueError, match="not supported"):
        seq.measure(basis="XY")
    with pytest.raises(
        RuntimeError, match="The sequence has not been measured"
    ):
        seq.get_measurement_basis()
    with pytest.warns(
        UserWarning,
        match="'digital' is not being addressed by "
        "any channel in the sequence",
    ):
        seq.measure(basis="digital")
    assert seq.get_measurement_basis() == "digital"
    with pytest.raises(
        RuntimeError,
        match="sequence has been measured, no further changes are allowed.",
    ):
        seq.add(pulse, "ch0")

    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "mw_global")
    assert "digital" in MockDevice.supported_bases
    with pytest.raises(ValueError, match="not supported"):
        seq.measure(basis="digital")
    seq.measure(basis="XY")


@pytest.mark.parametrize(
    "call, args",
    [
        ("declare_channel", ("ch1", "rydberg_global")),
        ("add", (Pulse.ConstantPulse(1000, 1, 0, 0), "ch0")),
        ("target", ("q1", "ch0")),
        ("target_index", (2, "ch0")),
        ("delay", (1000, "ch0")),
        ("align", ("ch0", "ch01")),
        ("measure", tuple()),
    ],
)
def test_block_if_measured(reg, call, args):
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_local", initial_target="q0")
    # For the align command
    seq.declare_channel("ch01", "rydberg_local", initial_target="q0")
    # Check there's nothing wrong with the call
    if call != "measure":
        getattr(seq, call)(*args)
    seq.measure(basis="ground-rydberg")
    with pytest.raises(
        RuntimeError,
        match="sequence has been measured, no further changes are allowed.",
    ):
        getattr(seq, call)(*args)


def test_str(reg, device, mod_device, det_map):
    seq = Sequence(reg, mod_device)
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    pulse = Pulse.ConstantPulse(500, 2, -10, 0, post_phase_shift=np.pi)
    seq.add(pulse, "ch0")
    seq.delay(300, "ch0")
    seq.target("q7", "ch0")

    seq.declare_channel("ch1", "rydberg_global")
    seq.enable_eom_mode("ch1", 2, 0, optimal_detuning_off=10.0)
    seq.add_eom_pulse("ch1", duration=100, phase=0, protocol="no-delay")
    seq.delay(500, "ch1")

    seq.config_detuning_map(det_map, "dmm_0")
    seq.add_dmm_detuning(ConstantWaveform(100, -10), "dmm_0")
    seq.add_dmm_detuning(RampWaveform(100, -10, 0), "dmm_0")

    seq.measure("digital")
    msg_ch0 = (
        "Channel: ch0\nt: 0 | Initial targets: q0 | Phase Reference: 0.0 "
        + "\nt: 0->500 | Pulse(Amp=2 rad/µs, Detuning=-10 rad/µs, Phase=0) "
        + "| Targets: q0\nt: 500->800 | Delay \nt: 800->800 | Target: q7 | "
        + "Phase Reference: 0.0"
    )
    targets = ", ".join(sorted(reg.qubit_ids))
    msg_ch1 = (
        f"\n\nChannel: ch1\nt: 0 | Initial targets: {targets} "
        "| Phase Reference: 0.0 "
        "\nt: 0->100 | Pulse(Amp=2 rad/µs, Detuning=0 rad/µs, Phase=0) "
        f"| Targets: {targets}"
        "\nt: 100->600 | Detuned Delay | Detuning: -1 rad/µs"
    )

    msg_det_map = (
        f"\n\nChannel: dmm_0\nt: 0 | Initial targets: {targets} "
        "| Phase Reference: 0.0 "
        f"\nt: 0->100 | Detuning: -10 rad/µs | Targets: {targets}"
        f"\nt: 100->200 | Detuning: Ramp(-10->0) rad/µs | Targets: {targets}"
    )

    measure_msg = "\n\nMeasured in basis: digital"
    assert seq.__str__() == msg_ch0 + msg_ch1 + msg_det_map + measure_msg
    with pytest.warns(
        DeprecationWarning,
        match="Usage of `int`s or any non-`str`types as `QubitId`s",
    ):
        seq2 = Sequence(Register({"q0": (0, 0), 1: (5, 5)}), device)
    seq2.declare_channel("ch1", "rydberg_global")
    with pytest.raises(
        NotImplementedError,
        match="Can't print sequence with qubit IDs of different types.",
    ):
        str(seq2)

    # Check qubit IDs are sorted
    seq3 = Sequence(Register({"q1": (0, 0), "q0": (5, 5)}), device)
    seq3.declare_channel("ch2", "rydberg_global")
    assert str(seq3) == (
        "Channel: ch2\n"
        "t: 0 | Initial targets: q0, q1 | Phase Reference: 0.0 \n\n"
    )


def test_sequence(reg, device, patch_plt_show):
    seq = Sequence(reg, device)
    assert seq.get_duration() == 0
    with pytest.raises(RuntimeError, match="empty sequence"):
        seq.draw()
    seq.declare_channel("ch0", "raman_local", initial_target="q0")
    seq.declare_channel("ch1", "rydberg_local", initial_target="q0")
    seq.declare_channel("ch2", "rydberg_global")
    assert seq.get_duration("ch0") == 0
    assert seq.get_duration("ch2") == 0

    with patch("matplotlib.figure.Figure.savefig"):
        seq.draw(fig_name="my_sequence.pdf")
        seq.draw(draw_register=True, fig_name="both.pdf")

    pulse1 = Pulse(
        InterpolatedWaveform(500, [0, 1, 0]),
        InterpolatedWaveform(500, [-1, 1, 0]),
        phase=0,
        post_phase_shift=np.pi,
    )
    pulse2 = Pulse.ConstantDetuning(
        BlackmanWaveform(1e3, np.pi / 4), 25, np.pi, post_phase_shift=1
    )
    with pytest.raises(TypeError):
        seq.add([1, 5, 3], "ch0")
    with pytest.raises(ValueError, match="amplitude goes over the maximum"):
        seq.add(
            Pulse.ConstantPulse(20, 2 * np.pi * 10, -2 * np.pi * 100, 0), "ch2"
        )
    with pytest.raises(
        ValueError, match="detuning values go out of the range"
    ):
        seq.add(
            Pulse.ConstantPulse(500, 2 * np.pi, -2 * np.pi * 100, 0), "ch0"
        )
    seq.phase_shift(np.pi, "q0", basis="ground-rydberg")
    with pytest.raises(ValueError, match="qubits with different phase ref"):
        seq.add(pulse2, "ch2")
    with pytest.raises(ValueError, match="Invalid protocol"):
        seq.add(pulse1, "ch0", protocol="now")

    wf_ = CompositeWaveform(BlackmanWaveform(30, 1), RampWaveform(15, 0, 2))
    with pytest.raises(TypeError, match="Failed to automatically adjust"):
        with pytest.warns(UserWarning, match="rounded up to 48 ns"):
            seq.add(Pulse.ConstantAmplitude(1, wf_, 0), "ch0")

    pulse1_ = Pulse.ConstantPulse(499, 2, -10, 0, post_phase_shift=np.pi)
    with pytest.warns(UserWarning, match="rounded up to 500 ns"):
        seq.add(pulse1_, "ch0")
    seq.add(pulse1, "ch1")
    seq.add(pulse2, "ch2")

    assert seq._last("ch0").ti == 0
    assert seq._last("ch0").tf == seq._last("ch1").ti
    assert seq._last("ch2").tf == seq._last("ch2").ti + 1000
    assert seq.current_phase_ref("q0", "digital") == np.pi

    seq.add(pulse1, "ch2")
    assert seq.get_duration("ch2") == 2500
    seq.add(pulse2, "ch1", protocol="no-delay")
    assert seq.get_duration("ch1") == 3500
    seq.add(pulse1, "ch0", protocol="no-delay")
    assert seq._last("ch0").ti == 500
    assert seq.get_duration("ch0") == 1000
    assert seq.current_phase_ref("q0", "digital") == 0
    seq.phase_shift(np.pi / 2, "q1")
    seq.target("q1", "ch0")
    assert seq._basis_ref["digital"]["q1"].last_used == 0
    assert seq._schedule["ch0"].last_target() == 1000
    assert seq._last("ch0").ti == 1000
    assert seq.get_duration("ch0") == 1000
    seq.add(pulse1, "ch0")
    assert seq._last("ch0").ti == 2500
    assert seq.get_duration("ch0") == 3000
    seq.add(pulse1, "ch0", protocol="wait-for-all")
    assert seq._last("ch0").ti == 3500
    assert seq.get_duration("ch2") != seq.get_duration("ch0")
    seq.align("ch0", "ch2")
    assert seq.get_duration("ch2") == seq.get_duration("ch0")

    seq.draw(draw_phase_shifts=True)

    assert seq.get_duration() == 4000

    seq.measure(basis="digital")

    seq.draw(draw_phase_area=True)
    seq.draw(draw_phase_curve=True)
    seq.draw(as_phase_modulated=True)

    s = seq._serialize()
    assert json.loads(s)["__version__"] == pulser.__version__
    seq_ = Sequence._deserialize(s)
    assert str(seq) == str(seq_)


@pytest.mark.parametrize("eom", [False, True])
def test_estimate_added_delay(eom):
    reg = Register.square(2, 5)
    seq = Sequence(reg, AnalogDevice)
    pulse_0 = Pulse.ConstantPulse(100, 1, 0, 0)
    pulse_pi_2 = Pulse.ConstantPulse(100, 1, 0, np.pi / 2)

    with pytest.raises(
        ValueError, match="Use the name of a declared channel."
    ):
        seq.estimate_added_delay(pulse_0, "ising", "min-delay")
    seq.declare_channel("ising", "rydberg_global")
    ising_obj = seq.declared_channels["ising"]
    if eom:
        seq.enable_eom_mode("ising", 1, 0)
        with pytest.warns(
            UserWarning,
            match="Channel ising is in EOM mode, the amplitude",
        ):
            assert (
                seq.estimate_added_delay(
                    Pulse.ConstantPulse(100, 2, 0, 0), "ising"
                )
                == 0
            )
        with pytest.warns(
            UserWarning,
            match="Channel ising is in EOM mode, the detuning",
        ):
            assert (
                seq.estimate_added_delay(
                    Pulse.ConstantPulse(100, 1, 1, 0), "ising"
                )
                == 0
            )
    assert seq.estimate_added_delay(pulse_0, "ising", "min-delay") == 0
    seq._add(pulse_0, "ising", "min-delay")
    first_pulse = seq._last("ising")
    assert first_pulse.ti == 0
    delay = pulse_0.fall_time(ising_obj, eom) + ising_obj.phase_jump_time
    assert seq.estimate_added_delay(pulse_pi_2, "ising") == delay
    seq._add(pulse_pi_2, "ising", "min-delay")
    second_pulse = seq._last("ising")
    assert second_pulse.ti - first_pulse.tf == delay
    assert seq.estimate_added_delay(pulse_0, "ising") == delay
    seq.delay(100, "ising")
    assert seq.estimate_added_delay(pulse_0, "ising") == delay - 100
    with pytest.warns(
        UserWarning,
        match="The sequence's duration exceeded the maximum duration",
    ):
        seq.estimate_added_delay(
            pulser.Pulse.ConstantPulse(4000, 1, 0, np.pi), "ising"
        )
    var = seq.declare_variable("var", dtype=int)
    with pytest.raises(
        ValueError, match="Can't compute the delay to add before a pulse"
    ):
        seq.estimate_added_delay(Pulse.ConstantPulse(var, 1, 0, 0), "ising")
    # We shift the phase of just one qubit, which blocks addition
    # of new pulses on this basis
    seq.phase_shift(1.0, 0, basis="ground-rydberg")
    with pytest.raises(
        ValueError,
        match="Cannot do a multiple-target pulse on qubits with different",
    ):
        seq.estimate_added_delay(pulse_0, "ising")


def test_estimate_added_delay_dmm():
    pulse_0 = Pulse.ConstantPulse(100, 1, 0, 0)
    det_pulse = Pulse.ConstantPulse(100, 0, -1, 0)
    seq = Sequence(Register.square(2, 5), DigitalAnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.config_slm_mask([0, 1])
    with pytest.raises(
        ValueError, match="You should add a Pulse to a Global Channel"
    ):
        seq.estimate_added_delay(det_pulse, "dmm_0")
    seq.add(pulse_0, "ising")
    assert seq.estimate_added_delay(det_pulse, "dmm_0") == 0
    with pytest.raises(
        ValueError, match="The detuning in a DMM must not be positive."
    ):
        seq.estimate_added_delay(Pulse.ConstantPulse(100, 0, 1, 0), "dmm_0")
    with pytest.raises(
        ValueError, match="The pulse's amplitude goes over the maximum"
    ):
        seq.estimate_added_delay(pulse_0, "dmm_0")


@pytest.mark.parametrize("qubit_ids", [["q0", "q1", "q2"], [0, 1, 2]])
def test_config_slm_mask(qubit_ids, device, det_map):
    reg: Register | MappableRegister
    trap_ids = [(0, 0), (10, 10), (-10, -10)]
    reg = Register(dict(zip(qubit_ids, trap_ids)))
    is_str_qubit_id = isinstance(qubit_ids[0], str)
    seq = Sequence(reg, device)
    with pytest.raises(ValueError, match="does not have an SLM mask."):
        seq_ = Sequence(reg, AnalogDevice)
        seq_.config_slm_mask(["q0" if is_str_qubit_id else 0])

    with pytest.raises(TypeError, match="must be castable to set"):
        seq.config_slm_mask(0)
    with pytest.raises(TypeError, match="must be castable to set"):
        seq.config_slm_mask((0))
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask("q0")
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask(["q3" if is_str_qubit_id else 3])
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask(("q3" if is_str_qubit_id else 3,))
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask({"q3" if is_str_qubit_id else 3})
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask([0 if is_str_qubit_id else "0"])
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask((0 if is_str_qubit_id else "0",))
    with pytest.raises(ValueError, match="exist in the register"):
        seq.config_slm_mask({0 if is_str_qubit_id else "0"})

    targets = ["q0" if is_str_qubit_id else 0, "q2" if is_str_qubit_id else 2]
    seq.config_slm_mask(targets)
    if is_str_qubit_id:
        assert seq._slm_mask_targets == {"q0", "q2"}
    else:
        assert seq._slm_mask_targets == {0, 2}
    assert not seq._schedule
    with pytest.raises(ValueError, match="DMM dmm_0 is not available."):
        seq.config_detuning_map(det_map, "dmm_0")
    seq.declare_channel("rydberg_global", "rydberg_global")
    assert set(seq._schedule.keys()) == {"dmm_0", "rydberg_global"}
    assert seq._schedule["dmm_0"].detuning_map.weights[0] == 1.0
    assert seq._schedule["dmm_0"].detuning_map.weights[2] == 1.0

    with pytest.raises(ValueError, match="configured only once"):
        seq.config_slm_mask(targets)
    mapp_reg = MappableRegister(
        RegisterLayout(trap_ids + [(0, 10), (0, 20), (0, -10)]), *qubit_ids
    )
    fail_seq = Sequence(mapp_reg, device)
    with pytest.raises(
        RuntimeError,
        match="The SLM mask can't be combined with a mappable register.",
    ):
        fail_seq.config_slm_mask({trap_ids[0], trap_ids[2]})


def test_slm_mask_in_xy(reg, patch_plt_show):
    reg = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    targets = ["q0", "q2"]
    pulse1 = Pulse.ConstantPulse(100, 10, 0, 0)
    pulse2 = Pulse.ConstantPulse(200, 10, 0, 0)

    # Set mask when an XY pulse is already in the schedule
    seq_xy1 = Sequence(reg, MockDevice)
    seq_xy1.declare_channel("ch_xy", "mw_global")
    seq_xy1.add(pulse1, "ch_xy")
    seq_xy1.add(pulse2, "ch_xy")
    seq_xy1.config_slm_mask(targets)
    assert seq_xy1._slm_mask_time == [0, 100]
    assert "dmm_0" not in seq_xy1._schedule

    # Set mask and then add an XY pulse to the schedule
    seq_xy2 = Sequence(reg, MockDevice)
    seq_xy2.config_slm_mask(targets)
    seq_xy2.declare_channel("ch_xy", "mw_global")
    seq_xy2.add(pulse1, "ch_xy")
    assert seq_xy2._slm_mask_time == [0, 100]
    assert "dmm_0" not in seq_xy2._schedule

    # Check that adding extra pulses does not change SLM mask time
    seq_xy2.add(pulse2, "ch_xy")
    assert seq_xy2._slm_mask_time == [0, 100]

    # Check that SLM mask time is updated accordingly if a new pulse with
    # earlier start is added
    seq_xy3 = Sequence(reg, MockDevice)
    seq_xy3.declare_channel("ch_xy1", "mw_global")
    seq_xy3.config_slm_mask(targets)
    seq_xy3.delay(duration=100, channel="ch_xy1")
    seq_xy3.add(pulse1, "ch_xy1")
    assert seq_xy3._slm_mask_time == [100, 200]
    seq_xy3.declare_channel("ch_xy2", "mw_global")
    seq_xy3.add(pulse1, "ch_xy2", "no-delay")
    assert seq_xy3._slm_mask_time == [0, 100]

    # Same as previous check, but mask is added afterwards
    seq_xy4 = Sequence(reg, MockDevice)
    seq_xy4.declare_channel("ch_xy1", "mw_global")
    seq_xy4.delay(duration=100, channel="ch_xy1")
    seq_xy4.add(pulse1, "ch_xy1")
    seq_xy4.declare_channel("ch_xy2", "mw_global")
    seq_xy4.add(pulse1, "ch_xy2", "no-delay")
    seq_xy4.config_slm_mask(targets)
    assert seq_xy4._slm_mask_time == [0, 100]

    # Check that paramatrize works with SLM mask
    seq_xy5 = Sequence(reg, MockDevice)
    seq_xy5.declare_channel("ch", "mw_global")
    var = seq_xy5.declare_variable("var")
    seq_xy5.add(Pulse.ConstantPulse(200, var, 0, 0), "ch")
    assert seq_xy5.is_parametrized()
    seq_xy5.config_slm_mask(targets)
    seq_xy5_str = seq_xy5._serialize()
    seq_xy5_ = Sequence._deserialize(seq_xy5_str)
    assert str(seq_xy5) == str(seq_xy5_)

    # Check drawing method
    seq_xy2.draw()


@pytest.mark.parametrize("dims3D", [False, True])
@pytest.mark.parametrize("draw_qubit_amp", [True, False])
@pytest.mark.parametrize("draw_qubit_det", [True, False])
@pytest.mark.parametrize("draw_register", [True, False])
@pytest.mark.parametrize("mode", ["input", "input+output"])
@pytest.mark.parametrize("mod_bandwidth", [0, 10])
def test_draw_slm_mask_in_ising(
    patch_plt_show,
    dims3D,
    mode,
    mod_bandwidth,
    draw_qubit_amp,
    draw_qubit_det,
    draw_register,
):
    reg = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    if dims3D:
        reg = Register3D(
            {"q0": (0, 0, 0), "q1": (10, 10, 0), "q2": (-10, -10, 0)}
        )
    det_map = reg.define_detuning_map({"q0": 0.2, "q1": 0.8, "q2": 0.0})
    targets = ["q0", "q2"]
    pulse1 = Pulse.ConstantPulse(100, 10, 0, 0)
    pulse2 = Pulse.ConstantPulse(200, 10, 0, 0)
    mymockdevice = (
        MockDevice
        if mod_bandwidth == 0
        else dataclasses.replace(
            MockDevice,
            dmm_objects=(DMM(mod_bandwidth=mod_bandwidth),),
            channel_objects=(
                dataclasses.replace(
                    MockDevice.channels["rydberg_global"],
                    mod_bandwidth=mod_bandwidth,
                ),
                dataclasses.replace(
                    MockDevice.channels["rydberg_local"],
                    mod_bandwidth=mod_bandwidth,
                ),
                MockDevice.channels["raman_global"],
            ),
            channel_ids=("rydberg_global", "rydberg_local", "raman_global"),
        )
    )
    # Set mask when ising pulses are already in the schedule
    seq1 = Sequence(reg, mymockdevice)
    seq1.declare_channel("ryd_glob", "rydberg_global")
    if not draw_register:
        with patch("matplotlib.figure.Figure.savefig"):
            with pytest.warns(
                UserWarning,
                match="Provide a register and select draw_register",
            ):
                seq1.draw(draw_qubit_det=True, fig_name="empty_rydberg")
    seq1.config_detuning_map(det_map, "dmm_0")
    if mod_bandwidth == 0:
        with pytest.warns() as record:
            seq1.draw(
                draw_qubit_det=True, draw_interp_pts=False, mode="output"
            )  # Drawing Sequence with only a DMM
        assert len(record) == 9
        assert np.all(
            str(record[i].message).startswith(
                "No modulation bandwidth defined"
            )
            for i in range(len(record) - 1)
        )
        assert str(record[-1].message).startswith(
            "Can't display modulated quantities per qubit"
        )
    seq1.draw(mode, draw_qubit_det=draw_qubit_det, draw_interp_pts=False)
    seq1.add_dmm_detuning(RampWaveform(300, -10, 0), "dmm_0")
    # pulse is added on rydberg global with a delay (protocol is "min-delay")
    seq1.add(pulse1, "ryd_glob")  # slm pulse between 0 and 400
    seq1.add(pulse2, "ryd_glob")
    seq1.config_slm_mask(targets)
    mask_time = 400 + 2 * mymockdevice.channels["rydberg_global"].rise_time
    assert seq1._slm_mask_time == [0, mask_time]
    assert seq1._schedule["dmm_0_1"].slots[1].type == Pulse.ConstantPulse(
        mask_time, 0, -100, 0
    )
    # Possible to modulate dmm_0_1 after slm declaration
    seq1.add_dmm_detuning(RampWaveform(300, 0, -10), "dmm_0_1")
    assert seq1._slm_mask_time == [0, mask_time]
    # Possible to add pulses afterwards,
    seq1.declare_channel("ryd_loc", "rydberg_local", ["q0", "q1"])
    seq1.add(pulse2, "ryd_loc", protocol="no-delay")
    assert seq1._slm_mask_time == [0, mask_time]
    with patch("matplotlib.figure.Figure.savefig"):
        seq1.draw(
            mode,
            draw_qubit_det=draw_qubit_det,
            draw_qubit_amp=draw_qubit_amp,
            draw_interp_pts=False,
            draw_register=draw_register,
            fig_name="local_quantities",
        )
    seq1.declare_channel("raman_glob", "raman_global")
    if draw_qubit_det or draw_qubit_amp:
        with pytest.raises(
            NotImplementedError,
            match="Can only draw qubit contents for channels in rydberg basis",
        ):
            seq1.draw(
                mode,
                draw_qubit_det=draw_qubit_det,
                draw_qubit_amp=draw_qubit_amp,
            )


@pytest.mark.parametrize(
    "bottom_detunings", [(None, None), (-20, None), (None, -20), (-20, -20)]
)
def test_slm_mask_in_ising(patch_plt_show, bottom_detunings):
    reg = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    det_map = reg.define_detuning_map({"q0": 0.2, "q1": 0.8, "q2": 0.0})
    targets = ["q0", "q2"]
    amp = 10
    pulse = Pulse.ConstantPulse(200, amp, 0, 0)
    # Set mask and then add ising pulses to the schedule
    seq2 = Sequence(
        reg,
        dataclasses.replace(
            MockDevice,
            dmm_objects=(
                DMM(
                    bottom_detuning=bottom_detunings[0],
                    total_bottom_detuning=bottom_detunings[1],
                ),
            ),
        ),
    )
    seq2.config_slm_mask(targets)
    seq2.declare_channel("ryd_glob", "rydberg_global")
    seq2.config_detuning_map(det_map, "dmm_0")  # configured as dmm_0_1
    with pytest.raises(
        ValueError, match="You should add a Pulse to a Global Channel"
    ):
        seq2.add_dmm_detuning(RampWaveform(300, -10, 0), "dmm_0")
    with pytest.raises(
        ValueError, match="You should add a Pulse to a Global Channel"
    ):
        seq2.add(Pulse.ConstantPulse(300, 0, -10, 0), "dmm_0")
    seq2.add_dmm_detuning(RampWaveform(300, -10, 0), "dmm_0_1")  # not slm
    seq2.add(pulse, "ryd_glob")  # slm pulse between 0 and 500
    assert seq2._slm_mask_time == [0, 500]
    slm_det: float
    if bottom_detunings == (None, None):
        slm_det = -10 * amp
    elif bottom_detunings[0] is None:
        slm_det = max(-10 * amp, bottom_detunings[1] / len(targets))
    elif bottom_detunings[1] is None:
        slm_det = max(-10 * amp, bottom_detunings[0])
    else:
        assert bottom_detunings[1] / len(targets) > bottom_detunings[0]
        slm_det = max(-10 * amp, bottom_detunings[1] / len(targets))
    assert seq2._schedule["dmm_0"].slots[1].type == Pulse.ConstantPulse(
        500, 0, slm_det, 0
    )

    # Check that adding extra pulses does not change SLM mask time
    seq2.add(pulse, "ryd_glob")
    assert seq2._slm_mask_time == [0, 500]

    seq5 = Sequence(reg, MockDevice)
    seq5.declare_channel("ch", "rydberg_global")
    var = seq5.declare_variable("var")
    seq5.add(Pulse.ConstantPulse(200, var, 0, 0), "ch")
    assert seq5.is_parametrized()
    seq5.config_slm_mask(targets)
    seq5_str = seq5._serialize()
    seq5_ = Sequence._deserialize(seq5_str)
    assert str(seq5) == str(seq5_)


@pytest.mark.parametrize("ch_name", ["rydberg_global", "mw_global"])
def test_draw_register_det_maps(reg, ch_name, patch_plt_show):
    # Draw 2d register from sequence
    reg_layout = RegisterLayout(
        [(0, 0), (10, 10), (-10, -10), (20, 20), (30, 30), (40, 40)]
    )
    det_map = reg_layout.define_detuning_map(
        {0: 0, 1: 0, 2: 0, 3: 1.0, 4: 1.0}
    )
    reg = reg_layout.define_register(0, 1, 2, qubit_ids=["q0", "q1", "q2"])
    targets = ["q0", "q2"]
    pulse = Pulse.ConstantPulse(100, 10, 0, 0)
    seq = Sequence(reg, MockDevice)
    seq.declare_channel(ch_name, ch_name)
    seq.add(pulse, ch_name)
    if ch_name == "rydberg_global":
        seq.config_detuning_map(det_map, "dmm_0")
    seq.config_slm_mask(targets)
    seq.draw(draw_register=True)
    seq.draw(draw_detuning_maps=True)
    seq.draw(draw_register=True, draw_detuning_maps=True)

    # Draw 3d register from sequence
    reg3d = Register3D.cubic(3, 8)
    seq3d = Sequence(reg3d, MockDevice)
    seq3d.declare_channel(ch_name, ch_name)
    seq3d.add(pulse, ch_name)
    seq3d.config_slm_mask([6, 15])
    seq3d.measure(basis="XY" if ch_name == "mw_global" else "ground-rydberg")
    seq3d.draw(draw_register=True)
    seq3d.draw(draw_detuning_maps=True)
    seq3d.draw(draw_register=True, draw_detuning_maps=True)


@pytest.mark.parametrize("align_at_rest", [True, False])
def test_hardware_constraints(reg, align_at_rest, patch_plt_show):
    rydberg_global = Rydberg.Global(
        2 * np.pi * 20,
        2 * np.pi * 2.5,
        clock_period=4,
        mod_bandwidth=4,  # MHz
    )

    raman_local = Raman.Local(
        2 * np.pi * 20,
        2 * np.pi * 10,
        min_retarget_interval=220,
        fixed_retarget_t=200,  # ns
        max_targets=1,
        clock_period=4,
        mod_bandwidth=7,  # MHz
    )

    ConstrainedChadoq2 = Device(
        name="ConstrainedChadoq2",
        dimensions=2,
        rydberg_level=70,
        max_atom_num=100,
        max_radial_distance=50,
        min_atom_distance=4,
        channel_objects=(rydberg_global, raman_local),
    )

    seq = Sequence(reg, ConstrainedChadoq2)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "raman_local", initial_target="q1")

    const_pls = Pulse.ConstantPulse(100, 1, 0, np.pi)
    seq.add(const_pls, "ch0")
    black_wf = BlackmanWaveform(500, np.pi)
    black_pls = Pulse.ConstantDetuning(black_wf, 0, 0)
    seq.add(black_pls, "ch1")
    blackman_slot = seq._last("ch1")
    # The pulse accounts for the modulation buffer
    assert (
        blackman_slot.ti == const_pls.duration + rydberg_global.rise_time * 2
    )
    seq.target("q0", "ch1")
    target_slot = seq._last("ch1")
    fall_time = black_pls.fall_time(raman_local)
    assert (
        fall_time
        == raman_local.rise_time + black_wf.modulation_buffers(raman_local)[1]
    )
    fall_time += (
        raman_local.clock_period - fall_time % raman_local.clock_period
    )
    assert target_slot.ti == blackman_slot.tf + fall_time
    assert target_slot.tf == target_slot.ti + raman_local.fixed_retarget_t

    assert raman_local.min_retarget_interval > raman_local.fixed_retarget_t
    seq.target("q2", "ch1")
    assert (
        seq.get_duration("ch1")
        == target_slot.tf + raman_local.min_retarget_interval
    )

    # Check for phase jump buffer
    seq.add(black_pls, "ch0")  # Phase = 0
    tf_ = seq.get_duration("ch0")
    mid_delay = 40
    seq.delay(mid_delay, "ch0")
    seq.add(const_pls, "ch0")  # Phase = π
    interval = seq._schedule["ch0"].adjust_duration(
        rydberg_global.phase_jump_time + black_pls.fall_time(rydberg_global)
    )
    assert seq._schedule["ch0"][-1].ti - tf_ == interval
    added_delay_slot = seq._schedule["ch0"][-2]
    assert added_delay_slot.type == "delay"
    assert added_delay_slot.tf - added_delay_slot.ti == interval - mid_delay

    # Check that there is no phase jump buffer with 'no-delay'
    seq.add(black_pls, "ch0", protocol="no-delay")  # Phase = 0
    assert seq._schedule["ch0"][-1].ti == seq._schedule["ch0"][-2].tf

    tf_ = seq.get_duration("ch0")
    seq.align("ch0", "ch1", at_rest=align_at_rest)
    fall_time = black_pls.fall_time(rydberg_global)
    assert seq.get_duration() == seq._schedule["ch0"].adjust_duration(
        tf_ + fall_time * align_at_rest
    )

    with pytest.raises(ValueError, match="'mode' must be one of"):
        seq.draw(mode="all")

    with pytest.warns(
        UserWarning,
        match="'draw_phase_area' doesn't work in 'output' mode",
    ):
        seq.draw(mode="output", draw_interp_pts=False, draw_phase_area=True)
    with pytest.warns(
        UserWarning,
        match="'draw_interp_pts' doesn't work in 'output' mode",
    ):
        seq.draw(mode="output")
    seq.draw(mode="input+output")


@pytest.mark.parametrize("with_dmm", [False, True])
def test_mappable_register(det_map, patch_plt_show, with_dmm):
    layout = TriangularLatticeLayout(100, 5)
    mapp_reg = layout.make_mappable_register(10)
    seq = Sequence(mapp_reg, DigitalAnalogDevice)
    assert seq.is_register_mappable()
    assert isinstance(seq.get_register(), MappableRegister)
    with pytest.raises(
        RuntimeError, match="Can't access the sequence's register"
    ):
        seq.get_register(include_mappable=False)
    reserved_qids = tuple([f"q{i}" for i in range(10)])
    assert seq._qids == set(reserved_qids)
    with pytest.raises(RuntimeError, match="Can't access the qubit info"):
        seq.qubit_info
    with pytest.raises(
        RuntimeError, match="Can't access the sequence's register"
    ):
        seq.register

    seq.declare_channel("ram", "raman_local", initial_target="q0")
    seq.declare_channel("ryd_loc", "rydberg_local")
    # No Global channel shown, sequence can be printed without warnings
    seq.__str__()
    # Warning if sequence has Global channels and a mappable register
    seq.declare_channel("ryd_glob", "rydberg_global")
    global_channels = ["ryd_glob"]
    if with_dmm:
        seq.config_detuning_map(det_map, "dmm_0")
        global_channels.append("dmm_0")
    warn_message_rydberg = [
        "Showing the register for a sequence with a mappable register."
        + f"Target qubits of channel {ch} will be defined in build."
        for ch in global_channels
    ]
    with pytest.warns(UserWarning) as records:
        seq.__str__()
    assert len(records) == len(global_channels)
    assert [
        str(records[i].message) for i in range(len(global_channels))
    ] == warn_message_rydberg
    # Index of mappable register can be accessed
    seq.phase_shift_index(np.pi / 4, 0, basis="digital")  # 0 -> q0
    seq.target_index(2, "ryd_loc")  # 2 -> q2
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd_glob")
    if with_dmm:
        seq.add_dmm_detuning(RampWaveform(100, -10, 0), "dmm_0")
    seq.add(Pulse.ConstantPulse(200, 1, 0, 0), "ram")
    seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ryd_loc")
    assert seq._last("ryd_glob").targets == set(reserved_qids)
    if with_dmm:
        assert seq._last("dmm_0").targets == set(reserved_qids)
    assert seq._last("ram").targets == {"q0"}
    assert seq._last("ryd_loc").targets == {"q2"}

    with pytest.raises(ValueError, match="Can't draw the register"):
        seq.draw(draw_register=True)

    # Can draw if 'draw_register=False'
    if with_dmm:
        with pytest.raises(
            NotImplementedError,
            match=(
                "Sequences with a DMM channel can't be sampled while "
                "their register is mappable."
            ),
        ):
            seq.draw()
    else:
        seq.draw()
    with pytest.raises(ValueError, match="'qubits' must be specified"):
        seq.build()

    with pytest.raises(
        ValueError, match="targeted but have not been assigned"
    ):
        seq.build(qubits={"q1": 1, "q0": 10})

    with pytest.warns(UserWarning, match="No declared variables named: a"):
        seq.build(qubits={"q2": 20, "q0": 10, "q1": 0}, a=5)

    with pytest.raises(ValueError, match="To declare 3 qubits"):
        seq.build(qubits={"q2": 20, "q0": 10, "q3": 0})

    seq_ = seq.build(qubits={"q2": 20, "q0": 10, "q1": 0})
    assert seq_._last("ryd_glob").targets == {"q0", "q1", "q2"}
    # Check the original sequence is unchanged
    assert seq.is_register_mappable()
    init_call = seq._calls[0]
    assert init_call.name == "__init__"
    assert isinstance(init_call.kwargs["register"], MappableRegister)
    assert not seq_.is_register_mappable()
    assert isinstance(seq_.get_register(), BaseRegister)
    assert isinstance(seq_.get_register(include_mappable=False), BaseRegister)
    assert seq_.register == Register(
        {
            "q0": layout.traps_dict[10],
            "q1": layout.traps_dict[0],
            "q2": layout.traps_dict[20],
        }
    )
    with pytest.raises(ValueError, match="already has a concrete register"):
        seq_.build(qubits={"q2": 20, "q0": 10, "q1": 0})

    # Also possible to build the default register
    with pytest.raises(ValueError, match="'qubits' must be specified"):
        seq.build()


index_function_non_mappable_register_values: Any = [
    (Register(dict(b=[10, 10], c=[5, 5], a=[0, 0])), dict(), 0, "b"),
    (
        TriangularLatticeLayout(100, 5).define_register(
            2, 3, 0, qubit_ids=["a", "b", "c"]
        ),
        dict(),
        2,
        "c",
    ),
    (
        TriangularLatticeLayout(100, 5).define_register(2, 3, 0),
        dict(),
        2,
        "q2",
    ),
]

index_function_mappable_register_values = [
    (
        TriangularLatticeLayout(100, 5).make_mappable_register(10),
        dict(qubits=dict(q0=1, q2=2, q1=0)),
        1,
        "q1",
    ),
]

index_function_params = "register, build_params, index, expected_target"


@pytest.mark.parametrize(
    index_function_params,
    [
        *index_function_non_mappable_register_values,
        *index_function_mappable_register_values,
    ],
)
def test_parametrized_index_functions(
    register, build_params, index, expected_target
):
    phi = np.pi / 4
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("ch0", "rydberg_local")
    seq.declare_channel("ch1", "raman_local")
    index_var = seq.declare_variable("index", dtype=int)
    seq.target_index(index_var, channel="ch0")
    seq.phase_shift_index(phi, index_var)
    built_seq = seq.build(**build_params, index=index)
    assert built_seq._last("ch0").targets == {expected_target}
    assert built_seq.current_phase_ref(expected_target, "digital") == phi

    with pytest.raises(
        IndexError, match="Indices must exist for the register"
    ):
        seq.build(**build_params, index=20)


@pytest.mark.parametrize(
    index_function_params,
    [
        *index_function_non_mappable_register_values,
        *index_function_mappable_register_values,
    ],
)
def test_non_parametrized_index_functions_in_parametrized_context(
    register, build_params, index, expected_target
):
    phi = np.pi / 4
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("ch0", "raman_local")
    phi_var = seq.declare_variable("phi_var", dtype=int)

    seq.phase_shift_index(phi_var, 0)
    seq.target_index(index, channel="ch0")
    seq.phase_shift_index(phi, index)

    built_seq = seq.build(**build_params, phi_var=0)
    assert built_seq._last("ch0").targets == {expected_target}
    assert built_seq.current_phase_ref(expected_target, "digital") == phi


@pytest.mark.parametrize(
    index_function_params, index_function_non_mappable_register_values
)
def test_non_parametrized_non_mappable_register_index_functions(
    register, build_params, index, expected_target
):
    seq = Sequence(register, DigitalAnalogDevice)
    seq.declare_channel("ch0", "rydberg_local")
    seq.declare_channel("ch1", "raman_local")
    phi = np.pi / 4
    with pytest.raises(
        IndexError, match="Indices must exist for the register"
    ):
        seq.target_index(20, channel="ch0")
    with pytest.raises(
        IndexError, match="Indices must exist for the register"
    ):
        seq.phase_shift_index(phi, 20)
    seq.target_index(index, channel="ch0")
    seq.phase_shift_index(phi, index)
    assert seq._last("ch0").targets == {expected_target}
    assert seq.current_phase_ref(expected_target, "digital") == phi


def test_multiple_index_targets(reg):
    test_device = Device(
        name="test_device",
        dimensions=2,
        rydberg_level=70,
        max_atom_num=100,
        max_radial_distance=50,
        min_atom_distance=4,
        channel_objects=(
            Raman.Local(2 * np.pi * 20, 2 * np.pi * 10, max_targets=2),
        ),
    )

    seq = Sequence(reg, test_device)
    var_array = seq.declare_variable("var_array", size=2, dtype=int)
    seq.declare_channel("ch0", "raman_local")

    seq.target_index([0, 1], channel="ch0")
    assert seq._last("ch0").targets == {"q0", "q1"}

    seq.target_index(var_array, channel="ch0")
    built_seq = seq.build(var_array=[1, 2])
    assert built_seq._last("ch0").targets == {"q1", "q2"}

    seq.target_index(var_array + 1, channel="ch0")
    built_seq = seq.build(var_array=[1, 2])
    assert built_seq._last("ch0").targets == {"q2", "q3"}


@pytest.mark.parametrize("check_wait_for_fall", (True, False))
@pytest.mark.parametrize("correct_phase_drift", (True, False))
@pytest.mark.parametrize("custom_buffer_time", (None, 400))
def test_eom_mode(
    reg,
    mod_device,
    custom_buffer_time,
    correct_phase_drift,
    check_wait_for_fall,
    patch_plt_show,
):
    # Setting custom_buffer_time
    channels = mod_device.channels
    eom_config = dataclasses.replace(
        channels["rydberg_global"].eom_config,
        custom_buffer_time=custom_buffer_time,
    )
    channels["rydberg_global"] = dataclasses.replace(
        channels["rydberg_global"], eom_config=eom_config
    )
    dev_ = dataclasses.replace(
        mod_device, channel_ids=None, channel_objects=tuple(channels.values())
    )
    seq = Sequence(reg, dev_)
    seq.declare_channel("ch0", "rydberg_global")
    ch0_obj = seq.declared_channels["ch0"]
    assert not seq.is_in_eom_mode("ch0")

    amp_on = 1.0
    detuning_on = 0.0
    seq.enable_eom_mode("ch0", amp_on, detuning_on, optimal_detuning_off=-100)
    assert seq.is_in_eom_mode("ch0")

    delay_duration = 200
    seq.delay(delay_duration, "ch0")
    detuning_off = seq._schedule["ch0"].eom_blocks[-1].detuning_off
    assert detuning_off != 0

    with pytest.raises(RuntimeError, match="There is no slot with a pulse."):
        # The EOM delay slot (which is a pulse slot) is ignored
        seq._schedule["ch0"].last_pulse_slot(ignore_detuned_delay=True)

    delay_slot = seq._schedule["ch0"][-1]
    assert seq._schedule["ch0"].in_eom_mode(delay_slot)
    assert seq._schedule["ch0"].is_detuned_delay(delay_slot.type)
    assert delay_slot.ti == 0
    assert delay_slot.tf == delay_duration
    assert delay_slot.type == Pulse.ConstantPulse(
        delay_duration, 0.0, detuning_off, 0.0
    )

    assert seq._schedule["ch0"].get_eom_mode_intervals() == [
        (0, delay_slot.tf)
    ]

    pulse_duration = 100
    seq.add_eom_pulse(
        "ch0",
        pulse_duration,
        phase=0.0,
        correct_phase_drift=correct_phase_drift,
    )
    first_pulse_slot = seq._schedule["ch0"].last_pulse_slot()
    assert first_pulse_slot.ti == delay_slot.tf
    assert first_pulse_slot.tf == first_pulse_slot.ti + pulse_duration
    phase_ref = (
        detuning_off * first_pulse_slot.ti * 1e-3 * correct_phase_drift
    ) % (2 * np.pi)
    # The phase correction becomes the new phase reference point
    assert seq.current_phase_ref("q0", basis="ground-rydberg") == phase_ref
    eom_pulse = Pulse.ConstantPulse(
        pulse_duration, amp_on, detuning_on, phase_ref
    )
    assert first_pulse_slot.type == eom_pulse
    assert not seq._schedule["ch0"].is_detuned_delay(eom_pulse)

    # Check phase jump buffer
    phase_ = np.pi
    seq.add_eom_pulse(
        "ch0",
        pulse_duration,
        phase=phase_,
        correct_phase_drift=correct_phase_drift,
    )
    second_pulse_slot = seq._schedule["ch0"].last_pulse_slot()
    phase_buffer = (
        eom_pulse.fall_time(ch0_obj, in_eom_mode=True)
        + seq.declared_channels["ch0"].phase_jump_time
    )
    assert second_pulse_slot.ti == first_pulse_slot.tf + phase_buffer
    # Corrects the phase acquired during the phase buffer
    phase_ref += detuning_off * phase_buffer * 1e-3 * correct_phase_drift
    assert second_pulse_slot.type == Pulse.ConstantPulse(
        pulse_duration, amp_on, detuning_on, phase_ + phase_ref
    )

    # Check phase jump buffer is not enforced with "no-delay"
    seq.add_eom_pulse("ch0", pulse_duration, phase=0.0, protocol="no-delay")
    last_pulse_slot = seq._schedule["ch0"].last_pulse_slot()
    assert last_pulse_slot.ti == second_pulse_slot.tf

    eom_intervals = seq._schedule["ch0"].get_eom_mode_intervals()
    assert eom_intervals == [(0, last_pulse_slot.tf)]

    with pytest.raises(
        RuntimeError, match="The chosen channel is in EOM mode"
    ):
        seq.add(eom_pulse, "ch0")

    assert seq.get_duration() == last_pulse_slot.tf
    assert seq.get_duration(include_fall_time=True) == (
        last_pulse_slot.tf + eom_pulse.fall_time(ch0_obj, in_eom_mode=True)
    )

    seq.disable_eom_mode("ch0")
    assert not seq.is_in_eom_mode("ch0")
    # Check the EOM interval did not change
    assert seq._schedule["ch0"].get_eom_mode_intervals() == eom_intervals
    buffer_delay = seq._schedule["ch0"][-1]
    assert buffer_delay.ti == last_pulse_slot.tf
    assert buffer_delay.tf == buffer_delay.ti + (
        custom_buffer_time or eom_pulse.fall_time(ch0_obj)
    )
    assert buffer_delay.type == "delay"

    assert seq.current_phase_ref("q0", basis="ground-rydberg") == phase_ref
    # Check buffer when EOM is not enabled at the start of the sequence
    interval_time = 0
    if check_wait_for_fall:
        cte_pulse = Pulse.ConstantPulse(100, 1, 0, 0)
        seq.add(cte_pulse, "ch0")
        interval_time = cte_pulse.duration + cte_pulse.fall_time(
            seq.declared_channels["ch0"]
        )
    seq.enable_eom_mode(
        "ch0",
        amp_on,
        detuning_on,
        optimal_detuning_off=-100,
        correct_phase_drift=correct_phase_drift,
    )
    last_slot = seq._schedule["ch0"][-1]
    assert len(seq._schedule["ch0"].eom_blocks) == 2
    new_eom_block = seq._schedule["ch0"].eom_blocks[1]
    assert new_eom_block.detuning_off != 0
    assert last_slot.ti == buffer_delay.tf + interval_time
    duration = last_slot.tf - last_slot.ti
    assert (
        duration == custom_buffer_time
        or 2 * seq.declared_channels["ch0"].rise_time
    )
    # The buffer is a Pulse at 'detuning_off' and zero amplitude
    assert last_slot.type == Pulse.ConstantPulse(
        duration, 0.0, new_eom_block.detuning_off, last_pulse_slot.type.phase
    )
    # Check the phase shift that corrects for the drift
    phase_ref += (
        (new_eom_block.detuning_off * duration * 1e-3)
        % (2 * np.pi)
        * correct_phase_drift
    )
    assert np.isclose(
        seq.current_phase_ref("q0", basis="ground-rydberg"),
        float(phase_ref) % (2 * np.pi),
    )

    # Add delay to test the phase drift correction in disable_eom_mode
    last_delay_time = 400
    seq.delay(last_delay_time, "ch0")

    seq.disable_eom_mode("ch0", correct_phase_drift=True)
    phase_ref += new_eom_block.detuning_off * last_delay_time * 1e-3
    assert np.isclose(
        seq.current_phase_ref("q0", basis="ground-rydberg"),
        float(phase_ref) % (2 * np.pi),
    )

    # Test drawing in eom mode
    seq.draw()


@pytest.mark.parametrize(
    "initial_instruction, non_zero_detuning_off",
    list(itertools.product([None, "delay", "add"], [True, False])),
)
def test_eom_buffer(
    reg, mod_device, initial_instruction, non_zero_detuning_off
):
    seq = Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_local", initial_target="q0")
    seq.declare_channel("other", "rydberg_global")
    if initial_instruction == "delay":
        seq.delay(16, "ch0")
        phase = 0
    elif initial_instruction == "add":
        phase = np.pi
        seq.add(Pulse.ConstantPulse(16, 1, 0, np.pi), "ch0")
    eom_block_starts = seq.get_duration(include_fall_time=True)
    # Adjust the moment the EOM block starts to the clock period
    eom_block_starts = seq._schedule["ch0"].adjust_duration(eom_block_starts)

    eom_config = seq.declared_channels["ch0"].eom_config
    limit_rabi_freq = eom_config.max_limiting_amp**2 / (
        2 * eom_config.intermediate_detuning
    )
    amp_on = limit_rabi_freq * (1.1 if non_zero_detuning_off else 0.5)

    # Show that EOM mode ignores other channels and uses "no-delay" by default
    seq.add(Pulse.ConstantPulse(100, 1, -1, 0), "other")
    seq.enable_eom_mode("ch0", amp_on, 0)
    assert len(seq._schedule["ch0"].eom_blocks) == 1
    eom_block = seq._schedule["ch0"].eom_blocks[0]
    if non_zero_detuning_off:
        assert eom_block.detuning_off != 0
    else:
        assert eom_block.detuning_off == 0
    if not initial_instruction:
        assert seq.get_duration(channel="ch0") == 0  # Channel remains empty
    else:
        last_slot = seq._schedule["ch0"][-1]
        assert last_slot.ti == eom_block_starts  # Nothing else was added
        duration = last_slot.tf - last_slot.ti
        # The buffer is a Pulse at 'detuning_off' and zero amplitude
        assert (
            last_slot.type
            == Pulse.ConstantPulse(
                duration, 0.0, eom_block.detuning_off, phase
            )
            if non_zero_detuning_off
            else "delay"
        )


@pytest.mark.parametrize("correct_phase_drift", [True, False])
@pytest.mark.parametrize("amp_diff", [0, -0.5, 0.5])
@pytest.mark.parametrize("det_diff", [0, -5, 10])
def test_modify_eom_setpoint(
    reg, mod_device, amp_diff, det_diff, correct_phase_drift
):
    seq = Sequence(reg, mod_device)
    seq.declare_channel("ryd", "rydberg_global")
    params = seq.declare_variable("params", dtype=float, size=2)
    dt = 100
    amp, det_on = params
    with pytest.raises(
        RuntimeError, match="The 'ryd' channel is not in EOM mode"
    ):
        seq.modify_eom_setpoint("ryd", amp, det_on)
    seq.enable_eom_mode("ryd", amp, det_on)
    assert seq.is_in_eom_mode("ryd")
    seq.add_eom_pulse("ryd", dt, 0.0)
    seq.delay(dt, "ryd")

    new_amp, new_det_on = amp + amp_diff, det_on + det_diff
    seq.modify_eom_setpoint(
        "ryd", new_amp, new_det_on, correct_phase_drift=correct_phase_drift
    )
    assert seq.is_in_eom_mode("ryd")
    seq.add_eom_pulse("ryd", dt, 0.0)
    seq.delay(dt, "ryd")

    ryd_ch_obj = seq.declared_channels["ryd"]
    eom_buffer_dt = ryd_ch_obj._eom_buffer_time
    param_vals = [1.0, 0.0]
    built_seq = seq.build(params=param_vals)
    expected_duration = 4 * dt + eom_buffer_dt
    assert built_seq.get_duration() == expected_duration

    amp, det = param_vals
    ch_samples = sample(built_seq).channel_samples["ryd"]
    expected_amp = np.zeros(expected_duration)
    expected_amp[:dt] = amp
    expected_amp[-2 * dt : -dt] = amp + amp_diff
    np.testing.assert_array_equal(expected_amp, ch_samples.amp)

    det_off = ryd_ch_obj.eom_config.calculate_detuning_off(amp, det, 0.0)
    new_det_off = ryd_ch_obj.eom_config.calculate_detuning_off(
        amp + amp_diff, det + det_diff, 0.0
    )
    expected_det = np.zeros(expected_duration)
    expected_det[:dt] = det
    expected_det[dt : 2 * dt] = det_off
    expected_det[2 * dt : 2 * dt + eom_buffer_dt] = new_det_off
    expected_det[-2 * dt : -dt] = det + det_diff
    expected_det[-dt:] = new_det_off
    np.testing.assert_array_equal(expected_det, ch_samples.det)

    final_phase = built_seq.current_phase_ref("q0", "ground-rydberg")
    if not correct_phase_drift:
        assert final_phase == 0.0
    else:
        assert final_phase != 0.0
    np.testing.assert_array_equal(ch_samples.phase[: 2 * dt], 0.0)
    np.testing.assert_array_equal(ch_samples.phase[-2 * dt :], final_phase)


def test_max_duration(reg, mod_device):
    dev_ = dataclasses.replace(mod_device, max_sequence_duration=100)
    seq = Sequence(reg, dev_)
    seq.declare_channel("ch0", "rydberg_global")
    seq.delay(100, "ch0")
    catch_statement = pytest.raises(
        RuntimeError, match="duration exceeded the maximum duration allowed"
    )
    with catch_statement:
        seq.delay(16, "ch0")
    with catch_statement:
        seq.add(Pulse.ConstantPulse(100, 1, 0, 0), "ch0")


def test_add_to_dmm_fails(reg, device, det_map):
    seq = Sequence(reg, device)
    seq.config_detuning_map(det_map, "dmm_0")
    pulse = Pulse.ConstantPulse(100, 0, -1, 0)
    with pytest.raises(ValueError, match="can't be used on a DMM"):
        seq.add(pulse, "dmm_0")

    seq.declare_channel("ryd", "rydberg_global")
    with pytest.raises(ValueError, match="not the name of a DMM channel"):
        seq.add_dmm_detuning(pulse.detuning, "ryd")


@pytest.mark.parametrize(
    "with_eom, with_modulation", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("parametrized", [True, False])
def test_sequence_diff(device, parametrized, with_modulation, with_eom):
    torch = pytest.importorskip("torch")
    reg = Register(
        {"q0": torch.tensor([0.0, 0.0], requires_grad=True), "q1": (-5.0, 5.0)}
    )
    seq = Sequence(reg, AnalogDevice if with_eom else device)
    seq.declare_channel("ryd_global", "rydberg_global")

    if parametrized:
        amp = seq.declare_variable("amp", dtype=float)
        dets = seq.declare_variable("dets", dtype=float, size=2)
    else:
        amp = torch.tensor(1.0, requires_grad=True)
        dets = torch.tensor([-2.0, -1.0], requires_grad=True)

    # The phase is never a variable so we're sure the gradient
    # is kept after build
    phase = torch.tensor(2.0, requires_grad=True)

    if with_eom:
        seq.enable_eom_mode("ryd_global", amp, dets[0], dets[1])
        seq.add_eom_pulse("ryd_global", 100, phase, correct_phase_drift=False)
        seq.delay(100, "ryd_global")
        seq.modify_eom_setpoint("ryd_global", amp * 2, dets[1], -dets[0])
        seq.add_eom_pulse("ryd_global", 100, -phase, correct_phase_drift=True)
        seq.disable_eom_mode("ryd_global")

    else:
        pulse = Pulse.ConstantDetuning(
            BlackmanWaveform(1000, amp), dets[0], phase
        )
        seq.add(pulse, "ryd_global")
        det_map = reg.define_detuning_map({"q0": 1.0})
        seq.config_detuning_map(det_map, "dmm_0")
        seq.add_dmm_detuning(RampWaveform(2000, *dets), "dmm_0")

    if parametrized:
        seq = seq.build(
            amp=torch.tensor(1.0, requires_grad=True),
            dets=torch.tensor([-2.0, -1.0], requires_grad=True),
        )

    seq_samples = sample(seq, modulation=with_modulation)
    ryd_ch_samples = seq_samples.channel_samples["ryd_global"]
    assert ryd_ch_samples.amp.is_differentiable
    assert ryd_ch_samples.det.is_differentiable
    assert ryd_ch_samples.phase.is_differentiable
    if "dmm_0" in seq_samples.channel_samples:
        dmm_ch_samples = seq_samples.channel_samples["dmm_0"]
        # Only detuning is modulated
        assert not dmm_ch_samples.amp.is_differentiable
        assert dmm_ch_samples.det.is_differentiable
        assert not dmm_ch_samples.phase.is_differentiable
