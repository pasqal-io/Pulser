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

from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from pulser import Register
from pulser.channels.dmm import DMM
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register_layout import RegisterLayout
from pulser.register.weight_maps import DetuningMap


@pytest.fixture
def trap_coordinates():
    return [[0, 0], [1, 0], [0, 1], [1, 1]]


@pytest.fixture
def layout(trap_coordinates) -> RegisterLayout:
    return RegisterLayout(trap_coordinates)


@pytest.fixture
def register(layout: RegisterLayout) -> Register:
    return layout.define_register(0, 1, 2, 3, qubit_ids=(0, 1, 2, 3))


@pytest.fixture
def map_reg(layout: RegisterLayout) -> MappableRegister:
    return layout.make_mappable_register(4)


def test_init(
    layout: RegisterLayout, register: Register, map_reg: MappableRegister
):
    str_key = {"1": 1.0}
    wrong_key = {4: 1.0}
    bad_weights = {0: -1.0, 1: 1.0, 2: 1.0}
    bad_sum = {0: 0.1, 2: 0.9, 3: 0.1}
    det_map = {0: 0.7, 1: 0.3, 2: 0}
    slm_map = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    for bad_key in (str_key, wrong_key):
        print(bad_key)
        for reg in (layout, map_reg):
            print(reg)
            with pytest.raises(
                ValueError,
                match=(
                    "The trap ids of detuning weights have to be integers"
                    " between 0 and 4"
                ),
            ):
                reg.define_detuning_map(bad_key)
        with pytest.raises(
            ValueError,
            match=(
                "The qubit ids of detuning weights have to be defined in the"
                " register."
            ),
        ):
            register.define_detuning_map(bad_key)
    with pytest.raises(
        ValueError, match="Number of traps and weights don't match."
    ):
        DetuningMap([(0, 0), (1, 0)], [0])
    for reg in (layout, map_reg, register):
        with pytest.raises(
            ValueError, match="All weights must be non-negative."
        ):
            reg.define_detuning_map(bad_weights)
        with pytest.raises(
            ValueError, match="The sum of the weights should be 1."
        ):
            reg.define_detuning_map(bad_sum)
        for map in (det_map, slm_map):
            detuning_map = reg.define_detuning_map(map)
            assert np.all(
                map[i] == detuning_map.weights[i] for i in range(len(map))
            )
            assert np.all(
                layout.coords[i] == detuning_map.trap_coordinates[i]
                for i in range(len(map))
            )


@pytest.fixture
def det_map(layout: RegisterLayout) -> DetuningMap:
    return layout.define_detuning_map({0: 0.7, 1: 0.3, 2: 0})


@pytest.fixture
def slm_map(layout: RegisterLayout) -> DetuningMap:
    return layout.define_detuning_map({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def test_draw(det_map, slm_map, patch_plt_show):
    for detuning_map in (det_map, slm_map):
        detuning_map.draw(with_labels=True, show=True, custom_ax=None)
        with patch("matplotlib.pyplot.savefig"):
            detuning_map.draw(fig_name="det_map.pdf")
    with pytest.raises(
        ValueError, match="masked qubits and dmm qubits must be the same."
    ):
        slm_map._draw_2D(
            slm_map._initialize_fig_axes(np.array(slm_map.trap_coordinates))[
                1
            ],
            np.array(slm_map.trap_coordinates),
            [i for i, _ in enumerate(cast(list, slm_map.trap_coordinates))],
            with_labels=True,
            dmm_qubits=dict(enumerate(slm_map.weights)),
            masked_qubits={
                1,
            },
        )


def test_DMM():
    dmm = DMM(
        bottom_detuning=-1,
        clock_period=1,
        min_duration=1,
        max_duration=1e6,
        mod_bandwidth=20,
    )
    assert dmm.basis == "ground-rydberg"
    assert dmm.addressing == "Global"
    assert dmm.bottom_detuning == -1
    assert dmm.max_amp == 1e-16
    for value in (
        dmm.max_abs_detuning,
        dmm.min_retarget_interval,
        dmm.fixed_retarget_t,
        dmm.max_targets,
    ):
        assert value is None
    with pytest.raises(ValueError, match="bottom_detuning must be negative."):
        DMM(bottom_detuning=1)
    assert dmm._has_fixed_addressing
    with pytest.raises(
        NotImplementedError,
        match=f"{DMM} cannot be initialized from `Global` method.",
    ):
        DMM.Global(None, None, bottom_detuning=1)
    with pytest.raises(
        NotImplementedError,
        match=f"{DMM} cannot be initialized from `Local` method.",
    ):
        DMM.Local(None, None, bottom_detuning=1)