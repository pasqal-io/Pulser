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

import re
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from pulser.channels.dmm import DMM
from pulser.pulse import Pulse
from pulser.register.base_register import BaseRegister
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register_layout import RegisterLayout
from pulser.register.weight_maps import DetuningMap, WeightMap


class TestDetuningMap:
    @pytest.fixture
    def layout(self) -> RegisterLayout:
        return RegisterLayout([[0, 0], [1, 0], [0, 1], [1, 1]])

    @pytest.fixture
    def register(self, layout: RegisterLayout) -> BaseRegister:
        return layout.define_register(0, 1, 2, 3, qubit_ids=(0, 1, 2, 3))

    @pytest.fixture
    def map_reg(self, layout: RegisterLayout) -> MappableRegister:
        return layout.make_mappable_register(4)

    @pytest.fixture
    def det_dict(self) -> dict[int, float]:
        return {0: 0.7, 1: 0.3, 2: 0}

    @pytest.fixture
    def det_map(
        self, layout: RegisterLayout, det_dict: dict[int, float]
    ) -> DetuningMap:
        return layout.define_detuning_map(det_dict)

    @pytest.fixture
    def slm_dict(self) -> dict[int, float]:
        return {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}

    @pytest.fixture
    def slm_map(
        self, layout: RegisterLayout, slm_dict: dict[int, float]
    ) -> DetuningMap:
        return layout.define_detuning_map(slm_dict)

    @pytest.mark.parametrize("bad_key", [{"1": 1.0}, {4: 1.0}])
    def test_define_detuning_map(
        self,
        layout: RegisterLayout,
        register: BaseRegister,
        map_reg: MappableRegister,
        bad_key: dict,
    ):
        for reg in (layout, map_reg):
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "The trap ids of detuning weights have to be integers"
                    " in [0, 3]."
                ),
            ):
                reg.define_detuning_map(bad_key)  # type: ignore
        with pytest.raises(
            ValueError,
            match=(
                "The qubit ids linked to detuning weights have to be defined"
                " in the register."
            ),
        ):
            register.define_detuning_map(bad_key)

    def test_qubit_weight_map(self, register):
        # Purposefully unsorted
        qid_weight_map = {1: 0.5, 0: 0.1, 3: 0.4}
        sorted_qids = sorted(qid_weight_map)
        det_map = register.define_detuning_map(qid_weight_map)
        qubits = register.qubits
        coords = [qubits[qid] for qid in sorted_qids]
        weights = [qid_weight_map[qid] for qid in sorted_qids]

        np.testing.assert_equal(det_map.sorted_coords, coords)
        np.testing.assert_equal(det_map.sorted_weights, weights)

        # We recover the original qid_weight_map (and undefined qids show as 0)
        assert det_map.get_qubit_weight_map(qubits) == {
            **qid_weight_map,
            2: 0.0,
        }

    def test_hash(self, det_map, det_dict, layout):
        disordered_det_dict = {
            i: det_dict[i] for i in sorted(det_dict, reverse=True)
        }
        assert disordered_det_dict == det_dict
        assert list(disordered_det_dict) != list(det_dict)

        det_map2 = layout.define_detuning_map(disordered_det_dict)

        # The maps differ in the arguments order
        assert np.any(det_map.trap_coordinates != det_map2.trap_coordinates)
        assert det_map.weights != det_map2.weights

        # But are equal in sorted content
        np.testing.assert_equal(det_map.sorted_coords, det_map2.sorted_coords)
        np.testing.assert_equal(
            det_map.sorted_weights, det_map2.sorted_weights
        )

        # And they have the same type, so they should be equal
        assert type(det_map) == type(det_map2)
        assert det_map == det_map2

        # This means their static hashes and reprs match
        static_hash = det_map.static_hash()
        assert static_hash == det_map2.static_hash()
        assert repr(det_map) == repr(det_map2) == f"DetuningMap_{static_hash}"

        # However, if the types don't match, this should no longer hold
        w_map = WeightMap(det_map.trap_coordinates, det_map.weights)

        # Content is still the same
        np.testing.assert_equal(det_map.sorted_coords, w_map.sorted_coords)
        np.testing.assert_equal(det_map.sorted_weights, w_map.sorted_weights)

        # But the rest isn't
        assert static_hash != w_map.static_hash()
        assert repr(w_map) != repr(det_map)
        assert repr(w_map) == f"WeightMap_{w_map.static_hash()}"
        assert w_map != det_map

    def test_detuning_map_bad_init(
        self,
        layout: RegisterLayout,
        register: BaseRegister,
        map_reg: MappableRegister,
    ):
        with pytest.raises(
            ValueError, match="Number of traps and weights don't match."
        ):
            DetuningMap([(0, 0), (1, 0)], [0])

        bad_weights = {0: -1.0, 1: 1.0, 2: 1.0}
        bad_sum = {0: 0.1, 2: 0.9, 3: 0.1}
        for reg in (layout, map_reg, register):
            with pytest.raises(
                ValueError, match="All weights must be non-negative."
            ):
                reg.define_detuning_map(bad_weights)  # type: ignore
            with pytest.raises(
                ValueError, match="The sum of the weights should be 1."
            ):
                reg.define_detuning_map(bad_sum)  # type: ignore

    def test_init(
        self,
        layout: RegisterLayout,
        register: BaseRegister,
        map_reg: MappableRegister,
        det_dict: dict[int, float],
        slm_dict: dict[int, float],
    ):
        for reg in (layout, map_reg, register):
            for detuning_map_dict in (det_dict, slm_dict):
                detuning_map = cast(
                    DetuningMap,
                    reg.define_detuning_map(detuning_map_dict),  # type: ignore
                )
                assert np.all(
                    [
                        detuning_map_dict[i] == detuning_map.weights[i]
                        for i in range(len(detuning_map_dict))
                    ]
                )
                assert np.all(
                    [
                        layout.coords[i]
                        == np.array(detuning_map.trap_coordinates)[i]
                        for i in range(len(detuning_map_dict))
                    ]
                )

    @pytest.mark.parametrize("with_labels", [False, True])
    def test_draw(self, det_map, slm_map, patch_plt_show, with_labels):
        for detuning_map in (det_map, slm_map):
            labels = (
                list(range(detuning_map.number_of_traps))
                if with_labels
                else None
            )
            detuning_map.draw(labels=labels, show=True, custom_ax=None)
            with patch("matplotlib.pyplot.savefig"):
                detuning_map.draw(fig_name="det_map.pdf")
        with pytest.raises(
            ValueError, match="masked qubits and dmm qubits must be the same."
        ):
            slm_map._draw_2D(
                slm_map._initialize_fig_axes(
                    np.array(slm_map.trap_coordinates)
                )[1],
                np.array(slm_map.trap_coordinates),
                [
                    i
                    for i, _ in enumerate(cast(list, slm_map.trap_coordinates))
                ],
                with_labels=True,
                dmm_qubits=dict(enumerate(slm_map.weights)),
                masked_qubits={
                    1,
                },
            )


class TestDMM:
    @pytest.fixture
    def physical_dmm(self):
        return DMM(
            bottom_detuning=-1,
            clock_period=1,
            min_duration=1,
            max_duration=1e6,
            mod_bandwidth=20,
        )

    def test_init(self, physical_dmm):
        assert DMM().is_virtual()

        dmm = physical_dmm
        assert not dmm.is_virtual()
        assert dmm.basis == "ground-rydberg"
        assert dmm.addressing == "Global"
        assert dmm.bottom_detuning == -1
        assert dmm.max_amp == 0
        for value in (
            dmm.max_abs_detuning,
            dmm.min_retarget_interval,
            dmm.fixed_retarget_t,
            dmm.max_targets,
        ):
            assert value is None
        with pytest.raises(
            ValueError, match="bottom_detuning must be negative."
        ):
            DMM(bottom_detuning=1)
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

    def test_validate_pulse(self, physical_dmm):
        pos_det_pulse = Pulse.ConstantPulse(100, 0, 1e-3, 0)
        with pytest.raises(
            ValueError, match="The detuning in a DMM must not be positive."
        ):
            physical_dmm.validate_pulse(pos_det_pulse)

        too_low_pulse = Pulse.ConstantPulse(
            100, 0, physical_dmm.bottom_detuning - 0.01, 0
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The detuning goes below the bottom detuning "
                f"of the DMM ({physical_dmm.bottom_detuning} rad/µs)"
            ),
        ):
            physical_dmm.validate_pulse(too_low_pulse)

        # Should be valid in a virtual DMM
        virtual_dmm = DMM()
        assert virtual_dmm.is_virtual()
        virtual_dmm.validate_pulse(too_low_pulse)
