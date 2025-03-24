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

import re
from types import TracebackType
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pulser.channels import Raman, Rydberg
from pulser.channels.dmm import DMM
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices import Device


@pytest.fixture
def mod_device() -> Device:
    return Device(
        name="ModDevice",
        dimensions=3,
        rydberg_level=70,
        max_atom_num=2000,
        max_radial_distance=1000,
        min_atom_distance=1,
        supports_slm_mask=True,
        channel_objects=(
            Rydberg.Global(
                1000,
                200,
                clock_period=1,
                min_duration=1,
                mod_bandwidth=4.0,  # MHz
                eom_config=RydbergEOM(
                    mod_bandwidth=30.0,
                    limiting_beam=RydbergBeam.RED,
                    max_limiting_amp=50 * 2 * np.pi,
                    intermediate_detuning=800 * 2 * np.pi,
                    controlled_beams=(RydbergBeam.BLUE,),
                ),
            ),
            Rydberg.Local(
                2 * np.pi * 20,
                2 * np.pi * 10,
                max_targets=2,
                fixed_retarget_t=0,
                clock_period=4,
                min_retarget_interval=220,
                mod_bandwidth=4.0,
                eom_config=RydbergEOM(
                    mod_bandwidth=20.0,
                    limiting_beam=RydbergBeam.RED,
                    max_limiting_amp=60 * 2 * np.pi,
                    intermediate_detuning=700 * 2 * np.pi,
                    controlled_beams=tuple(RydbergBeam),
                ),
            ),
            Raman.Local(
                2 * np.pi * 20,
                2 * np.pi * 10,
                max_targets=2,
                fixed_retarget_t=0,
                min_retarget_interval=220,
                clock_period=4,
                mod_bandwidth=4.0,
            ),
        ),
        dmm_objects=(
            DMM(bottom_detuning=-100, total_bottom_detuning=-10000),
            DMM(
                clock_period=4,
                mod_bandwidth=4.0,
                bottom_detuning=-50,
                total_bottom_detuning=-5000,
            ),
        ),
    )


@pytest.fixture()
def patch_plt_show(monkeypatch):
    # Close residual figures
    plt.close("all")
    # Closes a figure instead of showing it
    monkeypatch.setattr(plt, "show", plt.close)


@pytest.fixture()
def catch_phase_shift_warning():
    return pytest.warns(
        UserWarning,
        match="In version v1.4.0 the behavior of `Sequence.phase_shift`",
    )


class _RaisesAllContext:
    """Utility: check exceptions raised by a block.

    This is a variant of `pytest.raises` that checks that a block
    raises an exception *and* that this exception is an instance
    of all the classes listed.

    The main use of this class is to check that a block raises an
    exception that both maintains backwards compatibility (e.g.
    a `ValueError` with a given message) *and* matches a new
    exception style (e.g. a `DimensionsError`).
    """

    def __init__(
        self, expected: list[type[Exception]], match: Optional[str] = None
    ):
        self.expected = expected
        if match is None:
            self.match = None
        else:
            self.match = re.compile(match)

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is None:
            pytest.fail(f"DID NOT RAISE ANY OF {self.expected}")
        for expected_type in self.expected:
            if not issubclass(exc_type, expected_type):
                pytest.fail(
                    f"exception has type {exc_type},"
                    f"expected a subclass of {expected_type}"
                )
        if self.match is not None:
            assert exc_val is not None
            message = str(exc_val)
            if self.match.search(message) is None:
                pytest.fail(
                    reason=f"exception '{message}' did not match"
                    f" '{self.match.pattern}'"
                )
        return True


class Helpers:
    """Testing helpers."""

    @staticmethod
    def raises_all(
        expected: list[type[Exception]], match: str
    ) -> _RaisesAllContext:
        """Utility: check exceptions raised by a block.

        This is a variant of `pytest.raises` that checks that a block
        raises an exception *and* that this exception is an instance
        of all the classes listed.

        The main use of this class is to check that a block raises an
        exception that both maintains backwards compatibility (e.g.
        a `ValueError` with a given message) *and* matches a new
        exception style (e.g. a `DimensionsError`).
        """
        return _RaisesAllContext(expected, match)


@pytest.fixture
def helpers() -> type[Helpers]:
    """Testing helpers."""
    return Helpers
