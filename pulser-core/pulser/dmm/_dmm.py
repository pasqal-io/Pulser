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

from dataclasses import dataclass, field
from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from pulser.channels.base_channel import Channel
from pulser.register._reg_drawer import RegDrawer


@dataclass
class DetuningMap(RegDrawer):
    """Defines a DetuningMap.

    A DetuningMap associates a detuning weight to the coordinates of a trap.
    The sum of the provided weights must be equal to 1.

    Args:
        trap_coordinates: an array containing the coordinates of the traps.
        detuning_weights: A list of detuning weights to associate to the traps.
    """

    trap_coordinates: ArrayLike
    detuning_weights: list[float]

    def __post_init__(self) -> None:
        if len(cast(list, self.trap_coordinates)) != len(
            self.detuning_weights
        ):
            raise ValueError("Number of traps and weights don't match.")
        if not np.all(np.array(self.detuning_weights) >= 0):
            raise ValueError("detuning weights should be positive.")
        if not np.isclose(sum(self.detuning_weights), 1.0, atol=1e-16):
            raise ValueError("The sum of weights should be 1.")

    def draw(
        self,
        with_labels: bool = True,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
        custom_ax: Optional[Axes] = None,
        show: bool = True,
    ) -> None:
        """Draws the detuning map.

        Args:
            with_labels: If True, writes the qubit ID's
                next to each qubit.
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.
            custom_ax: If present, instead of creating its own Axes object,
                the function will use the provided one. Warning: if fig_name
                is set, it may save content beyond what is drawn in this
                function.
            show: Whether or not to call `plt.show()` before returning. When
                combining this plot with other ones in a single figure, one may
                need to set this flag to False.
        """
        pos = np.array(self.trap_coordinates)
        if custom_ax is None:
            _, custom_ax = self._initialize_fig_axes(pos)

        super()._draw_2D(
            custom_ax,
            pos,
            [i for i, _ in enumerate(cast(list, self.trap_coordinates))],
            with_labels=with_labels,
            dmm_qubits=dict(enumerate(self.detuning_weights)),
        )

        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)

        if show:
            plt.show()


@dataclass(init=True, repr=False, frozen=True)
class DMM(Channel):
    """Defines a Detuning Map Modulator (DMM) Channel.

    A Detuning Map Modulator can be used to define `Global` detuning Pulses
    (of zero amplitude and phase). These Pulses are locally modulated by the
    weights of a `DetuningMap`, thus providing a local control over the
    detuning. The detuning of the pulses added to a DMM has to be negative,
    between 0 and `bottom_detuning`. Channel targeting the transition between
    the ground and rydberg states, thus enconding the 'ground-rydberg' basis.

    Note: The protocol to add pulses to the DMM Channel is by default
    "no-delay".

    Args:
        bottom_detuning: Minimum possible detuning (in rad/µs), must be below
            zero.
        clock_period: The duration of a clock cycle (in ns). The duration of a
            pulse or delay instruction is enforced to be a multiple of the
            clock cycle.
        min_duration: The shortest duration an instruction can take.
        max_duration: The longest duration an instruction can take.
        mod_bandwidth: The modulation bandwidth at -3dB (50% reduction), in
            MHz.
    """

    bottom_detuning: Optional[float] = field(default=None, init=True)
    addressing: Literal["Global"] = field(default="Global", init=False)
    max_abs_detuning: Optional[float] = field(init=False, default=None)
    max_amp: float = field(default=0, init=False)
    min_retarget_interval: Optional[int] = field(init=False, default=None)
    fixed_retarget_t: Optional[int] = field(init=False, default=None)
    max_targets: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bottom_detuning and self.bottom_detuning > 0:
            raise ValueError("bottom_detuning must be negative.")

    @property
    def basis(self) -> Literal["ground-rydberg"]:
        """The addressed basis name."""
        return "ground-rydberg"
