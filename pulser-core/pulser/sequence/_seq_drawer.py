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

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline

import pulser
import pulser.math as pm
from pulser import Register, Register3D
from pulser.channels.base_channel import Channel
from pulser.channels.dmm import DMM
from pulser.pulse import Pulse
from pulser.register.base_register import BaseRegister
from pulser.register.weight_maps import DetuningMap
from pulser.sampler.sampler import sample
from pulser.sampler.samples import ChannelSamples, DMMSamples, SequenceSamples
from pulser.waveforms import InterpolatedWaveform

# Color scheme
COLORS = ["darkgreen", "indigo", "#c75000"]

CURVES_ORDER = ("amplitude", "detuning", "phase")

SIZE_PER_WIDTH = {1: 3, 2: 4, 3: 5}
LABELS = [
    r"$\Omega$ (rad/µs)",
    r"$\delta$ (rad/µs)",
    r"$\varphi$ / 2π",
]


class EOMSegment:
    """The class to mark an EOM slot."""

    def __init__(self, ti: int | None = None, tf: int | None = None) -> None:
        """Class is defined from its start and end value."""
        self.ti = ti
        self.tf = tf
        self.color = "steelblue"
        self.alpha = 0.3

    @property
    def isempty(self) -> bool:
        """Defines if the class is empty."""
        return self.ti is None or self.tf is None

    @property
    def nvspan(self) -> int:
        """Defines the number of points in the slot."""
        return cast(int, self.tf) - cast(int, self.ti)

    def draw(self, ax: Axes) -> None:
        """Draws a rectangle between the start and end value."""
        if not self.isempty:
            ax.axvspan(
                self.ti,
                self.tf,
                color=self.color,
                alpha=self.alpha,
                zorder=-100,
            )

    def smooth_draw(self, ax: Axes, decreasing: bool = False) -> None:
        """Draws a rectangle with an increasing/decreasing opacity."""
        if not self.isempty:
            for i in range(self.nvspan):
                ax.axvspan(
                    cast(int, self.ti) + i,
                    cast(int, self.ti) + i + 1,
                    facecolor=self.color,
                    alpha=self.alpha
                    * (
                        decreasing + (-1) ** decreasing * (i + 1) / self.nvspan
                    ),
                    zorder=-100,
                )
            ax.axvline(
                self.tf if decreasing else self.ti,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color=self.color,
                alpha=self.alpha / 2.0,
            )


@dataclass
class ChannelDrawContent:
    """The contents for drawing a single channel."""

    samples: ChannelSamples
    target: dict[Union[str, tuple[int, int]], Any]
    eom_intervals: list[EOMSegment]
    eom_start_buffers: list[EOMSegment]
    eom_end_buffers: list[EOMSegment]
    interp_pts: dict[str, list[list[float]]] = field(default_factory=dict)
    phase_modulated: bool = False

    def __post_init__(self) -> None:
        # Make sure there are no tensors in the channel samples
        self.samples.amp = pm.AbstractArray(
            self.samples.amp.as_array(detach=True)
        )
        self.samples.det = pm.AbstractArray(
            self.samples.det.as_array(detach=True)
        )
        self.samples.phase = pm.AbstractArray(
            self.samples.phase.as_array(detach=True)
        )
        if self.samples._centered_phase is not None:
            self.samples._centered_phase = pm.AbstractArray(
                self.samples._centered_phase.as_array(detach=True)
            )

        is_dmm = isinstance(self.samples, DMMSamples)
        self.curves_on = {
            "amplitude": not is_dmm,
            "detuning": is_dmm,
            "phase": False,
        }

    @property
    def _samples_from_curves(self) -> dict[str, str]:
        return {
            "amplitude": "amp",
            "detuning": "det",
            "phase": ("phase_modulation" if self.phase_modulated else "phase"),
        }

    @property
    def n_axes_on(self) -> int:
        """The number of axes to draw for this channel."""
        return sum(self.curves_on.values())

    def get_input_curves(self) -> list[np.ndarray]:
        """The samples for the curves, as programmed."""
        return self._give_curves_from_samples(self.samples)

    def get_output_curves(self, ch_obj: Channel) -> list[np.ndarray]:
        """The modulated samples for the curves."""
        mod_samples = self.samples.modulate(ch_obj)
        return self._give_curves_from_samples(mod_samples)

    def interpolate_curves(
        self, curves: list[np.ndarray], sampling_rate: float
    ) -> list[np.ndarray]:
        """The curves with a fractional sampling rate."""
        indices = np.linspace(
            0,
            self.samples.duration,
            num=int(sampling_rate * self.samples.duration),
            endpoint=False,
            dtype=int,
        )
        sampled_curves = [curve[indices] for curve in curves]
        t = np.arange(self.samples.duration)
        return [CubicSpline(indices, sc)(t) for sc in sampled_curves]

    def curves_on_indices(self) -> list[int]:
        """The indices of the curves to draw."""
        return [i for i, qty in enumerate(CURVES_ORDER) if self.curves_on[qty]]

    def _give_curves_from_samples(
        self, samples: ChannelSamples
    ) -> list[np.ndarray]:
        curves = []
        for qty in CURVES_ORDER:
            qty_arr = cast(
                pm.AbstractArray,
                getattr(samples, self._samples_from_curves[qty]),
            ).as_array(detach=True)
            if "phase" in qty:
                qty_arr = qty_arr / (2 * np.pi)
            curves.append(qty_arr)
        return curves


def gather_data(
    sampled_seq: SequenceSamples,
    shown_duration: Optional[int] = None,
) -> dict:
    """Collects the whole sequence data for plotting.

    Args:
        sampled_seq: The samples of a sequence of operations on a device.
        shown_duration: If present, is the total duration to be shown in
            the X axis.

    Returns:
        The data to plot.
    """
    # The minimum time axis length is 100 ns
    total_duration = max(sampled_seq.max_duration, 100, shown_duration or 100)
    data: dict[str, Any] = {}
    for ch, ch_samples in sampled_seq.channel_samples.items():
        target: dict[Union[str, tuple[int, int]], Any] = {}
        # Extracting the EOM Buffers
        eom_intervals = [
            EOMSegment(eom_interval[0], eom_interval[1])
            for eom_interval in ch_samples.get_eom_mode_intervals()
        ]
        # Last eom interval is extended if eom mode not disabled at the end
        if (
            len(eom_intervals) > 0
            and ch_samples.duration == eom_intervals[-1].tf
        ):
            eom_intervals[-1].tf = total_duration
        # sampling the channel schedule
        extended_samples = ch_samples.extend_duration(total_duration)

        eom_start_buffers = [
            EOMSegment(eom_interval[0], eom_interval[1])
            for eom_interval in ch_samples.eom_start_buffers
        ]
        eom_end_buffers = [
            EOMSegment(eom_interval[0], eom_interval[1])
            for eom_interval in ch_samples.eom_end_buffers
        ]

        for time_slot in ch_samples.target_time_slots:
            if time_slot.ti == -1:
                target["initial"] = time_slot.targets
                continue
            target[(time_slot.ti, time_slot.tf - 1)] = time_slot.targets

        # Store everything
        data[ch] = ChannelDrawContent(
            extended_samples,
            target,
            eom_intervals,
            eom_start_buffers,
            eom_end_buffers,
        )

    if sampled_seq._measurement is not None:
        data["measurement"] = sampled_seq._measurement
    data["total_duration"] = total_duration
    return data


def gather_qubit_data(
    sampled_seq: SequenceSamples,
    data: dict,
    register: Optional[BaseRegister] = None,
    modulate: bool = False,
) -> list[dict[tuple, np.ndarray]]:
    """Collects all the data per qubit for plotting.

    Args:
        sampled_seq: The sampled sequence whose amplitude/detuning should
            be displayed per qubit.
        data: A dictionary associating ChannelDrawContents to channels.
        register: The register associated with the sampled sequence.
        modulate: Whether to gather the modulated amplitude/
            detuning using the modulation bandwidth of the channels.

    Returns:
        A list of two dictionaries, associating an array of amplitude
        (1st dict)/detuning (2nd dict) to tuple of qubits sharing same
        amplitude/detuning.
    """
    # Gathers all the targeted qubits
    all_targets = (
        set().union(
            *[
                set(target)
                for ch, ch_data in data.items()
                if ch not in ["measurement", "total_duration"]
                for target in list(ch_data.target.values())
            ]
        )
        if register is None
        else set(register.qubit_ids)
    )
    total_duration = data["total_duration"]
    # Init: All the qubits are targeted by a pulse of zero amp/det
    qubit_data: list[dict[tuple, np.ndarray]] = [
        {
            tuple(all_targets): np.zeros(total_duration),
        }
        for _ in range(2)
    ]
    for ch, ch_data in data.items():
        # for each channel
        if ch in ["measurement", "total_duration"]:
            continue
        # Associate a set of targets to a weight for each new target slot
        for times, target in ch_data.target.items():
            y = (
                ch_data.get_input_curves()
                if not modulate
                else ch_data.get_output_curves(sampled_seq._ch_objs[ch])
            )
            weight_target_map: list[dict[float, set]] = [{} for _ in range(2)]
            if isinstance(ch_data.samples, DMMSamples):
                assert len(ch_data.target) == 1
                # DMM channel
                # Defining targeted qubits
                det_map = cast(DetuningMap, ch_data.samples.detuning_map)
                det_weight_map = defaultdict(
                    int,
                    det_map.get_qubit_weight_map(
                        ch_data.samples.qubits
                        if not register
                        else register.qubits
                    ),
                )
                # 0 amplitude on all qubits:
                weight_target_map[0][0] = all_targets
                # If zero detuning, equivalent to having zero weight
                if np.all(0.0 == ch_data.samples.det):
                    weight_target_map[1][0] = all_targets
                else:
                    # Regroup the qubits targeted by the same weight
                    for t, w in det_weight_map.items():
                        if w not in weight_target_map[1]:
                            weight_target_map[1][w] = set()
                        weight_target_map[1][w].add(t)
            else:
                for i, samples in enumerate(y[:2]):
                    if np.all(samples == 0.0):
                        weight_target_map[i][0] = all_targets
                        continue
                    weight_target_map[i][1] = set(target)
                    weight_target_map[i][0] = all_targets - set(target)

            # Update qubit data
            old_qubit_data = qubit_data.copy()
            qubit_data = [{} for _ in range(2)]
            for i, ch_samples in enumerate(y[:2]):  # not interested in phase
                for q, q_data in old_qubit_data[i].items():
                    for w, set_t in weight_target_map[i].items():
                        sub_target = set_t.intersection(set(q))
                        if sub_target != set():
                            pad_times = (
                                0 if times == "initial" else target[0],
                                (
                                    0
                                    if times == "initial"
                                    else total_duration - target[1]
                                ),
                            )
                            qubit_data[i][tuple(sub_target)] = (
                                w
                                * np.pad(
                                    ch_samples[:total_duration], pad_times
                                )
                                + q_data
                            )
    return qubit_data


def _draw_register_det_maps(
    sampled_seq: SequenceSamples,
    register: Optional[BaseRegister] = None,
    draw_detuning_maps: bool = False,
) -> Figure | None:
    fig_reg: Figure | None = None
    det_maps = {
        ch: cast(DetuningMap, cast(DMMSamples, ch_samples).detuning_map)
        for (ch, ch_samples) in sampled_seq.channel_samples.items()
        if isinstance(sampled_seq._ch_objs[ch], DMM)
    }
    n_det_maps = len(det_maps)
    nregisters = (
        int(register is not None) + int(draw_detuning_maps) * n_det_maps
    )
    # Draw masked register
    if register:
        pos = register._coords_arr.as_array(detach=True)
        title = (
            "Register"
            if sampled_seq._slm_mask.targets == set()
            else "Masked register"
        )
        if isinstance(register, Register3D):
            labels = "xyz"
            fig_reg, axes_reg = register._initialize_fig_axes_projection(
                pos,
                blockade_radius=35,
                draw_half_radius=True,
                nregisters=nregisters,
            )

            for ax_reg, (ix, iy) in zip(
                axes_reg if nregisters == 1 else axes_reg[0],
                combinations(np.arange(3), 2),
            ):
                register._draw_2D(
                    ax=ax_reg,
                    pos=pos,
                    ids=register._ids,
                    plane=(ix, iy),
                    masked_qubits=sampled_seq._slm_mask.targets,
                )
                ax_reg.set_title(
                    title
                    + " projected onto\n the "
                    + labels[ix]
                    + labels[iy]
                    + "-plane"
                )

        elif isinstance(register, Register):
            fig_reg, axes_reg = register._initialize_fig_axes(
                pos,
                blockade_radius=35,
                draw_half_radius=True,
                nregisters=nregisters,
            )
            ax_reg = axes_reg if nregisters == 1 else axes_reg[0]
            register._draw_2D(
                ax=ax_reg,
                pos=pos,
                ids=register._ids,
                masked_qubits=sampled_seq._slm_mask.targets,
            )
            ax_reg.set_title(title, pad=10)
    # Draw detuning maps
    if draw_detuning_maps:
        # Initialize figure for detuning maps if register was not shown
        need_init = register is None
        for i, (ch, det_map) in enumerate(det_maps.items()):
            qubits = (
                register.qubits
                if register
                else cast(DMMSamples, sampled_seq.channel_samples[ch]).qubits
            )
            reg_det_map = det_map.get_qubit_weight_map(qubits)
            pos = np.array([c.as_array(detach=True) for c in qubits.values()])
            if need_init:
                if det_map.dimensionality == 3:
                    labels = "xyz"
                    (
                        fig_reg,
                        axes_reg,
                    ) = det_map._initialize_fig_axes_projection(
                        pos,
                        nregisters=nregisters,
                    )
                else:
                    fig_reg, axes_reg = det_map._initialize_fig_axes(
                        pos,
                        nregisters=nregisters,
                    )
                need_init = False
            ax_reg = (
                axes_reg
                if nregisters == 1
                else axes_reg[i + int(register is not None)]
            )
            if det_map.dimensionality == 3:
                for sub_ax_reg, (ix, iy) in zip(
                    ax_reg, combinations(np.arange(3), 2)
                ):
                    det_map._draw_2D(
                        ax=sub_ax_reg,
                        pos=pos,
                        ids=list(qubits.keys()),
                        plane=(ix, iy),
                        dmm_qubits=reg_det_map,
                    )
                    sub_ax_reg.set_title(
                        f"{ch} projected onto\n the "
                        + labels[ix]
                        + labels[iy]
                        + "-plane"
                    )
            else:
                det_map._draw_2D(
                    ax=ax_reg,
                    pos=pos,
                    ids=list(qubits.keys()),
                    dmm_qubits=reg_det_map,
                )
                ax_reg.set_title(ch, pad=10)
    return fig_reg


def _draw_channel_content(
    sampled_seq: SequenceSamples,
    sampling_rate: Optional[float] = None,
    draw_phase_area: bool = False,
    draw_phase_shifts: bool = False,
    draw_input: bool = True,
    draw_modulation: bool = False,
    draw_phase_curve: bool = False,
    draw_detuning_maps: bool = False,
    phase_modulated: bool = False,
    shown_duration: Optional[int] = None,
) -> tuple[Figure, Any, dict]:
    """Draws samples of a sequence.

    Args:
        sampled_seq: The input samples of a sequence of operations.
        register: If present, draw the register before the pulse
            sequence, with a visual indication (square halo) around the qubits
            masked by the SLM.
        sampling_rate: Sampling rate of the effective pulse used by
            the solver. If present, plots the effective pulse alongside the
            input pulse.
        draw_phase_area: Whether phase and area values need to be shown
            as text on the plot, defaults to False. If `draw_phase_curve=True`,
            phase values are ommited.
        draw_phase_shifts: Whether phase shift and reference information
            should be added to the plot, defaults to False.
        draw_input: Draws the programmed pulses on the channels, defaults
            to True.
        draw_modulation: Draws the expected channel output, defaults to
            False. If the channel does not have a defined 'mod_bandwidth', this
            is skipped unless 'draw_input=False'.
        draw_phase_curve: Draws the changes in phase in its own curve (ignored
            if the phase doesn't change throughout the channel).
        draw_detuning_maps: Draws the detuning maps applied on the qubits of
            the register of the sequence. Shown before the pulse sequence,
            defaults to False.
        phase_modulated: Show the phase modulation samples instead of the
            detuning and phase offset combination.
        shown_duration: Total duration to be shown in the X axis.
    """

    def phase_str(phi: Any) -> str:
        """Formats a phase value for printing."""
        value = (((float(phi) + np.pi) % (2 * np.pi)) - np.pi) / np.pi
        if value == -1:
            return r"$\pi$"
        elif value == 0:
            return "0"  # pragma: no cover - just for safety
        else:
            return rf"{float(value):.2g}$\pi$"

    data = gather_data(sampled_seq, shown_duration)
    n_channels = len(sampled_seq.channels)
    total_duration = data["total_duration"]
    time_scale = 1e3 if total_duration > 1e4 else 1
    for ch in sampled_seq.channels:
        data[ch].phase_modulated = phase_modulated
        curves_on = data[ch].curves_on.copy()
        _, det_samples_, phase_samples_ = data[ch].get_input_curves()
        non_zero_det = np.count_nonzero(det_samples_) > 0
        non_zero_phase = np.count_nonzero(phase_samples_) > 0
        curves_on["detuning"] = non_zero_det ^ (
            phase_modulated and non_zero_phase
        )
        curves_on["phase"] = (
            phase_modulated or draw_phase_curve
        ) and non_zero_phase

        if any(curve_on for curve_on in curves_on.values()):
            # The channel is not empty
            data[ch].curves_on = curves_on

    # Boxes for qubit and phase text
    q_box = dict(boxstyle="round", facecolor="orange")
    ph_box = dict(boxstyle="round", facecolor="ghostwhite")
    area_ph_box = dict(boxstyle="round", facecolor="ghostwhite", alpha=0.7)
    slm_box = dict(boxstyle="round", alpha=0.4, facecolor="grey", hatch="//")
    eom_box = dict(boxstyle="round", facecolor="lightsteelblue")

    ratios = [
        SIZE_PER_WIDTH[data[ch].n_axes_on] for ch in sampled_seq.channels
    ]
    fig = plt.figure(
        constrained_layout=False,
        figsize=(20, sum(ratios)),
    )
    gs = fig.add_gridspec(n_channels, 1, hspace=0.075, height_ratios=ratios)

    ch_axes = {}
    for i, (ch, gs_) in enumerate(zip(sampled_seq.channels, gs)):
        ax = fig.add_subplot(gs_)
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color("none")
        ax.tick_params(
            labelcolor="w", top=False, bottom=False, left=False, right=False
        )
        ax.set_ylabel(ch, labelpad=40, fontsize=18)
        subgs = gs_.subgridspec(data[ch].n_axes_on, 1, hspace=0.0)
        ch_axes[ch] = [
            fig.add_subplot(subgs[i, :]) for i in range(data[ch].n_axes_on)
        ]
        for j, ax in enumerate(ch_axes[ch]):
            ax.axvline(0, linestyle="--", linewidth=0.5, color="grey")
            if j > 0:
                ax.spines["top"].set_visible(False)
            if j < len(ch_axes[ch]) - 1:
                ax.spines["bottom"].set_visible(False)
            if i < n_channels - 1 or j < len(ch_axes[ch]) - 1:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                    top=False,
                    labelbottom=False,
                    direction="in",
                )
            else:
                unit = "ns" if time_scale == 1 else r"$\mu s$"
                ax.set_xlabel(f"t ({unit})", fontsize=12)

    # The time axis of all channels is the same
    t = np.arange(total_duration) / time_scale
    final_t = t[-1]
    t_min = -final_t * 0.03
    t_max = final_t * 1.05

    for ch, axes in ch_axes.items():
        ch_data = data[ch]
        ch_obj = sampled_seq._ch_objs[ch]
        ch_eom_intervals = data[ch].eom_intervals
        ch_eom_start_buffers = data[ch].eom_start_buffers
        ch_eom_end_buffers = data[ch].eom_end_buffers
        basis = ch_obj.basis
        ys = ch_data.get_input_curves()
        ys_mod = [()] * 3
        yseff = [()] * 3
        draw_output = draw_modulation and (
            ch_obj.mod_bandwidth or not draw_input
        )
        if draw_output:
            ys_mod = ch_data.get_output_curves(ch_obj)

        if sampling_rate:
            curves = ys_mod if draw_output else ys
            yseff = ch_data.interpolate_curves(curves, sampling_rate)
        ref_ys = [
            list(chain.from_iterable(all_ys))
            for all_ys in zip(ys, ys_mod, yseff)
        ]
        max_amp = np.max(ref_ys[0])
        max_amp = 1 if max_amp == 0 else max_amp
        amp_top = max_amp * 1.2
        amp_bottom = min(0.0, *ref_ys[0])
        # Makes sure that [-1, 1] range is always represented
        det_max = max(*ref_ys[1], 1)
        det_min = min(*ref_ys[1], -1)
        det_range = det_max - det_min
        det_top = det_max + det_range * 0.15
        det_bottom = det_min - det_range * 0.05
        # Phase limits
        phase_min = min(*ref_ys[2], 0.0)
        phase_max = max(*ref_ys[2], 1.0 if not phase_modulated else 0.1)
        phase_range = phase_max - phase_min
        phase_top = phase_max + phase_range * 0.15
        phase_bottom = phase_min - phase_range * 0.05
        ax_lims = [
            (amp_bottom, amp_top),
            (det_bottom, det_top),
            (phase_bottom, phase_top),
        ]
        ax_lims = [ax_lims[i] for i in ch_data.curves_on_indices()]
        for ax, ylim in zip(axes, ax_lims):
            ax.set_xlim(t_min, t_max)
            ax.set_ylim(*ylim)

        for i, ax in zip(ch_data.curves_on_indices(), axes):
            if draw_input:
                ax.plot(t, ys[i], color=COLORS[i], linewidth=0.8)
            if sampling_rate:
                ax.plot(
                    t,
                    yseff[i],
                    color=COLORS[i],
                    linewidth=0.8,
                )
                ax.fill_between(t, 0, yseff[i], color=COLORS[i], alpha=0.3)
            elif draw_input:
                ax.fill_between(t, 0, ys[i], color=COLORS[i], alpha=0.3)
            if draw_output:
                if not sampling_rate:
                    ax.fill_between(
                        t,
                        0,
                        ys_mod[i][:total_duration],
                        color=COLORS[i],
                        alpha=0.3,
                        hatch="////",
                    )
                else:
                    ax.plot(
                        t,
                        ys_mod[i][:total_duration],
                        color=COLORS[i],
                        linestyle="dotted",
                    )
            special_kwargs = dict(labelpad=10) if i == 0 else {}
            ax.set_ylabel(LABELS[i], fontsize=14, **special_kwargs)

        if draw_phase_area:
            top = False  # Variable to track position of box, top or center.
            print_phase = not draw_phase_curve and any(
                np.any(ch_data.samples.phase[slot.ti : slot.tf] != 0)
                for slot in ch_data.samples.slots
            )

            for slot in ch_data.samples.slots:
                if sampling_rate:
                    area_val = (
                        np.sum(yseff[0][slot.ti : slot.tf]) * 1e-3 / np.pi
                    )
                else:
                    area_val = (
                        np.sum(ch_data.samples.amp[slot.ti : slot.tf])
                        * 1e-3
                        / np.pi
                    )
                phase_val = ch_data.samples.phase[slot.tf - 1]
                x_plot = (slot.ti + slot.tf) / 2 / time_scale
                target_slot_tf_list = [
                    target_slot.tf
                    for target_slot in sampled_seq.channel_samples[
                        ch
                    ].target_time_slots
                ]
                if slot.ti in target_slot_tf_list or not top:
                    y_plot = np.max(ch_data.samples.amp[slot.ti : slot.tf]) / 2
                    top = True  # Next box at the top.
                elif top:
                    y_plot = np.max(ch_data.samples.amp[slot.ti : slot.tf])
                    top = False  # Next box at the center.
                area_fmt = (
                    r"A: $\pi$"
                    if round(area_val, 2) == 1
                    else rf"A: {float(area_val):.2g}$\pi$"
                )
                if not print_phase:
                    txt = area_fmt
                else:
                    phase_fmt = rf"$\phi$: {phase_str(phase_val)}"
                    txt = "\n".join([phase_fmt, area_fmt])
                axes[0].text(
                    x_plot,
                    y_plot,
                    txt,
                    fontsize=10,
                    ha="center",
                    va="center",
                    bbox=area_ph_box,
                )

        target_regions = []  # [[start1, [targets1], end1],...]
        tgt_txt_ymax = ax_lims[0][1] * 0.92
        for coords in ch_data.target:
            targets = list(ch_data.target[coords])
            tgt_strs = [str(q) for q in targets]
            if isinstance(ch_obj, DMM):
                tgt_strs = ["⚄"]
            elif ch_obj.addressing == "Global":
                tgt_strs = ["GLOBAL"]
            tgt_txt_y = tgt_txt_ymax - 0.25 * (len(tgt_strs) - 1)
            tgt_str = "\n".join(tgt_strs)
            if coords == "initial":
                x = t_min + final_t * 0.005
                target_regions.append([0, targets])
                if ch_obj.addressing == "Global":
                    axes[0].text(
                        x,
                        tgt_txt_ymax * 1.065,
                        tgt_strs[0],
                        fontsize=13 if tgt_strs == ["GLOBAL"] else 17,
                        rotation=90 if tgt_strs == ["GLOBAL"] else 0,
                        ha="left",
                        va="top",
                        bbox=q_box,
                    )
                else:
                    axes[0].text(
                        x,
                        tgt_txt_y,
                        tgt_str,
                        fontsize=12,
                        ha="left",
                        bbox=q_box,
                    )
                    phase = sampled_seq._basis_ref[basis][targets[0]].phase[0]
                    if phase and draw_phase_shifts:
                        msg = r"$\phi=$" + phase_str(phase)
                        axes[0].text(
                            0,
                            tgt_txt_ymax,
                            msg,
                            ha="left",
                            fontsize=12,
                            bbox=ph_box,
                        )
            else:
                ti, tf = np.array(coords) / time_scale
                target_regions[-1].append(ti)  # Closing previous regions
                target_regions.append(
                    [tf + 1 / time_scale, targets]
                )  # New one
                phase = sampled_seq._basis_ref[basis][targets[0]].phase[
                    tf * time_scale + 1
                ]
                for ax in axes:
                    ax.axvspan(ti, tf, alpha=0.4, color="grey", hatch="//")
                axes[0].text(
                    tf + final_t * 5e-3,
                    tgt_txt_y,
                    tgt_str,
                    ha="left",
                    fontsize=12,
                    bbox=q_box,
                )
                if phase and draw_phase_shifts:
                    msg = r"$\phi=$" + phase_str(phase)
                    wrd_len = len(max(tgt_strs, key=len))
                    x = tf + final_t * 0.01 * (wrd_len + 1)
                    axes[0].text(
                        x,
                        tgt_txt_ymax,
                        msg,
                        ha="left",
                        fontsize=12,
                        bbox=ph_box,
                    )

        # Terminate the last open regions
        if target_regions:
            target_regions[-1].append(final_t)
        for start, targets_, end in (
            target_regions if draw_phase_shifts else []
        ):
            start = cast(float, start)
            targets_ = cast(list, targets_)
            end = cast(float, end)
            # All targets have the same ref, so we pick
            q = targets_[0]
            ref = sampled_seq._basis_ref[basis][q].phase
            if end != total_duration - 1 or "measurement" in data:
                end += 1 / time_scale
            for t_, delta in ref.changes(start, end, time_scale=time_scale):
                conf = dict(linestyle="--", linewidth=1.5, color="black")
                for ax in axes:
                    ax.axvline(t_, **conf)
                msg = "\u27F2 " + phase_str(delta)
                axes[0].text(
                    t_ - final_t * 8e-3,
                    tgt_txt_ymax,
                    msg,
                    ha="right",
                    fontsize=14,
                    bbox=ph_box,
                )

        # Draw the EOM intervals
        for ch_eom_start_buffer, ch_eom_interval, ch_eom_end_buffer in zip(
            ch_eom_start_buffers, ch_eom_intervals, ch_eom_end_buffers
        ):
            for ax in axes:
                ch_eom_start_buffer.smooth_draw(ax, decreasing=False)
                ch_eom_interval.draw(ax)
                ch_eom_end_buffer.smooth_draw(ax, decreasing=True)
            tgt_txt_x = ch_eom_start_buffer.ti or ch_eom_interval.ti
            tgt_txt_y = axes[0].get_ylim()[1]
            axes[0].text(
                tgt_txt_x,
                tgt_txt_y,
                "EOM",
                fontsize=12,
                ha="left",
                va="top",
                bbox=eom_box,
            )
        # Draw the SLM mask
        if sampled_seq._slm_mask.targets and sampled_seq._slm_mask.end:
            tf_m = sampled_seq._slm_mask.end
            for ax in axes:
                ax.axvspan(0, tf_m, color="black", alpha=0.1, zorder=-100)
            tgt_strs = [str(q) for q in sampled_seq._slm_mask.targets]
            tgt_txt_x = final_t * 0.005
            tgt_txt_y = axes[-1].get_ylim()[0]
            tgt_str = "\n".join(tgt_strs)
            axes[-1].text(
                tgt_txt_x,
                tgt_txt_y,
                tgt_str,
                fontsize=12,
                ha="left",
                bbox=slm_box,
            )

        hline_kwargs = dict(linestyle="-", linewidth=0.5, color="grey")
        if "measurement" in data:
            msg = f"Basis: {data['measurement']}"
            if len(axes) == 1:
                mid_ax = axes[0]
                mid_point = sum(ax_lims[0]) / 2
                fontsize = 12
            else:
                mid_ax = axes[-1]
                mid_point = (
                    ax_lims[-1][1]
                    if len(axes) == 2
                    else ax_lims[-1][0] + sum(ax_lims[-1]) * 1.5
                )
                fontsize = 14

            for ax in axes:
                ax.axvspan(final_t, t_max, color="midnightblue", alpha=1)

            mid_ax.text(
                final_t * 1.025,
                mid_point,
                msg,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
                rotation=90,
            )
            hline_kwargs["xmax"] = 0.95

        for i, ax in enumerate(axes):
            if i > 0:
                ax.axhline(ax_lims[i][1], **hline_kwargs)
            if ax_lims[i][0] < 0:
                ax.axhline(0, **hline_kwargs)
    return (fig, ch_axes, data)


def _draw_qubit_content(
    sampled_seq: SequenceSamples,
    data: dict,
    register: Optional[BaseRegister] = None,
    draw_input: bool = True,
    draw_modulation: bool = False,
    draw_qubit_amp: bool = False,
    draw_qubit_det: bool = True,
) -> tuple[Figure | None, Figure | None]:
    """Gets information to plot per qubits.

    Draws the amplitude and detuning seen locally by each qubit.

    Args:
        sampled_seq: The sampled sequence whose amplitude/detuning should
            be displayed per qubit.
        data: A dictionary associating ChannelDrawContents to channels.
        register: The register associated with the sampled sequence.
        draw_input: Whether to draw to input of the amplitude/detuning.
        draw_qubit_modulation: Whether to draw the modulated amplitude/
            detuning using the modulation bandwidth of the channels.
        draw_qubit_amp: Whether to draw the amplitude per qubit or not.
        draw_qubit_det: Whether to draw the detuning per qubit or not.

    Returns:
        A figure displaying the amplitude/detuning seen by the atoms locally.
        The atoms having same amplitude/detuning are grouped together. If a
        register is provided, these groups of atoms are displayed in space in
        a second figure.
    """
    # Show nothing if no drawing per qubit asked
    if not draw_qubit_det and not draw_qubit_amp:
        return (None, None)
    # Or if a channel is not in the ground-rydberg basis
    elif not np.all(
        [
            ch_obj.basis == "ground-rydberg"
            for ch, ch_obj in sampled_seq._ch_objs.items()
        ]
    ):
        raise NotImplementedError(
            "Can only draw qubit contents for channels in rydberg basis."
        )
    # Gather data per targeted qubits
    total_duration = data["total_duration"]
    draw_data = {"input": draw_input, "modulated": draw_modulation}
    n_data = sum(list(draw_data.values()))
    qubit_data = [
        (
            gather_qubit_data(
                sampled_seq, data, register, (data_name == "modulated")
            )
            if to_draw
            else None
        )
        for data_name, to_draw in draw_data.items()
    ]
    # Figure composed of 2 subplots (input, modulated) each composed
    # of 2 subplots (amplitude, detuning)
    draw_quantities = [draw_qubit_amp, draw_qubit_det]
    n_quantities = sum(draw_quantities)
    ratios = [SIZE_PER_WIDTH[n_quantities]] * n_data
    fig = plt.figure(
        constrained_layout=False,
        figsize=(20, sum(ratios)),
    )
    gs = fig.add_gridspec(n_data, 1, hspace=0.075, height_ratios=ratios)

    fig.suptitle("Quantities per qubit over time", fontsize=16)
    cmap = LinearSegmentedColormap.from_list("", COLORS)
    hline_kwargs = dict(linestyle="-", linewidth=0.5, color="grey")
    max_targets = 20  # maximum number of targets shown in legend
    # If qubits can be defined, another figure is added to display legend
    dmm_samples: list[DMMSamples] = [
        cast(DMMSamples, sampled_seq.channel_samples[ch])
        for ch in sampled_seq.channels
        if isinstance(sampled_seq.channel_samples[ch], DMMSamples)
    ]
    qubits: None | dict = None
    if register:
        qubits = register.qubits
    elif dmm_samples:
        qubits = dmm_samples[0].qubits
    else:
        warnings.warn(
            "Provide a register and select draw_register for a more"
            "visible representation",
            UserWarning,
        )
    fig_legend: None | Figure = None
    axes_legend: None | Axes = None
    dimensionality_3d: bool | None = None
    if register or dmm_samples:
        dimensionality_3d = isinstance(register, Register3D) or (
            dmm_samples != []
            and cast(DetuningMap, dmm_samples[0].detuning_map).dimensionality
            == 3
        )
        if dimensionality_3d:
            labels = "xyz"
            (
                fig_legend,
                axes_legend,
            ) = Register._initialize_fig_axes_projection(
                np.array(list(cast(dict, qubits).values())),
                nregisters=n_quantities,
            )
        else:
            (fig_legend, axes_legend) = Register._initialize_fig_axes(
                np.array(list(cast(dict, qubits).values())),
                nregisters=n_quantities,
            )
    # The time axis of all channels is the same
    time_scale = 1e3 if total_duration > 1e4 else 1
    time = np.arange(total_duration) / time_scale
    final_t = time[-1]
    t_min = -final_t * 0.03
    t_max = final_t * 1.05
    # Draw mode (input/modulated/both)
    plot_index = 0
    for data_index, (data_name, to_draw) in enumerate(draw_data.items()):
        if not to_draw:
            continue
        assert qubit_data[data_index] is not None
        # Define plot
        gs_ = gs[plot_index]
        ax = fig.add_subplot(gs_)
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color("none")
        ax.tick_params(
            labelcolor="w", top=False, bottom=False, left=False, right=False
        )
        ax.set_ylabel(data_name, labelpad=40, fontsize=18)
        subgs = gs_.subgridspec(n_quantities, 1, hspace=0.0)
        axes = [fig.add_subplot(subgs[i, :]) for i in range(n_quantities)]
        # Draw quantity (amplitude/detuning/both)
        subplot_index = 0
        for i, draw_qty in enumerate(draw_quantities):
            if not draw_qty:
                continue
            # Define subplot
            sub_ax = axes[subplot_index]
            sub_ax.axvline(0, linestyle="--", linewidth=0.5, color="grey")
            if subplot_index > 0:
                sub_ax.spines["top"].set_visible(False)
            if subplot_index < n_quantities - 1:
                sub_ax.spines["bottom"].set_visible(False)
            if plot_index < n_data - 1 or subplot_index < n_quantities - 1:
                sub_ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                    top=False,
                    labelbottom=False,
                    direction="in",
                )
            else:
                unit = "ns" if time_scale == 1 else r"$\mu s$"
                sub_ax.set_xlabel(f"t ({unit})", fontsize=12)
            # Define the y axis
            max_val = np.max(
                [
                    local_data
                    for local_data in list(
                        cast(list, qubit_data[data_index])[i].values()
                    )
                ]
            )
            min_val = np.min(
                [
                    local_data
                    for local_data in list(
                        cast(list, qubit_data[data_index])[i].values()
                    )
                ]
            )
            if i == 0:
                max_val = 1 if max_val == 0 else max_val
                max_val = max_val * 1.2
                min_val = min(min_val, 0.0)
            elif i == 1:
                # Makes sure that [-1, 1] range is always represented
                max_val = max(max_val, 1)
                min_val = min(min_val, -1)
                range_val = max_val - min_val
                max_val = max_val + range_val * 0.15
                min_val = min_val - range_val * 0.05
            sub_ax.set_xlim(t_min, t_max)
            sub_ax.set_ylim(min_val, max_val)
            # Plot one curve per target
            nb_targets = len(cast(list, qubit_data[data_index])[i])
            for target_index, (target, q_data) in enumerate(
                cast(list, qubit_data[data_index])[i].items()
            ):
                # label is simpler if qubits are defined
                label: str = ""
                if qubits:
                    label = f"targets_{target_index}"
                else:
                    for label_index in range(0, len(target), max_targets):
                        sub_target = map(
                            str,
                            target[label_index : label_index + max_targets],
                        )
                        label += ",".join(sub_target) + "\n"
                # Add curve
                color = cmap(target_index / nb_targets)
                sub_ax.plot(
                    time,
                    q_data,
                    label=label,
                    color=color,
                    linewidth=0.8,
                    linestyle="--" if data_name == "modulated" else "-",
                )
                # Add targets to legend if qubits are defined
                if plot_index == 0 and qubits:
                    ax_leg = (
                        axes_legend
                        if n_quantities == 1
                        else cast(list, axes_legend)[subplot_index]
                    )
                    targeted_atoms = {t: qubits[t] for t in target}
                    pos = np.array(list(targeted_atoms.values()))
                    if dimensionality_3d:
                        for sub_ax_leg, (ix, iy) in zip(
                            ax_leg, combinations(np.arange(3), 2)
                        ):
                            Register._draw_2D(
                                ax=sub_ax_leg,
                                pos=pos,
                                ids=list(targeted_atoms.keys()),
                                plane=(ix, iy),
                                qubit_colors={t: color for t in target},
                                masked_qubits=set(target),
                                label_name=label,
                            )
                            if target_index == 0:
                                sub_ax_leg.set_title(
                                    f"{LABELS[i]} projected onto\n the "
                                    + labels[ix]
                                    + labels[iy]
                                    + "-plane"
                                )
                    else:
                        Register._draw_2D(
                            ax=ax_leg,
                            pos=pos,
                            ids=list(targeted_atoms.keys()),
                            qubit_colors={t: color for t in target},
                            masked_qubits=set(target),
                            label_name=label,
                        )
                        if target_index == 0:
                            ax_leg.set_title(
                                f"Targeted atoms for {LABELS[i][:8]}", pad=10
                            )
            sub_ax.set_ylabel(LABELS[i], fontsize=14)
            # Show legend only if qubits can't be defined
            sub_ax.legend()

            if subplot_index > 0:
                sub_ax.axhline(max_val, **hline_kwargs)
            if min_val < 0:
                sub_ax.axhline(0, **hline_kwargs)
            subplot_index += 1
        plot_index += 1

    return fig, fig_legend


def draw_samples(
    sampled_seq: SequenceSamples,
    register: Optional[BaseRegister] = None,
    sampling_rate: Optional[float] = None,
    draw_phase_area: bool = False,
    draw_phase_shifts: bool = False,
    draw_phase_curve: bool = False,
    draw_detuning_maps: bool = False,
    draw_qubit_amp: bool = False,
    draw_qubit_det: bool = False,
    phase_modulated: bool = False,
) -> tuple[Figure | None, Figure, Figure | None, Figure | None]:
    """Draws a SequenceSamples.

    Args:
        sampled_seq: The input samples of a sequence of operations.
        register: If present, draw the register before the pulse
            sequence samples, with a visual indication (square halo)
            around the qubits masked by the SLM.
        sampling_rate: Sampling rate of the effective pulse used by
            the solver. If present, plots the effective pulse alongside the
            input pulse.
        draw_phase_area: Whether phase and area values need to be shown
            as text on the plot, defaults to False. If `draw_phase_curve=True`,
            phase values are ommited.
        draw_phase_shifts: Whether phase shift and reference information
            should be added to the plot, defaults to False.
        draw_phase_curve: Draws the changes in phase in its own curve (ignored
            if the phase doesn't change throughout the channel).
        draw_detuning_maps: Whether to draw the detuning maps applied on the
            qubits of the provided register. Shown before the samples,
            defaults to False.
        draw_qubit_amp: Draws the amplitude seen by the qubits locally after
            the drawing of the sequence.
        draw_qubit_det: Draws the detuning seen by the qubits locally after
            the drawing of the sequence.
        phase_modulated: Show the phase modulation samples instead of the
            detuning and phase offset combination.
    """
    if not len(sampled_seq.channels):
        raise RuntimeError("Can't draw an empty sequence.")
    slot_tfs = [
        ch_samples.slots[-1].tf
        for ch_samples in sampled_seq.channel_samples.values()
    ]
    max_slot_tf = max(slot_tfs) if len(slot_tfs) > 0 else None
    # Draw register and detuning maps
    fig_reg = _draw_register_det_maps(
        sampled_seq, register, draw_detuning_maps
    )
    (fig, ch_axes, data) = _draw_channel_content(
        sampled_seq,
        sampling_rate,
        draw_phase_area,
        draw_phase_shifts,
        draw_input=True,
        draw_modulation=False,
        draw_phase_curve=draw_phase_curve,
        phase_modulated=phase_modulated,
        shown_duration=max_slot_tf,
    )
    (fig_qubit, fig_legend) = _draw_qubit_content(
        sampled_seq,
        data,
        register,
        draw_input=True,
        draw_modulation=False,
        draw_qubit_amp=draw_qubit_amp,
        draw_qubit_det=draw_qubit_det,
    )
    return (fig_reg, fig, fig_qubit, fig_legend)


def draw_sequence(
    seq: pulser.sequence.Sequence,
    sampling_rate: Optional[float] = None,
    draw_phase_area: bool = False,
    draw_interp_pts: bool = True,
    draw_phase_shifts: bool = False,
    draw_register: bool = False,
    draw_input: bool = True,
    draw_modulation: bool = False,
    draw_phase_curve: bool = False,
    draw_detuning_maps: bool = False,
    draw_qubit_amp: bool = False,
    draw_qubit_det: bool = False,
    phase_modulated: bool = False,
) -> tuple[Figure | None, Figure, Figure | None, Figure | None]:
    """Draws the entire sequence.

    Args:
        seq: The input sequence of operations on a device.
        sampling_rate: Sampling rate of the effective pulse used by
            the solver. If present, plots the effective pulse alongside the
            input pulse.
        draw_phase_area: Whether phase and area values need to be shown
            as text on the plot, defaults to False. If `draw_phase_curve=True`,
            phase values are ommited.
        draw_interp_pts: When the sequence has pulses with waveforms of
            type InterpolatedWaveform, draws the points of interpolation on
            top of the respective waveforms (defaults to True).
        draw_phase_shifts: Whether phase shift and reference information
            should be added to the plot, defaults to False.
        draw_register: Whether to draw the register before the pulse
            sequence, with a visual indication (square halo) around the qubits
            masked by the SLM, defaults to False.
        draw_input: Draws the programmed pulses on the channels, defaults
            to True.
        draw_modulation: Draws the expected channel output, defaults to
            False. If the channel does not have a defined 'mod_bandwidth', this
            is skipped unless 'draw_input=False'.
        draw_phase_curve: Draws the changes in phase in its own curve (ignored
            if the phase doesn't change throughout the channel).
        draw_detuning_maps: Whether to draw the detuning maps applied on the
            qubits of the register of the sequence. Shown before the pulse
            sequence, defaults to False.
        draw_qubit_amp: Draws the amplitude seen by the qubits locally after
            the drawing of the sequence.
        draw_qubit_det: Draws the detuning seen by the qubits locally after
            the drawing of the sequence.
        phase_modulated: Show the phase modulation samples instead of the
            detuning and phase offset combination.
    """
    # Sample the sequence and get the data to plot
    shown_duration = seq.get_duration(include_fall_time=draw_modulation)
    sampled_seq = sample(seq)
    if not len(sampled_seq.channels):
        raise RuntimeError("Can't draw an empty sequence.")
    # Draw register and detuning maps
    fig_reg = _draw_register_det_maps(
        sampled_seq,
        seq.register if draw_register else None,
        draw_detuning_maps,
    )
    (fig, ch_axes, data) = _draw_channel_content(
        sampled_seq,
        sampling_rate,
        draw_phase_area,
        draw_phase_shifts,
        draw_input,
        draw_modulation,
        draw_phase_curve,
        draw_detuning_maps,
        phase_modulated,
        shown_duration,
    )
    draw_output = draw_modulation
    for ch_obj in list(seq.declared_channels.values()):
        draw_output = draw_output and ch_obj.mod_bandwidth is not None
    if (
        not draw_output
        and not draw_input
        and (draw_qubit_det or draw_qubit_amp)
    ):
        warnings.warn(
            "Can't display modulated quantities per qubit if a channel does "
            "not have a modulation bandwidth, displays the input per qubit.",
            UserWarning,
            stacklevel=2,
        )
        draw_input = True
    (fig_qubit, fig_legend) = _draw_qubit_content(
        sampled_seq,
        data,
        seq.register if draw_register else None,
        draw_input=draw_input,
        draw_modulation=draw_output,
        draw_qubit_amp=draw_qubit_amp,
        draw_qubit_det=draw_qubit_det,
    )
    # Gather additional data for sequence specific drawing
    for ch, sch in seq._schedule.items():
        interp_pts: defaultdict[str, list[list[float]]] = defaultdict(list)

        for slot in sch:
            if slot.ti == -1 or slot.type in ["target", "delay"]:
                continue

            pulse = cast(Pulse, slot.type)
            for wf_type in ["amplitude", "detuning"]:
                wf = getattr(pulse, wf_type)
                if isinstance(wf, InterpolatedWaveform):
                    pts = wf.data_points
                    pts[:, 0] += slot.ti
                    interp_pts[wf_type] += pts.tolist()

        if interp_pts:
            data[ch].interp_pts = dict(interp_pts)

    for ch, axes in ch_axes.items():
        ch_data = data[ch]

        if draw_interp_pts:
            for qty in ("amplitude", "detuning"):
                if qty in ch_data.interp_pts and ch_data.curves_on[qty]:
                    ind = CURVES_ORDER.index(qty)
                    pts = np.array(ch_data.interp_pts[qty])
                    axes[ind].scatter(pts[:, 0], pts[:, 1], color=COLORS[ind])

    return (fig_reg, fig, fig_qubit, fig_legend)
