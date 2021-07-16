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

from collections import defaultdict
from typing import Any, cast, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import pulser
from pulser.waveforms import ConstantWaveform, InterpolatedWaveform
from pulser.pulse import Pulse


def gather_data(seq: pulser.sequence.Sequence) -> dict:
    """Collects the whole sequence data for plotting.

    Args:
        seq (pulser.Sequence): The input sequence of operations on a device.

    Returns:
        dict: The data to plot.
    """
    # The minimum time axis length is 100 ns
    total_duration = max(seq.get_duration(), 100)
    data = {}
    for ch, sch in seq._schedule.items():
        time = [-1]  # To not break the "time[-1]" later on
        amp = []
        detuning = []
        # List of interpolation points
        interp_pts: defaultdict[str, list[list[float]]] = defaultdict(list)
        target: dict[Union[str, tuple[int, int]], Any] = {}
        # phase_shift = {}
        for slot in sch:
            if slot.ti == -1:
                target["initial"] = slot.targets
                time += [0]
                amp += [0.0]
                detuning += [0.0]
                continue
            if slot.type in ["delay", "target"]:
                time += [
                    slot.ti,
                    slot.tf - 1 if slot.tf > slot.ti else slot.ti,
                ]
                amp += [0.0, 0.0]
                detuning += [0.0, 0.0]
                if slot.type == "target":
                    target[(slot.ti, slot.tf - 1)] = slot.targets
                continue
            pulse = cast(Pulse, slot.type)
            if isinstance(pulse.amplitude, ConstantWaveform) and isinstance(
                pulse.detuning, ConstantWaveform
            ):
                time += [slot.ti, slot.tf - 1]
                amp += [float(pulse.amplitude._value)] * 2
                detuning += [float(pulse.detuning._value)] * 2
            else:
                time += list(range(slot.ti, slot.tf))
                amp += pulse.amplitude.samples.tolist()
                detuning += pulse.detuning.samples.tolist()
                for wf_type in ["amplitude", "detuning"]:
                    wf = getattr(pulse, wf_type)
                    if isinstance(wf, InterpolatedWaveform):
                        pts = wf.data_points
                        pts[:, 0] += slot.ti
                        interp_pts[wf_type] += pts.tolist()

        if time[-1] < total_duration - 1:
            time += [time[-1] + 1, total_duration - 1]
            amp += [0, 0]
            detuning += [0, 0]
        # Store everything
        time.pop(0)  # Removes the -1 in the beginning
        data[ch] = {
            "time": time,
            "amp": amp,
            "detuning": detuning,
            "target": target,
        }
        if hasattr(seq, "_measurement"):
            data[ch]["measurement"] = seq._measurement
        if interp_pts:
            data[ch]["interp_pts"] = interp_pts
    data["total_duration"] = total_duration
    return data


def draw_sequence(
    seq: pulser.sequence.Sequence,
    sampling_rate: Optional[float] = None,
    draw_phase_area: bool = False,
    draw_interp_pts: bool = True,
    draw_phase_shifts: bool = False,
) -> None:
    """Draws the entire sequence.

    Args:
        seq (pulser.Sequence): The input sequence of operations on a device.
        sampling_rate (float): Sampling rate of the effective pulse used by
            the solver. If present, plots the effective pulse alongside the
            input pulse.
        draw_phase_area (bool): Whether phase and area values need to be shown
            as text on the plot, defaults to False.
        draw_interp_pts (bool): When the sequence has pulses with waveforms of
            type InterpolatedWaveform, draws the points of interpolation on
            top of the respective waveforms (defaults to True).
        draw_phase_shifts (bool): Whether phase shift and reference information
            should be added to the plot, defaults to False.
    """

    def phase_str(phi: float) -> str:
        """Formats a phase value for printing."""
        value = (((phi + np.pi) % (2 * np.pi)) - np.pi) / np.pi
        if value == -1:
            return r"$\pi$"
        elif value == 0:
            return "0"  # pragma: no cover - just for safety
        else:
            return fr"{value:.2g}$\pi$"

    n_channels = len(seq._channels)
    if not n_channels:
        raise RuntimeError("Can't draw an empty sequence.")
    data = gather_data(seq)
    total_duration = data["total_duration"]
    time_scale = 1e3 if total_duration > 1e4 else 1

    # Boxes for qubit and phase text
    q_box = dict(boxstyle="round", facecolor="orange")
    ph_box = dict(boxstyle="round", facecolor="ghostwhite")
    area_ph_box = dict(boxstyle="round", facecolor="ghostwhite", alpha=0.7)

    fig = plt.figure(constrained_layout=False, figsize=(20, 4.5 * n_channels))
    gs = fig.add_gridspec(n_channels, 1, hspace=0.075)

    ch_axes = {}
    for i, (ch, gs_) in enumerate(zip(seq._channels, gs)):
        ax = fig.add_subplot(gs_)
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.tick_params(
            labelcolor="w", top=False, bottom=False, left=False, right=False
        )
        ax.set_ylabel(ch, labelpad=40, fontsize=18)
        subgs = gs_.subgridspec(2, 1, hspace=0.0)
        ax1 = fig.add_subplot(subgs[0, :])
        ax2 = fig.add_subplot(subgs[1, :])
        ch_axes[ch] = (ax1, ax2)
        for j, ax in enumerate(ch_axes[ch]):
            ax.axvline(0, linestyle="--", linewidth=0.5, color="grey")
            if j == 0:
                ax.spines["bottom"].set_visible(False)
            else:
                ax.spines["top"].set_visible(False)

            if i < n_channels - 1 or j == 0:
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

    if sampling_rate:
        indexes = np.linspace(
            0,
            total_duration - 1,
            int(sampling_rate * total_duration),
            dtype=int,
        )
        times = np.arange(total_duration, dtype=np.double) / time_scale
        solver_time = times[indexes]
        delta_t = np.diff(solver_time)[0]
        # Compare pulse with an interpolated pulse with 100 times more samples
        teff = np.arange(0, max(solver_time), delta_t / 100)

    for ch, (a, b) in ch_axes.items():
        basis = seq._channels[ch].basis
        t = np.array(data[ch]["time"]) / time_scale
        ya = data[ch]["amp"]
        yb = data[ch]["detuning"]
        if sampling_rate:
            t2 = 1
            ya2 = []
            yb2 = []
            for t_solv in solver_time:
                # Find the interval [t[t2],t[t2+1]] containing t_solv
                while t_solv > t[t2]:
                    t2 += 1
                ya2.append(ya[t2])
                yb2.append(yb[t2])
            cs_amp = CubicSpline(solver_time, ya2)
            cs_detuning = CubicSpline(solver_time, yb2)
            yaeff = cs_amp(teff)
            ybeff = cs_detuning(teff)

        t_min = -t[-1] * 0.03
        t_max = t[-1] * 1.05
        a.set_xlim(t_min, t_max)
        b.set_xlim(t_min, t_max)

        max_amp = np.max(ya)
        max_amp = 1 if max_amp == 0 else max_amp
        amp_top = max_amp * 1.2
        a.set_ylim(-0.02, amp_top)
        det_max = np.max(yb)
        det_min = np.min(yb)
        det_range = det_max - det_min
        if det_range == 0:
            det_min, det_max, det_range = -1, 1, 2
        det_top = det_max + det_range * 0.15
        det_bottom = det_min - det_range * 0.05
        b.set_ylim(det_bottom, det_top)

        a.plot(t, ya, color="darkgreen", linewidth=0.8)
        b.plot(t, yb, color="indigo", linewidth=0.8)
        if sampling_rate:
            a.plot(teff, yaeff, color="darkgreen", linewidth=0.8)
            b.plot(teff, ybeff, color="indigo", linewidth=0.8, ls="-")
            a.fill_between(teff, 0, yaeff, color="darkgreen", alpha=0.3)
            b.fill_between(teff, 0, ybeff, color="indigo", alpha=0.3)
        else:
            a.fill_between(t, 0, ya, color="darkgreen", alpha=0.3)
            b.fill_between(t, 0, yb, color="indigo", alpha=0.3)
        a.set_ylabel(r"$\Omega$ (rad/µs)", fontsize=14, labelpad=10)
        b.set_ylabel(r"$\delta$ (rad/µs)", fontsize=14)

        if draw_phase_area:
            top = False  # Variable to track position of box, top or center.
            draw_phase = any(
                seq_.type.phase != 0
                for seq_ in seq._schedule[ch]
                if isinstance(seq_.type, Pulse)
            )
            for pulse_num, seq_ in enumerate(seq._schedule[ch]):
                # Select only `Pulse` objects
                if isinstance(seq_.type, Pulse):
                    if sampling_rate:
                        area_val = (
                            np.sum(
                                cs_amp(
                                    np.arange(seq_.ti, seq_.tf) / time_scale
                                )
                            )
                            * 1e-3
                            / np.pi
                        )
                    else:
                        area_val = seq_.type.amplitude.integral / np.pi
                    phase_val = seq_.type.phase
                    x_plot = (seq_.ti + seq_.tf) / 2 / time_scale
                    if (
                        seq._schedule[ch][pulse_num - 1].type == "target"
                        or not top
                    ):
                        y_plot = np.max(seq_.type.amplitude.samples) / 2
                        top = True  # Next box at the top.
                    elif top:
                        y_plot = np.max(seq_.type.amplitude.samples)
                        top = False  # Next box at the center.
                    area_fmt = (
                        r"A: $\pi$"
                        if round(area_val, 2) == 1
                        else fr"A: {area_val:.2g}$\pi$"
                    )
                    if not draw_phase:
                        txt = area_fmt
                    else:
                        phase_fmt = fr"$\phi$: {phase_str(phase_val)}"
                        txt = "\n".join([phase_fmt, area_fmt])
                    a.text(
                        x_plot,
                        y_plot,
                        txt,
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=area_ph_box,
                    )

        target_regions = []  # [[start1, [targets1], end1],...]
        for coords in data[ch]["target"]:
            targets = list(data[ch]["target"][coords])
            tgt_strs = [str(q) for q in targets]
            tgt_txt_y = max_amp * 1.1 - 0.25 * (len(targets) - 1)
            tgt_str = "\n".join(tgt_strs)
            if coords == "initial":
                x = t_min + t[-1] * 0.005
                target_regions.append([0, targets])
                if seq._channels[ch].addressing == "Global":
                    a.text(
                        x,
                        amp_top * 0.98,
                        "GLOBAL",
                        fontsize=13,
                        rotation=90,
                        ha="left",
                        va="top",
                        bbox=q_box,
                    )
                else:
                    a.text(
                        x,
                        tgt_txt_y,
                        tgt_str,
                        fontsize=12,
                        ha="left",
                        bbox=q_box,
                    )
                    phase = seq._phase_ref[basis][targets[0]][0]
                    if phase and draw_phase_shifts:
                        msg = r"$\phi=$" + phase_str(phase)
                        a.text(
                            0,
                            max_amp * 1.1,
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
                phase = seq._phase_ref[basis][targets[0]][tf * time_scale + 1]
                a.axvspan(ti, tf, alpha=0.4, color="grey", hatch="//")
                b.axvspan(ti, tf, alpha=0.4, color="grey", hatch="//")
                a.text(
                    tf + t[-1] * 5e-3,
                    tgt_txt_y,
                    tgt_str,
                    ha="left",
                    fontsize=12,
                    bbox=q_box,
                )
                if phase and draw_phase_shifts:
                    msg = r"$\phi=$" + phase_str(phase)
                    wrd_len = len(max(tgt_strs, key=len))
                    x = tf + t[-1] * 0.01 * (wrd_len + 1)
                    a.text(
                        x,
                        max_amp * 1.1,
                        msg,
                        ha="left",
                        fontsize=12,
                        bbox=ph_box,
                    )
        # Terminate the last open regions
        if target_regions:
            target_regions[-1].append(t[-1])
        for start, targets_, end in (
            target_regions if draw_phase_shifts else []
        ):
            start = cast(float, start)
            targets_ = cast(list, targets_)
            end = cast(float, end)
            # All targets have the same ref, so we pick
            q = targets_[0]
            ref = seq._phase_ref[basis][q]
            if end != total_duration - 1 or "measurement" not in data[ch]:
                end += 1 / time_scale
            for t_, delta in ref.changes(start, end, time_scale=time_scale):
                conf = dict(linestyle="--", linewidth=1.5, color="black")
                a.axvline(t_, **conf)
                b.axvline(t_, **conf)
                msg = "\u27F2 " + phase_str(delta)
                a.text(
                    t_ - t[-1] * 8e-3,
                    max_amp * 1.1,
                    msg,
                    ha="right",
                    fontsize=14,
                    bbox=ph_box,
                )

        if "measurement" in data[ch]:
            msg = f"Basis: {data[ch]['measurement']}"
            b.text(
                t[-1] * 1.025,
                det_top,
                msg,
                ha="center",
                va="center",
                fontsize=14,
                color="white",
                rotation=90,
            )
            a.axvspan(t[-1], t_max, color="midnightblue", alpha=1)
            b.axvspan(t[-1], t_max, color="midnightblue", alpha=1)
            a.axhline(0, xmax=0.95, linestyle="-", linewidth=0.5, color="grey")
            b.axhline(0, xmax=0.95, linestyle=":", linewidth=0.5, color="grey")
        else:
            a.axhline(0, linestyle="-", linewidth=0.5, color="grey")
            b.axhline(0, linestyle=":", linewidth=0.5, color="grey")

        if "interp_pts" in data[ch] and draw_interp_pts:
            all_points = data[ch]["interp_pts"]
            if "amplitude" in all_points:
                pts = np.array(all_points["amplitude"])
                a.scatter(pts[:, 0], pts[:, 1], color="darkgreen")
            if "detuning" in all_points:
                pts = np.array(all_points["detuning"])
                b.scatter(pts[:, 0], pts[:, 1], color="indigo")

    plt.show()
