# Neutral-atom Hardware

## How does a neutral-atom Quantum Computer work ?

A neutral-atom Quantum Computer is composed of neutral-atoms (typically, atoms of Rubidium or Cesium that belong to the first column of the Mendeleiev table, but quantum computing platforms have also been built using atoms of Ytterbium and Dysprosium that belong to the second column of the Mendeleiev table). To perform quantum computations with these atoms, we "cool" them down to very small temperatures (typically, $`10\mu`K) using lasers. Note: The neutral-atoms are in fact slowed down, and the temperature here is associated to their speed. Neutral-atom Quantum Computers are sometimes named cold-atom Quantum Computers because the atoms are "cold", ie there associated temperature is small.

Since these atoms are cold, they can be trapped in space using another kind of lasers (see the `RegisterLayout` object in the extended section). They can then be rearranged and re-trapped in a specific pattern, that we name the register (see the `Register` section of the fundamentals). The atoms spend enough time trapped such that we are sure they are all in the ground state (the initial state for the computations, see the [conventions page](conventions.md)).

The computation in itself can then happen: a serie of laser pulses are applied on the atoms to modify their state. The laser pulses can target different transition, each transition being associated to a different laser channel (see below).

Once the computation is over, the state of the atoms is measured. Here as well, this is done using lasers: some light is shine on the atoms, the atoms that re-emit this light are measured in the ground-state and are labelled `0`, the others are described as being in an excited state and are labelled `1` (see the [convention page](conventions.md) for more). 

## Physical parameters to take into account in quantum computations

As sketched in the above section, making a quantum computation with a neutral-atom device involves multiple steps and the use of multiple lasers. At each step and laser used, the physics of the device constrains the displacement of the atoms. In `Pulser`, all the physical constrains are stored in the `Device` object.

### Device

The `Device` class stores all the physical constrains the quantum computations should match. To perform any computation using Pulser, it is necessary to provide a device. For convenience, some examples of typical physical devices are included and can be accessed via `pulser.devices`. These devices are instances of the `Device` class. They are constrained by physical considerations and all their parameters are defined. An example of such a `Device`, the `pulser.AnalogDevice`, is provided below.

If we go back to the timeline of a quantum computation using a neutral-atom device, the first step is the trapping of atoms in space and their rearrangement into a defined register. The first constraints concern the definition of the layout of traps and the register. They can be defined in 2D or 3D (check the `dimensions` attribute) but the register of atoms must be a subset of the layout of traps. Therefore, some constrains apply to both: the distance between each traps/atoms must be greater than the distance `min_atom_distance`, the traps/atoms should not be defined at a distance greater than `max_radial_distance` from the center.

The register is a subset of the number of traps because by default some stochastical effects are at play, such that in theory a trap has a probability 1/2 to be filled by an atom. That way, there is a maximum number of traps of the layout that can be filled by an atom from the register (given by `max_layout_filling`), and there is also an optimal for this filling, (given by `optimal_layout_filling`).

In terms of performance, there is a maximum number of atoms that can be trapped and rearranged in the register (`max_atom_num`) and a maximum number of traps that can be defined in the layout (`max_layout_traps`). From the considerations above, these two numbers are different. There is also a minimum number of traps to have in the layout (`min_layout_traps`).

The considerations about the layout are unnessary if the device does not require one to be associated with the `Register`, which is given by the `requires_layout` parameter. If `requires_layout` is True, then the `Register` must have a layout associated to it. We invite you to check the layout section in the advanced features to see how to attach a layout to a register in detail. Generating a layout spatially is a complex operation that needs some calibrations. Some devices provide a bank of pre-calibrated layouts in the `pre_calibrated_layouts` attribute, for which the calibration has already been performed (meaning, the computations will be faster than for a non-calibrated layout). If the device does not accept new layouts (`accepts_new_layouts` is False), then the register must be a subset of one of the pre-calibrated layouts.

Now that we have seen all the constraints regarding the definition of the register of atoms, the next step of a quantum computation is to apply a sequence of pulses targeting a specific transition using `Channels`. The list of channels available in the device are accessible via the `channels` property. Some specific `Channels` are the `DetuningMapModulator` channels, that are listed in the `dmm_channels` property. These specific channels are further detailed in [their section in the extended features](tutorials/dmm.nblink). They are associated with the concept of SLM mask that some device supports (the `supports_slm_mask` attribute, check [the section on SLM](tutorials/slm_mask.nblink) to learn more).

The physical constraints applied by a `Channel` are described below. More globally than a channel, the duration of the sequence of pulses should not be longer than `max_sequence_duration`. When a pulse is applied on a `Channel`, the atoms can interact between each other depending on the transition targeted by the `Channel`. When the channel aims at exciting the atoms to the rydberg state, then the Rydberg coefficient is defined from a `rydberg_level` defined in the device (see the [conventions page](conventions.md)). When using the `Microwave` channel, the `C_3` coefficient is defined by the `interaction_coeff_xy`.

The device can also define some parameters used at execution on the backend. A backend can be a QPU (it should then have an associated device) or an emulator. In either case, the maximum number of runs, that is, the maximum number of times the cycle register-pulser-measurement can be performed for a quantum program, can be defined in the device (`max_runs`). The device also contains all the information about the noise to model the device (like, the temperature of the atoms) in `default_noise_model`. You can read more about the backends and the noise model in the [backend section](./tutorials/backends.nblink).

### Channels

Each pulse modifying the state of the atoms is applied via a laser channel. A channel targets a specific transition between two states, named the eigenstates. These two eigenstates form a basis. You can check the basis currently implemented in Pulser and their associated eigenstates in the [conventions page](./conventions.md). A laser channel can address the atoms globally (all the atoms are addressed at the same time) or locally (atoms are targeted separately). This is described by the `addressing` attribute of the `Channel` object.

The pulses that are applied on these laser channels are defined by four parameters: their duration, amplitude, detuning and phase (learn more about pulses [here](./apidoc/core.rst)). The laser channels constrain these parameters: 
- the duration must to be longer than `min_duration` but shorter than `max_duration`. It also has to be a multiple of `clock_duration`. 
- the amplitude must be between 0 and `max_amp`. If the amplitude of a pulse is not constant equal to zero, its average must be higher than `min_avg_amp`.
- the detuning must be between -`max_abs_detuning` and `max_abs_detuning`.

When using channels having a local `addressing`, other constraints emerge. For instance, there is a maximum number of atoms that can be targeted by each pulse (`max_targets`). Changing the atoms targeted by the channel takes also some time. This time must be greater than `min_retarget_interval` and is sometimes forced to be `fixed_retarget_t`. However, these channels are hard to implement in hardware and a prefered way to program local channels must be the `DMM` channels, `DMM` meaning Detuning-Map Modulator, which is explained in [another section](./tutorials/dmm.nblink).

All the channels have a modulation bandwidth (`mod_bandwidth`), which impacts the shape of the pulses programmed by the users. This arises from the optical elements we use to control the amplitude, detuning, phase of the pulses. We detail the influence of this modulation bandwidth in the [extended features](./tutorials/output_mod_eom.nblink), but it means that the effective duration of the pulses seen by the atoms is longer than what was programmed. A remedy to this can be to use other optical elements, that have a longer modulation bandwidth (i.e. better response time) but can only take constant pulses. These optical elements are named the `EOM`, and one can activate an "EOM mode" on a channel that has an `eom_config`. To check whether or not a channel supports this operation, one can call the `supports_eom()` method of the `Channel`. To read more about EOM and its configuration in a `Channel`, please refer to the [EOM section](./tutorials/output_mod_eom.nblink).

Finally, the laser channel applied on the atoms has a propagation direction (`propagation_dir`). This information is only relevant for emulation purposes. 

### Conclusion and extension

The `Device` object stores all the physical information to take into account when making a quantum computation using a neutral-atom Quantum Processing Unit. An example of such a `Device` is detailed below with the `AnalogDevice`. As we have seen above, there are lots of physical constraints that are taken into account when you add an operation to your `Sequence`. However, for development perspectives, you could want to relax some conditions: what if there was no condition on the duration of the sequence ? or on the amplitude of the pulses applied to a specific channel ? You can work with a `VirtualDevice`, a kind of device that allows some of the physical constraints to not be defined. Check the [section on `VirtualDevice`](./tutorials/virtual_devices.nblink) to learn more.










