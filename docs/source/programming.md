# Programming a neutral-atom QPU

Pulser establishes a framework for programming Quantum Processing Units (QPUs) based on neutral atoms (also named _cold-atom Quantum Processing Units_). In this page, you will learn:

- [**What mathematical objects are you programming with Pulser?**](programming.md#introduction) In quantum computing, the evolution of a quantum state is defined by an Hamiltonian. What is a quantum state in Pulser? What are the Hamiltonians that can be defined in Pulser?
- [**How to program these mathematical objects with Pulser?**](programming.md#writing-a-pulser-program) How do you define the quantum state in Pulser? How do you define the Hamiltonian? We give you a step-by-step guide on how to create a quantum program using Pulser.

## Introduction

### 1. Atoms encode the state

Neutral atoms store the quantum information in their energy levels (also known as [_eigenstates_](./conventions.md)). When only two eigenstates are used to encode information, each atom is a qubit. If these eigenstates are $\left|a\right>$ and $\left|b\right>$, then the state of an atom is described by $\left|\psi\right> = \alpha \left|a\right> + \beta \left|b\right>$, with $|\alpha|^2 + |\beta|^2 = 1$.

<details>

  <summary>Definition of a quantum state with multiple atoms</summary>

When multiple atoms are used, the state of the system is described by a linear combination of the _eigenstates_ of the multi-atom system, whose set is obtained by making the cross product of the set of eigenstate of each atom. If each atom is described by $d$ eigenstates labelled ${\left|a_1\right>, \left|a_2\right>...\left|a_d\right>}$ (each atom is a _qudit_), and if there are $N$ atoms in the system, then the state of the whole system is provided by 

$$
\begin{align}
\left|\Psi\right> &= \sum_{i_1, ..., i_N \in [1, ..., d]} c_{i_1, ..., i_N} \left|a_{i_1}...a_{i_N}\right> \\
&= c_{1, 1, ..., 1}\left|a_{1}a_{1}...a_{1}\right> + ... + c_{1, 1, ..., d}\left|a_{1}a_{1}...a_{d}\right> + ... + c_{d, d, ..., d}\left|a_{d}a_{d}...a_{d}\right>
\end{align}
$$

where $\sum_{i_1, ..., i_N \in [1, ..., d]} |c_{i_1, ..., i_N}|^2 = 1$. If $d=2$, then this becomes

$$
\left|\Psi\right> = c_{1, 1, ..., 1}\left|a_{1}a_{1}...a_{1}\right> + c_{1, 1, ..., 2}\left|a_{1}a_{1}...a_{2}\right> + ... + c_{2, 2, ..., 2}\left|a_{2}a_{2}...a_{2}\right>
$$

If $d=2$ and $N=1$, you have the state of a qubit as above $\left|\Psi\right> = c_{1}\left|a_{1}\right> + c_{2}\left|a_{2}\right>$.

</details>

<br>

### 2. Hamiltonian evolves the state

In quantum physics, the state of a quantum system evolves along time following the Schrödinger equation: 

$$i\frac{d\left|\Psi\right>(t)}{dt} = \frac{H(t)}{\hbar} \left|\Psi\right>(t)$$

Here $H(t)$ is the Hamiltonian describing the evolution of the system. For a system of atoms in the initial state $\left|\Psi_0\right>$, the final state of the system after a time $\Delta t$ is:

$$ \left|\Psi_f\right> = \exp\left(-\frac{i}{\hbar}\int_0^{\Delta t} H(t) dt\right)\left|\Psi_0\right>$$

The Hamiltonian describing the evolution of the system can be written as

$$
H(t) = \sum_i \left (H^D_i(t) + \sum_{j<i}H^\text{int}_{ij} \right),
$$

where $H^D_i$ is the driving Hamiltonian for atom $i$ and
$H^\text{int}_{ij}$ is the interaction Hamiltonian between atoms $i$
and $j$.

:::{important}
**Pulser provides you the tools to program this Hamiltonian**. 
:::

#### 2.1. Driving Hamiltonian

The driving Hamiltonian describes the effect of a pulse on two energy levels of an individual atom, $|a\rangle$ and $|b\rangle$. A pulse is determined by its duration $\Delta t$, its Rabi frequency $\Omega(t)$, its detuning $\delta(t)$ and its phase $\phi$ (constant along the duration of the pulse). Between $0$ and $\Delta t$, the driving Hamiltonian is:

$$
H^D(t) / \hbar = \frac{\Omega(t)}{2} e^{-j\phi} |a\rangle\langle b| + \frac{\Omega(t)}{2} e^{j\phi} |b\rangle\langle a| - \delta(t) |b\rangle\langle b|
$$

<details>

  <summary>Rotations on the Bloch sphere</summary>


In the Bloch sphere representation, this Hamiltonian describes a rotation around the axis $\overrightarrow{\Omega}(t) = (\Omega(t)\cos(\phi), -\Omega(t)\sin(\phi), -\delta(t))^T$, with angular velocity $\Omega_{eff}(t) = |\overrightarrow{\Omega}(t)| = \sqrt{\Omega^2(t) + \delta^2(t)}$. 

For a resonant pulse ($\delta(t)=0$) of duration $\Delta t$, we rotate of an angle $\int_0^{\Delta t} \Omega (t) dt$ around the fixed axis $(\cos(\phi), -\sin(\phi), 0)$ (on the equator of the Bloch sphere).

:::{figure} files/bloch_rotation_a_b.png
:align: center
:alt: Representation of the drive Hamiltonian's dynamics as a rotation in the Bloch sphere.
:width: 300

Representation of the drive Hamiltonian's dynamics as a rotation in the Bloch sphere. The coherent excitation is driven between a lower energy level, $|a\rangle$, and a higher energy level,
$|b\rangle$, with Rabi frequency $\Omega(t)$, detuning $\delta(t)$ and phase $\phi$.
:::

</details>

<br>

:::{important}
With Pulser, you program the driving Hamiltonian by setting $\Omega(t)$, $\delta(t)$ and $\phi(t)$, all the while Pulser ensures that you respect the constraints of your chosen device.
:::

#### 2.2. Interaction Hamiltonian

The interaction Hamiltonian depends on the distance between the atoms $i$ and $j$, $R_{ij}$, and the energy levels in which the information is encoded in these atoms, that define the interaction between the atoms $\hat{U}_{ij}$

$$
H^\text{int}_{ij} = \hat{U}_{ij}(R_{ij})
$$

The interaction operator $\hat{U}_{ij}$ is composed of an entangling operator and an interaction strength.

:::{note}
The interaction Hamiltonian is constant over time. It is always on, no matter the values of the drive Hamiltonian (even if the values of the parameters $\Omega$, $\delta$, $\phi$ are equal to $0$ over a time $\Delta t$).
:::

##### Ising Hamiltonian

If the Rydberg state $\left|r\right>$ is involved in the computation, then 

$$
\hat{U}_{ij}(R_{ij}) = \frac{C_6}{R_{ij}^6} \hat{n}_i \hat{n}_j
$$
<details>

  <summary>Interaction strength and entangling operator</summary>

- The interaction strength is $\frac{C_6}{R_{ij}^6}$, with $C_6$ the Ising interaction coefficient that depends on the principal quantum number of the Rydberg state.
- The entangling operator between atom $i$ and $j$ is $\hat{n}_i\hat{n}_j = |r\rangle\langle r|_i |r\rangle\langle r|_j$. 

</details>

Together with the driving Hamiltonian, this interaction encodes the _Ising Hamiltonian_ and is the **most common choice in neutral-atom devices.**

##### XY Hamiltonian

If the information is stored in the Rydberg states $\left|0\right>$ and $\left|1\right>$ (the so-called `XY` basis), then

$$
\hat{U}_{ij}(R_{ij}) =\frac{C_3}{R_{ij}^3} (|1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j)
$$
<details>

  <summary>Interaction strength and entangling operator</summary>

- The interaction strength is $\frac{C_3}{R_{ij}^3}$, with $C_3$ a coefficient that depends on the energy levels used to encode $\left|0\right>$ and $\left|1\right>$. 
- The entangling operator between atom $i$ and $j$ is $\hat{\sigma}_i^{+}\hat{\sigma}_j^{-} + \hat{\sigma}_i^{-}\hat{\sigma}_j^{+} = |1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j$. 

</details>

This interaction hamiltonian is associated with the _XY Hamiltonian_ and is a less common mode of operation, usually accessible only in select neutral-atom devices.

:::{important}
With Pulser, you program the interaction Hamiltonian by setting the distance between atoms, $R_{ij}$. Additionally, the choice of eigenstates used in the computation and the Rydberg level(s) targeted by the device fully determine the interaction strength. Most commonly, the ground and Rydberg states ($\left|g\right>$ and $\left|r\right>$) are used, such that   

$$H^\text{int}_{ij} = \frac{C_6}{R_{ij}^6} \hat{n}_i\hat{n}_j$$

When providing the distance between atoms, Pulser ensures that you respect the constraints of your chosen device.
:::

## Writing a Pulser program

As outlined above, Pulser lets you program an Hamiltonian ([the Hamiltonian $H$](programming.md#2-hamiltonian-evolves-the-state)) so that you can manipulate the quantum state of a system of atoms. The series of necessary instructions is encapsulated in the so-called Pulser `Sequence`. Here is a step-by-step guide to create your own Pulser `Sequence`.

### 1. Pick a Device


:::{figure} files/decision_diagram_device.png
:align: center
:alt: Decision Diagram to select a Device for the computation
:width: 600
:::

The `Device` you select will dictate some parameters and constrain others. For instance, the value of the $C_6$ and $C_3$ coefficients of the [interaction Hamiltonian](programming.md#22-interaction-hamiltonian) are defined by the device. Notably, the `Device` defines the list of `Channels` that can be used in the computation, which have a direct impact on the Hamiltonian that can be implemented. For a complete view of the constraints introduced by the device, [check its description](./apidoc/core.rst).

### 2. Create the Register

The `Register` defines the position of the atoms. This determines:

- the number of atoms to use in the quantum computation, i.e, the size of the system (let's note it $N$).
- the distance between the atoms, the $R_{ij}\ (1\le i, j\le N)$ parameters in the [interaction Hamiltonian](programming.md#22-interaction-hamiltonian).

### 3. Pick the Channels

A `Channel` targets the transition between two energy levels. Therefore, picking channels defines the energy levels that will be used in the computation. The channels must be picked from the `Device.channels`, so your device selection should take into account the channels it supports.

Picking the channel will initialize the state of the system, and fully determine the [interaction Hamiltonian](programming.md#22-interaction-hamiltonian):

- If the selected Channel is the `Rydberg` or the `Raman` channel, the system is initialized in $\left|gg...g\right>$ and the interaction Hamiltonian is the [Ising Hamiltonian](programming.md#ising-hamiltonian)

$$H^\text{int}_{ij} =\frac{C_6}{R_{ij}^6}|r\rangle\langle r|_i |r\rangle\langle r|_j$$

- If the selected Channel is the `Microwave` channel, the system is initialized in $\left|00...0\right>$ and the interaction Hamiltonian is the [XY Hamiltonian](programming.md#xy-hamiltonian)

$$H^\text{int}_{ij} =\frac{C_3}{R_{ij}^3}|1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j$$

:::{important}
At this stage, the [interaction Hamiltonian](programming.md#22-interaction-hamiltonian) is fully determined.
:::

A `Channel` is also characterized by its addressing, which defines the number of atoms that are going to be targeted by a pulse. If the addressing of a `Channel` is `Global`, all the atoms will experience the same pulse targetting the same transition. In the [Hamiltonian $H$](programming.md#2-hamiltonian-evolves-the-state), all the driving Hamiltonians $H^D_i$ are expressed as

$$
H^D_i(t) / \hbar = \frac{\Omega(t)}{2} e^{-j\phi(t)} |a\rangle_i \langle b|_i + \frac{\Omega(t)}{2} e^{j\phi(t)} |b\rangle_i\langle a|_i - \delta(t) |b\rangle_i\langle b|_i
$$

If the addressing of a `Channel` is `Local`, then only certain atoms (the "targets") will experience the pulse and have their evolution follow $H^D_i$. The driving Hamiltonian of the other atoms is $H^D_i = \hat{0}_i$. The Hamiltonian $H$ can also be rewritten:

$$
H(t) = \sum_{i \in targets} H^D_i(t) + \sum_i \sum_{j<i}H^\text{int}_{ij},
$$

:::{important}
The addressing of a `Channel` defines the number of atoms whose transition will be targeted by a pulse.

The most common addressing for a `Channel` is the `Global` one: all the atoms evolve under the same [driving Hamiltonian](programming.md#21-driving-hamiltonian). 
:::

### 4. Add the Pulses

By adding pulses to a channel, we incrementally construct the [driving Hamiltonian](programming.md#21-driving-hamiltonian):
- Each `Pulse` defines the Rabi frequency $\Omega(t)$, the detuning $\delta(t)$ and the phase $\phi$ of the driving Hamiltonian over a duration $\Delta t$. Similarly, a delay sets all these parameters to zero for a given amount of time.
- The channel dictates the states $\left|a\right>$ and $\left|b\right>$ of the driving Hamiltonian.

By applying a series of pulses and delays, one defines the entire driving Hamiltonian of each atom over time.

## Conclusion

We have successfully defined the [Hamiltonian](programming.md#2-hamiltonian-evolves-the-state) $H$ describing the evolution of the system over time, by:
- Picking a `Device`, which defines the value of the $C_6$ or $C_3$ coefficients.
- Creating a `Register` of atoms, which defines the number of atoms used and the distance between them, $R_{ij}$.
- Selecting the `Channels` of the `Device` to use, which define the energy levels of the atoms to use - this step completely defines the [interaction Hamiltonian](programming.md#22-interaction-hamiltonian). The addressing property of each `Channel` also dictates the atoms that will be targeted by the `Pulse`.
- Adding `Pulse` and delays to the `Channel`s defines the [driving Hamiltonian](programming.md#21-driving-hamiltonian) of each atom along time.    

You can now simulate your first Hamiltonian by programming your first `Sequence`! [In this tutorial](tutorials/creating.nblink), you will simulate the evolution of the state of an atom initialized in $\left|g\right>$ under a Hamiltonian $H(t)=\frac{\Omega(t)}{2} |g\rangle \langle r|+\frac{\Omega(t)}{2} |r\rangle\langle g|$, with $\Omega$ chosen such that the final state of the atom is the excited state $\left|r\right>$.

Many concepts have been introduced here and you might want further explanations.
- The `Device` object contains all the constraints and physical quantities that are defined in a QPU. [This section in the fundamentals](apidoc/core.rst) details these and provides examples of `Devices`. The `VirtualDevices` were also mentioned in this document ([here](programming.md#1-pick-a-device)), which is a more advanced feature described [here](tutorials/virtual_devices.nblink).
- There are multiple ways of defining a `Register`, as is further detailed [in this section](tutorials/reg_layouts.nblink).
- The energy levels associated with each `Channel` and the interaction Hamiltonian they implement are summed up in [the conventions page](conventions.md). The channels contain lots of constraints and physical informations, they are detailed in [the same section as the `Device`](apidoc/core.rst).
- The quantities in a `Pulse` are defined using `Waveform`s, you can read more about these [on this page](tutorials/composite_wfs.nblink).