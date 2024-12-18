# Programming a neutral-atom Quantum Computer

## Introduction

### 1. Atoms encode the state

Neutral atoms store the quantum information in their energy levels (also known as [_eigenstates_](./conventions.md#Bases)). When only two eigenstates are used to encode information, each atom is a qubit. If these eigenstates are $\left|a\right>$ and $\left|b\right>$, then the state of an atom is described by $\left|\psi\right> = \alpha \left|a\right> + \beta \left|b\right>$, with $|\alpha|^2 + |\beta|^2 = 1$.

When multiple atoms are used, the state of the system is described by a linear combination of the _eigenstates_ of the multi-atom system, whose set is obtained by making the cross product of the set of eigenstate of each atom. If each atom is described by $d$ eigenstates labelled ${\left|a_1\right>, \left|a_2\right>...\left|a_d\right>}$ (each atom is a _qudit_), and if there are $N$ atoms in the system, then the state of the whole system is provided by $\left|\Psi\right> = \sum_{i_1, ..., i_N \in [1, ..., d]} c_{i_1, ..., i_N} \left|a_{i_1}...a_{i_N}\right>$, with $\sum_{i_1, ..., i_N \in [1, ..., d]} |c_{i_1, ..., i_N}|^2 = 1$.

### 2. Hamiltonian evolves the state

In quantum physics, the state of a quantum system evolves along time following the Schrödinger equation: $i\frac{d\left|\Psi\right>(t)}{dt} = \frac{H(t)}{\hbar} \left|\Psi\right>(t)$, where $H(t)$ is the Hamiltonian describing the evolution of the system. 

**Pulser provides you the tools to program this Hamiltonian**. 

#### 2.1. The Hamiltonian

The Hamiltonian describing the evolution of the system can be written as

$$
H(t) = \sum_i \left (H^D_i(t) + \sum_{j<i}H^\text{int}_{ij} \right),
$$

where $H^D_i$ is the driving Hamiltonian for atom $i$ and
$H^\text{int}_{ij}$ is the interaction Hamiltonian between atoms $i$
and $j$.

#### 2.2. Driving Hamiltonian

The driving Hamiltonian describes the effect of a pulse of Rabi frequency $\Omega(t)$, detuning $\delta(t)$ and phase $\phi(t)$ on an individual atom, driving the transition
between two of its energies levels, $|a\rangle$ and $|b\rangle$.

:::{figure} files/two_level_ab.png
:align: center
:alt: The energy levels for the driving Hamiltonian.
:width: 200

The coherent excitation is driven between a lower energy level, $|a\rangle$, and a higher energy level,
$|b\rangle$, with Rabi frequency $\Omega(t)$ and detuning $\delta(t)$.
:::

$$
H^D(t) / \hbar = \frac{\Omega(t)}{2} e^{-i\phi(t)} |a\rangle\langle b| + \frac{\Omega(t)}{2} e^{i\phi(t)} |b\rangle\langle a| - \delta(t) |b\rangle\langle b|
$$

:::{important}
With Pulser, you program the driving Hamiltonian by setting $\Omega(t)$, $\delta(t)$ and $\phi(t)$, all the while Pulser ensures that you respect the constraints of your chosen device.
:::

#### 2.3. Interaction Hamiltonian


The interaction Hamiltonian depends on the distance between the atoms $i$ and $j$, $R_{ij}$, and the energy levels in which the information is encoded in these atoms, that define the interaction between the atoms $\hat{U}_{ij}$

$$
H^\text{int}_{ij} = \hat{U}_{ij}(R_{ij})
$$

The interaction operator $\hat{U}_{ij}$ is composed of an entangling operator and an interaction strength:
- If the Rydberg state $\left|r\right>$ is involved in the computation, then $\hat{U}_{ij}(R_{ij}) = \frac{C_6}{R_{ij}^6} \hat{n}_i \hat{n}_j$, where the interaction strength is $\frac{C_6}{R_{ij}^6}$, with $C_6$ a coefficient that depends on the principal quantum number of the Rydberg state. The entangling operator between atom $i$ and $j$ is $\hat{n}_i\hat{n}_j = |r\rangle\langle r|_i |r\rangle\langle r|_j$. Together with the driving Hamiltonian, this interaction encodes the _Ising Hamiltonian_ and is the **most common choice in neutral-atom devices.**
- If the information is stored in the $\left|0\right>$ and $\left|1\right>$ states of the `XY` basis, $\hat{U}_{ij}(R_{ij}) =\frac{C_3}{R_{ij}^3} (|1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j)$, where the interaction strength is $\frac{C_3}{R_{ij}^3}$, with $C_3$ a coefficient that depends on the energy levels used to encode $\left|0\right>$ and $\left|1\right>$. The entangling operator between atom $i$ and $j$ is $\hat{\sigma}_i^{+}\hat{\sigma}_j^{-} + \hat{\sigma}_i^{-}\hat{\sigma}_j^{+} = |1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j$. This interaction hamiltonian is associated with the _XY Hamiltonian_ and is a less common mode of operation, usually accessible only in select neutral-atom devices.

:::{important}
With Pulser, you program the interaction Hamiltonian by setting the distance between atoms, $R_{ij}$, and the choice of eigenstates used in the computation and the Rydberg level(s) targeted by the device. Most commonly, the ground and Rydberg states ($\left|g\right>$ and $\left|r\right>$) are used, such that
$$
H^\text{int}_{ij} = \frac{C_6}{R_{ij}^6} \hat{n}_i\hat{n}_j
$$
When providing the distance between atoms, Pulser ensures that you respect the constraints of your chosen device.
:::

#### 2.4. Evolution of the system's state

For a system of atoms in the initial state $\left|\Psi_0\right>$, the final state of the system after a time $\Delta t$ is:
$$ \left|\Psi_f\right> = \exp\left(-\frac{i}{\hbar}\int_0^{\Delta t} H(t) dt\right)$$

## Writing a Pulser program

As outlined above, Pulser lets you program an Hamiltonian so that you can manipulate the quantum state of a system of atoms, that encode the states. The series of necessary instructions is encapsulated in the so-called Pulser `Sequence`. Here is a step-by-step guide to create your own Pulser `Sequence`.

### 1. Pick a Device

```{mermaid}

flowchart TB
  A[Picking a Device] --> B{Do I want to run on a QPU}
  B -->|Yes| C[Pick the Device from the QPU backend]
  B -->|No| D[Do I want to be constrained by all the QPU specs ?]
  D -->|Yes| E[Use a `Device` (like `pulser.AnalogDevice`)]
  D --> |No| F[Use a `VirtualDevice` (like `pulser.MockDevice`)]

```

The `Device` you select will dictate some parameters and constrain others. For instance, the value of the $C_6$ and $C_3$ coefficients of the interaction Hamiltonian are defined by the device. For a complete view of the constraints introduced by the device, [check its description](./apidoc/core.rst#structure-of-a-device).

### 2. Create the Register

The `Register` defines the position of the atoms. This determines:

- the number of atoms to use in the quantum computation, i.e, the size of the system (let's note it $N$).
- the distance between the atoms, the $R_{ij} (1\le i, j\le N)$ parameters in the interaction Hamiltonian.

### 3. Pick the Channels

A `Channel` targets the transition between two energy levels. Therefore, picking channels defines the energy levels that will be used in the computation. The channels must be picked from the `Device.channels`, so your device selection should take into account the channels it supports.

Picking the channel will initialize the state of the system, and fully determine the interaction Hamiltonian:
- If the selected Channel is the `Rydberg` or the `Raman` channel, the system is initialized in $\left|gg...g\right>$ and the interaction Hamiltonian is $H^\text{int}_{ij} =\frac{C_6}{R_{ij}^6}|r\rangle\langle r|_i |r\rangle\langle r|_j$.
- If the selected Channel is the `Microwave` channel, the system is initialized in $\left|00...0\right>$ and the interaction Hamiltonian is $H^\text{int}_{ij} =\frac{C_3}{R_{ij}^3}|1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j$.

Note: It is possible to pick a `Rydberg` and a `Raman` channel, the information will then be encoded in 3 energy levels $\left|r\right>$, $\left|g\right>$ and $\left|h\right>$: the atoms are no-longer qubits but qutrits. However, it is not possible to pick the `Microwave` channel and another channel.

At this stage, the interaction strength of the Hamiltonian is fully determined.

### 4. Add the Pulses

By adding pulses to a channel, we incrementally construct the driving Hamiltonian:
- Each `Pulse` defines the Rabi frequency $\Omega(t)$, the detuning $\delta(t)$ and the phase $\phi$ of the driving Hamiltonian over a duration $\Delta t$. Similarly, a delay sets all these parameters to zero for a given amount of time.
- The channel dictates the states $\left|a\right>$ and $\left|b\right>$ of the driving Hamiltonian.

By applying a series of pulses and delays, one defines the entire driving Hamiltonian of each atom over time.

**Conclusion**: We have successfully defined the hamiltonian $H$ describing the evolution of the system over time.