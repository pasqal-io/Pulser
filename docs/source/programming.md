# Programming a neutral-atom Quantum Computer

## Introduction

1. Atoms encode the state

Neutral atoms store the quantum information in thier energy levels. If only 2 energy levels of an atom are being used, the atom is a qubit. If these eigenstates are $\left|a\right>$ and $\left|b\right>$, then the state of the atom is described by $\left|\psi\right> = \alpha \left|a\right> + \beta \left|b\right>$, with $|\alpha|^2 + |\beta|^2 = 1$.

When multiple atoms are used, the state describing this multi-atom system is the tensor network of the states. Say these atoms are labeled $\[1, 2, ..., N\]$, then the state of the system is $\left|\Psi\right> = \left|\psi_1\right> \otimes \left|\psi_2\right> \otimes ... \otimes \left|\psi_N\right>$.

2. Hamiltonian evolves the state

In quantum physics, the state of a system of atoms evolve along time following the Schrödinger equation: $i\frac{d\left|\Psi\right>(t)}{dt} = \frac{H(t)}{\hbar} \left|\Psi\right>(t)$. $H(t)$ is the Hamiltonian describing the evolution of the system. **Pulser provides you the tools to program this Hamiltonian**. 

2.1. The Hamiltonian

The Hamiltonian describing the evolution of the system can be written as

$$
H(t) = \sum_i \left (H^D_i(t) + \sum_{j<i}H^\text{int}_{ij} \right),
$$

where $H^D_i$ is the driving Hamiltonian for atom $i$ and
$H^\text{int}_{ij}$ is the interaction Hamiltonian between atoms $i$
and $j$.

2.1. Driving Hamiltonian

The driving Hamiltonian describes the effect of a pulse of Rabi frequency $\Omega(t)$, detuning $\delta(t)$ and phase $\phi(t)$ on an individual atom, exciting the transition
between two of its energies levels, $|a\rangle$ and $|b\rangle$.


:::{figure} files/two_level_ab.png
:align: center
:alt: The energy levels for the driving Hamiltonian.
:width: 200

$$
H^D(t) / \hbar = \frac{\Omega(t)}{2} e^{-i\phi(t)} |a\rangle\langle b| + \frac{\Omega(t)}{2} e^{i\phi(t)} |b\rangle\langle a| - \delta(t) |b\rangle\langle b|
$$

2.2. Interaction Hamiltonian

The interaction Hamiltonian depends on the distance between the atoms $i$ and $j$, $R_{ij}$, and the energy levels in which the information is encoded in these atoms, that define the interaction between the atoms $\hat{E}_{ij}$

$$
H^\text{int}_{ij} = \hat{E}_{ij}(R_{ij})
$$

The interaction operator $\hat{E}_{ij}$ is composed of an entangling operator and an interaction strength:
- If the rydberg state $\left|r\right>$ is involved in the computation, then $\hat{E}_{ij}(R_{ij}) = \frac{C_6}{R_{ij}^6} |r\rangle\langle r|_i |r\rangle\langle r|_j$, where the interaction strength is $\frac{C_6}{R_{ij}^6}$, with $C_6$ a coefficient that can be programmed using Pulser. The entangling operator between atom $i$ and $j$ is $\hat{n}_i\hat{n}_j = |r\rangle\langle r|_i |r\rangle\langle r|_j$.
- If the information is stored in the $\left|0\right>$ and $\left|1\right>$ states of the `XY` basis, $\hat{E}_{ij}(R_{ij}) =\frac{C_3}{R_{ij}^3} (|1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j)$, where the interaction strength is $\frac{C_3}{R_{ij}^3}$, with $C_3$ a coefficient that can be programmed using Pulser. The entangling operator between atom $i$ and $j$ is $\hat{\sigma}_i^{+}\hat{\sigma}_j^{-} + \hat{\sigma}_i^{-}\hat{\sigma}_j^{+} = |1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j$.

2.3. Evolution of the system's state

For a system of atoms in the initial state $\left|\Psi_0\right>$, the final state of the system after a time $\Delta t$ is:
$$ \left|\Psi_f\right> = \exp\left(-\frac{i}{\hbar}\int_0^{\Delta t} H(t) dt\right)$$

## Recipe

As underlined above, Pulser lets the user program an Hamiltonian to modify the state of a system of atoms, that encode the states. What do you define at each step of a quantum program in Pulser ? Let's go over these steps.

1. Pick a Device

```{mermaid}

flowchart TB
  A[Picking a Device] --> B{Do I want to run on a QPU}
  B -->|Yes| C[Pick the Device from the [QPU backend](./tutorials/backends.nblink)]
  B -->|No| D[Do I want to mimic all the QPU's constraints ?]
  D -->|Yes| E[Use a `Device` (like `pulser.AnalogDevice`)]
  D --> |No| F[Use a [`VirtualDevice`](./tutorials/virtual_devices.nblink) (like `pulser.MockDevice`)]

```

The `Device` you select will dictate some parameters and constraint others. For instance, the value of the $C_6$ and $C_3$ coefficients of the Interaction Hamiltonian are defined by the device. For a complete description of the constraints introduced by the device, [check its description](./apidoc/core.rst).

2. Create the Register

The Register defines the position of the atoms. This determines:

- the number of atoms to use in the quantum computation, i.e, the size of the system (let's note it $N$).
- the distance between the atoms, the $R_{ij} (1\le i, j\le N)$ parameters in the interaction Hamiltonian.

At this stage, the interaction strength of the Hamiltonian is fully determined.

3. Pick the Channels

This defines the energy levels that will be used in the computation. A Channel targets the transition between two energy levels. The Channel must be picked from the `Device.channels`: select your device for it to have the channels you need.

Picking the channel will initialize the state of the system, and fully determine the Interaction Hamiltonian:
- If the selected Channel is the `Rydberg` or the `Raman` channel, the system is initialized in $\left|gg...g\right>$ and the interaction Hamiltonian is $H^\text{int}_{ij} =\frac{C_6}{R_{ij}^6}|r\rangle\langle r|_i |r\rangle\langle r|_j$.
- If the selected Channel is the `Microwave` channel, the system is initialized in $\left|11...1\right>$ and the interaction Hamiltonian is $H^\text{int}_{ij} =\frac{C_3}{R_{ij}^3}|1\rangle\langle 0|_i |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_j$.

Note: It is possible to pick a `Rydberg` and a `Raman` channel, the information will then be encoded in 3 energy levels $\left|r\right>$, $\left|g\right>$ and $\left|h\right>$: the atoms are no-longer qubits but qutrits. However, it is not possible to pick the `Microwave` channel and another channel.

4. Add the Pulses

By adding a pulse to a channel, one defines the Driving Hamiltonian:
- the pulse defines the rabi frequency $\Omega(t)$, the detuning $\delta(t)$ and the phase $\phi$ of the Driving Hamiltonian over a duration $\Delta t$.
- the channel defines the states $\left|a\right>$ and $\left|b\right>$ of the Driving Hamiltonian.

By applying a serie of pulses and delays, one define the evolution of the Driving Hamiltonian for each atom over time.

**Conclusion**: We have successfully defined the hamiltonian $H$ describing the evolution of the system over time.