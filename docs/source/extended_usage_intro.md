# Introduction to Extended Usage

Extended Usage cover features that are specific to some QPUs or applications. In the fundamentals section, we focused on Analog quantum computing with the Ising Hamiltonian. In this section, we extend the features for Analog quantum computing with the Ising Hamiltonian, and present tools for other quantum computing paradigms: Weighted Analog and Digital. 

## Extending Analog Quantum Computing with the Ising Hamiltonian

Let's follow the [step-by-step guide on how to create a quantum program using Pulser](./programming.md#writing-a-pulser-program):

### 1. Pick a Device

[VirtualDevice](./tutorials/virtual_devices.nblink) extends the definition of [Device](./hardware.ipynb), enabling the definition of a `Device` with less physical constraints.

### 2. Create the Register

- [3D Registers](./apidoc/_autosummary/pulser.Register3D) enable the definition of Registers in 3D.
- [RegisterLayout](./tutorials/reg_layouts.nblink) define the traps from which Registers are defined. For some QPU, it is mandatory to define the `Register` from a `RegisterLayout`. 
- [MappableRegister](./tutorials/reg_layouts.nblink) is a Register with the traps of each qubit still to be defined.

### 3. Pick the channels

- [Modulation](./tutorials/output_mod_eom.nblink): Each channel has a _modulation bandwidth_, that defines how the pulses that will be added to it will be affected by the modulation phenomenon.
- [EOM](./tutorials/output_mod_eom.nblink): Some channels support an "EOM" mode, a mode in which the pulses are less impacted by the modulation phenomenon, but have to be of square shape.

### 4. Add the Pulses

- [Arbitrary Phase Waveforms](./apidoc/_autosummary/pulser.Pulse) enables the definition of a Pulse with an arbitrary phase.

### Other extended usage

- [Parametrized Sequence](./tutorials/paramseqs.nblink) enables to generate multiple Sequences that vary only in a few parameters.
- [Serialization/Deserialization](./tutorials/serialization.nblink) enable the import/export of Sequences between different locations.

## Weighted Analog Quantum Computing

### Weighted Analog with the Ising Hamiltonian

The `Channel` associated with Analog Quantum Computing with the Ising Hamiltonian is the `Rydberg.Global` channel. It enables to manipulate the [Ising Hamiltonian](./programming.md#ising-hamiltonian):

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega(t)}{2} e^{-i\phi(t)} |g\rangle\langle r|_k + \frac{\Omega(t)}{2} e^{i\phi(t)} |r\rangle\langle g|_k - \delta(t) |r\rangle\langle r|_k(t) + \sum_{j<k}\frac{C_6}{\hbar R_{kj}^6} \hat{n}_k \hat{n}_j \right)
$$

Weighted Analog designs quantum programs combining the `Rydberg.Global` channel with a `DMM` channel. It enables the definition of an Ising Hamiltonian with local control over the detuning:

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega(t)}{2} e^{-i\phi(t)} |g\rangle\langle r|_k + \frac{\Omega(t)}{2} e^{i\phi(t)} |r\rangle\langle g|_k - (\delta(t)+\epsilon_k\delta_{DMM}(t)) + |r\rangle\langle r|_k(t) + \sum_{j<k}\frac{C_6}{\hbar R_{kj}^6} \hat{n}_k \hat{n}_j \right)
$$

Here, the _weights_ $\{\epsilon_k\}_{1\lt k\lt N}$ are defined by a `DetuningMap`, that has to be defined right after [you create the register](./programming.md#2-create-the-register).

### Weighted Analog with the XY Hamiltonian

One can also perform Analog Quantum Computing with the [XY Hamiltonian](./programming.md#xy-hamiltonian). The `Channel` associated with this is the `Microwave.Global` Channel:

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega(t)}{2} e^{-i\phi(t)} |g\rangle\langle r|_k + \frac{\Omega(t)}{2} e^{i\phi(t)} |r\rangle\langle g|_k - \delta(t) |r\rangle\langle r|_k(t) + \sum_{j<k}\frac{C_3}{\hbar R_{kj}^3} (|1\rangle\langle 0|_k |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_k) \right)
$$

An `SLM mask`, relying on a `DMM`, can be used here to prepare the initial state of the qubits in a state different than $\left|0\right>\otimes \left|0\right> \otimes ... \otimes \left|0\right>$.

### Documentation

- An in-depth presentation of Weighted Analog with the Ising Hamiltonian is available [in this notebook](./tutorials/dmm.nblink). Notably, it presents how to create a `DetuningMap`, how to pick a `DMM` and how to add detuning waveforms $\delta_{DMM}$ to it. 
- The `DMM` enables to use the [SLM mask feature](./tutorials/slm_mask.nblink), that allows the preparation of the qubits in an initial state composed of:
    - ground $\left|g\right>$ and Rydberg state $\left|r\right>$ when the SLM mask is applied on a `Rydberg.Global` channel.
    - XY states $\left|0\right>$ and $\left|1\right>$ when the SLM mask is applied on a `Microwave.Global` channel.

## Digital Quantum Computing

By using `Rydberg` and `Raman` channels in your quantum program, you can program digital quantum computations. To perform such operations, here are some extended usage of relevance:
- [Local pulses and target operations](./tutorials/phase_shifts_vz_gates.nblink) enable to define a driving Hamiltonian for a set of targeted atoms specifically.
- [Virtual Z gates and phase shifts](./tutorials/phase_shifts_vz_gates.nblink): phase shift is an operation on the Sequence, that can be programmed in between two pulses to program a _virtual-Z gate_, a phase gate. This tutorial presents how to use it to perform an Hadamard gate.
