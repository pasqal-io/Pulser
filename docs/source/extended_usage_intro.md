# Introduction to Extended Usage

In the "Fundamentals" section, we introduced the basic tools for Analog quantum computing with the Ising Hamiltonian. In this section, we present more tools for Analog quantum computing with the Ising Hamiltonian, as well as introduce other tools to program in other quantum computing paradigms: Weighted Analog with the Ising Hamiltonian, Analog with the XY Hamiltonian and Digital. 

:::{important}
To program in a specific quantum computing paradigm, ensure the needed features are in your chosen Device's specifications.
:::

## Extending Analog Quantum Computing with the Ising Hamiltonian

Analog Quantum Computing with the Ising Hamiltonian refers to quantum programs using only one `Channel`, the `Rydberg.Global` channel. It enables the programming of the [Ising Hamiltonian](./programming.md#ising-hamiltonian):

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega(t)}{2} e^{-i\phi(t)} |g\rangle\langle r|_k + \frac{\Omega(t)}{2} e^{i\phi(t)} |r\rangle\langle g|_k - \delta(t) |r\rangle\langle r|_k(t) + \sum_{j<k}\frac{C_6}{\hbar R_{kj}^6} \hat{n}_k \hat{n}_j \right)
$$

Let's follow the [step-by-step guide on how to create a quantum program using Pulser](./programming.md#writing-a-pulser-program), and introduce new features to extend the one introduced in "Fundamentals":

### 1. Pick a Device

- [VirtualDevice](./tutorials/virtual_devices.nblink) extends the definition of [Device](./hardware.ipynb), enabling the definition of a `Device` with less physical constraints.

### 2. Create the Register

- [3D Registers](./apidoc/_autosummary/pulser.register.Register3D) enable the definition of Registers in 3D.
- [RegisterLayout](./tutorials/reg_layouts.nblink) defines the layout of traps from which registers of atoms can be defined. For some QPU, it is mandatory to define the `Register` from a `RegisterLayout`. 
- [MappableRegister](./tutorials/reg_layouts.nblink) is a `Register` with the traps of each qubit still to be defined.

### 3. Pick the channels

- {ref}`Modulation </tutorials/output_mod_eom.nblink#Output-Modulation>`: Each channel has a _modulation bandwidth_, that defines how the pulses that will be added to it will be affected by the modulation phenomenon.
- {ref}`EOM </tutorials/output_mod_eom.nblink#EOM-Mode-Operation>`: Some channels support an "EOM" mode, a mode in which the pulses are less impacted by the modulation phenomenon, but have to be of square shape.

### 4. Add the Pulses

- {ref}`Arbitrary Phase Waveforms </apidoc/_autosummary/pulser.pulse.Pulse.rst#pulser.pulse.Pulse.ArbitraryPhase>` enables the definition of a Pulse with an arbitrary phase.

### Other extended usage

- [Parametrized Sequences](./tutorials/paramseqs.nblink) enable the generation of multiple Sequences that vary only in a few parameters.
- [Serialization/Deserialization](./tutorials/serialization.nblink) enable the import/export of Sequences between different locations.
- [Noise Models](./noise_model.ipynb) enable the emulation of Sequences, taking into account the noises of neutral-atom Quantum Processing Units.

## Weighted Analog Quantum Computing

Weighted Analog with the Ising Hamiltonian designates quantum programs combining the `Rydberg.Global` channel with a `DMM` channel. It enables the definition of an Ising Hamiltonian with local control over the detuning:

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega(t)}{2} e^{-i\phi(t)} |g\rangle\langle r|_k + \frac{\Omega(t)}{2} e^{i\phi(t)} |r\rangle\langle g|_k - \left[\delta(t)\mathbf{+\epsilon_k\delta_{DMM}(t)}\right] |r\rangle\langle r|_k + \sum_{j<k}\frac{C_6}{\hbar R_{kj}^6} \hat{n}_k \hat{n}_j \right)
$$

Here, the _weights_ $\{\epsilon_k\}_{1\lt k\lt N}$ are defined by a `DetuningMap`, that has to be defined right after [you create the register](programming.md#2-create-the-register).

- An in-depth presentation of Weighted Analog with the Ising Hamiltonian is available [in this notebook](./tutorials/dmm.nblink). Notably, it presents how to create a `DetuningMap`, how to pick a `DMM` and how to add detuning waveforms $\delta_{DMM}$ to it. 
- Weighted Analog can be used to prepare the qubits in a specific initial state. This is eased by using an {ref}`SLM Mask </tutorials/slm_mask.nblink#SLM-Mask-in-Ising-mode>`.

## Analog with the XY Hamiltonian

One can also perform Analog Quantum Computing with the [XY Hamiltonian](./programming.md#xy-hamiltonian). The `Channel` associated with this is the `Microwave.Global` Channel:

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega(t)}{2} e^{-i\phi(t)} |g\rangle\langle r|_k + \frac{\Omega(t)}{2} e^{i\phi(t)} |r\rangle\langle g|_k - \delta(t) |r\rangle\langle r|_k(t) + \sum_{j<k}\frac{C_3}{\hbar R_{kj}^3} (|1\rangle\langle 0|_k |0\rangle\langle 1|_j + |0\rangle\langle 1|_i |1\rangle\langle 0|_k) \right)
$$

- An in-depth presentation of Analog quantum computing with XY Hamiltonian can be found [in this notebook](tutorials/xy_spin_chain.nblink).
- An [SLM mask](./tutorials/slm_mask.nblink) can also be used to prepare the initial state in a combination of XY basis states, $\left|0\right>$ and $\left|1\right>$.

## Digital Quantum Computing

Digital Quantum Computing is a paradigm in which a system's state evolves through a series of discrete manipulation of its qubits' states, known as quantum gates. This is the underlying approach in quantum circuits, and can be replicated in neutral-atom devices at the pulse level.

To achieve this, the qubit states are encoded in the states of the "digital" basis $\left|g\right>$ and $\left|h\right>$. Digital Quantum Computing is thus associated with the `Raman` channel. When adding a pulse to the `Raman` channel, the Hamiltonian you program is:

$$\frac{H}{\hbar}(t) = \sum_{k=1}^N \left (\frac{\Omega_k(t)}{2} e^{-i\phi_k(t)} |h\rangle\langle g|_k + \frac{\Omega_k(t)}{2} e^{i\phi_k(t)} |g\rangle\langle h|_k - \delta_k(t) |g\rangle\langle g|_k(t) \right)
$$

- {ref}`Local pulses and target operations </tutorials/phase_shifts_vz_gates.nblink#Phase-shifts-with-multiple-channels-and-different-targets>` enable to define gates applying on only specific qubits, by defining a driving Hamiltonian for a set of targeted atoms specifically (the quantities $\Omega_k$, $\delta_k$ and $\phi_k$ in the Hamiltonian above depend on the atoms).
- [Virtual Z gates and phase shifts](tutorials/phase_shifts_vz_gates.nblink): phase shift is an operation that can be programmed in between two pulses to program a _virtual-Z gate_, a phase gate. This tutorial presents how to use it to perform an Hadamard gate.
