# Pulser

A framework for interaction with Pasqal devices at the *pulse* level. 

Pulser is a library that allows writing sequences of pulses representing the behaviour of some Pasqal processor prototypes.

- The user can define one or several channels to target the qubits in the device. 
- A basis can be chosen to represent the transition levels of the Rydberg atom-arrays setup.
- Channels can be local or global depending on the application. In the local case, a phase-shift option is included to reduce complexity
- There's a visualization routine for ease of use.

The pulse sequences can then be read and operated by real Pasqal devices or emulated (using QuTiP libraries)
