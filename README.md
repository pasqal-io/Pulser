# Pulser

A framework for interacting with [Pasqal][pasqal] devices at the **pulse** level.

Pulser is a library that allows for writing sequences of pulses representing the
behaviour of some Pasqal processor prototypes.

- The user can define one or several channels to target the qubits in the device. 
- A basis can be chosen to represent the transition levels of the Rydberg atom-arrays setup.
- Channels can be local or global depending on the application. In the local case,
  a phase-shift option is included to reduce complexity
- There's a visualization routine for ease of use.

The pulse sequences can then be read and operated by real Pasqal devices or
emulated (using [QuTiP][qutip] libraries).

## Installation

To install Pulser from source, do the following from within the repository
after cloning it:

```bash
pip install -e .
```

## Testing

To run the test suite, after installation first run the following to install
development requirements:

```bash
pip install -r requirements.txt
```

Then, do the following to run the test suite and report test coverage:

```bash
pytest --cov pulser
```

[pasqal]: https://pasqal.io/
[qutip]: http://qutip.org/
