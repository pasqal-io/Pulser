# Pulser

Pulser is a framework for composing, simulating and executing **pulse** sequences for neutral-atom quantum devices.

**Documentation** for the [latest release](https://pypi.org/project/pulser/) of `pulser` is available at https://pulser.readthedocs.io.

The source code can be found at https://github.com/pasqal-io/Pulser.

## Overview of Pulser

Pulser is designed to let users create experiments that are tailored to specific neutral atom devices,
like those developed at [Pasqal][pasqal].
This reduces the level of abstraction and gives you maximal flexibility and control over the behaviour of the relevant physical parameters, within the bounds set by the chosen device.

Consequently, Pulser breaks free from the paradigm of digital quantum computing
and also allows the creation of **analog** quantum simulations, outside of the
scope of traditional quantum circuit approaches. Whatever the type of experiment
or paradigm, if it can be done on the device, it can be done with Pulser.

Additionally, Pulser features built-in tools for classical simulation (using [QuTiP][qutip] libraries) to aid in the development and testing of new pulse sequences.

## Installation

To install the latest release of ``pulser``, have Python 3.7.0 or higher installed, then use ``pip``:

```bash
pip install pulser
```

If you wish to **install Pulser from source** instead, do the following from within this repository after cloning it:

```bash
pip install -e .
```

### Development Requirements (Optional)

To run the tutorials or the test suite locally, after installation first run the following to install the development requirements:

```bash
pip install -r requirements.txt
```

Then, you can do the following to run the test suite and report test coverage:

```bash
pytest --cov pulser
```
## Contributing

Want to contribute to Pulser? Great! See [How to Contribute][contributing] for information on how you can do so.

[pasqal]: https://pasqal.io/
[qutip]: http://qutip.org/
[contributing]: https://github.com/pasqal-io/Pulser/blob/master/CONTRIBUTING.md
