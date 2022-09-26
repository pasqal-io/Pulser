# Pulser

Pulser is a framework for composing, simulating and executing **pulse** sequences for neutral-atom quantum devices.

**Documentation** for the [latest release](https://pypi.org/project/pulser/) of `pulser` is available at <https://pulser.readthedocs.io> (for the docs tracking the `develop` branch of this repository, visit <https://pulser.readthedocs.io/en/latest> instead).

The source code can be found at <https://github.com/pasqal-io/Pulser>.

## Overview of Pulser

Pulser is designed to let users create experiments that are tailored to specific neutral-atom devices. This reduces the level of abstraction and gives you maximal flexibility and control over the behaviour of the relevant physical parameters, within the bounds set by the chosen device.

Consequently, Pulser breaks free from the paradigm of digital quantum computing
and also allows the creation of **analog** quantum simulations, outside of the
scope of traditional quantum circuit approaches. Whatever the type of experiment
or paradigm, if it can be done on the device, it can be done with Pulser.

Additionally, the ``pulser_simulation`` extension provides tools for classical simulation (using [QuTiP][qutip] libraries) to aid in the development and testing of new pulse sequences.

For a comprehensive overview of Pulser, check out [Pulser's white paper](https://quantum-journal.org/papers/q-2022-01-24-629/).

## Installation

**Note**: *Pulser v0.6 introduced a split of the ``pulser`` package that prevents it from being correctly upgraded. If you have an older version of ``pulser`` installed and wish to upgrade, make sure to uninstall it first by running ``pip uninstall pulser``.*

To install the latest release of ``pulser``, have Python 3.7.0 or higher installed, then use ``pip``:

```bash
pip install pulser
```

The standard ``pulser`` distribution will install the core ``pulser`` package
and the ``pulser_simulation`` extension package, which is required if you want
to access the emulation features.

If you wish to install only the core ``pulser`` features, you can instead run:

```bash
pip install pulser-core
```

If you wish to **install the development version of Pulser from source** instead, do the following from within this repository after cloning it:

```bash
git checkout develop
make dev-install
```

Bear in mind that this installation will track the contents of your local
Pulser repository folder, so if you checkout a different branch (e.g. ``master``),
your installation will change accordingly.

### Development Requirements (Optional)

To run the tutorials or the test suite locally, after installation first run the following to install the development requirements:

```bash
pip install -r dev_requirements.txt
```

Then, you can do the following to run the test suite and report test coverage:

```bash
pytest --cov
```

## Contributing

Want to contribute to Pulser? Great! See [How to Contribute][contributing] for information on how you can do so.

[qutip]: http://qutip.org/
[contributing]: https://github.com/pasqal-io/Pulser/blob/master/CONTRIBUTING.md

## Citing Pulser

Citation references are generated through Zenodo. Click the badge below to get the citation to the latest Pulser release.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4707943.svg)](https://doi.org/10.5281/zenodo.4707943)
