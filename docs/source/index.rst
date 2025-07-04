Pulser
==================================

**Pulser** is an open-source Python software package. It provides easy-to-use
libraries for designing and simulating pulse sequences that act on
programmable arrays of neutral atoms, a promising platform for quantum computation
and simulation.

**Online documentation**: `<https://pulser.readthedocs.io>`_

**White paper**: `Quantum 6, 629 (2022) <https://quantum-journal.org/papers/q-2022-01-24-629/>`_

**Source code repository** (go `here <https://pulser.readthedocs.io/en/latest/>`_
for the latest docs): `<https://github.com/pasqal-io/Pulser>`_

**License**: Apache 2.0 -- see `LICENSE <https://github.com/pasqal-io/Pulser/blob/master/LICENSE>`_
for details

Overview
---------------

Pulser is designed to let users create experiments that are tailored to a
specific device. In this way, you can have maximal flexibility and control over
the behaviour of relevant physical parameters, within the bounds set by the chosen device.

.. figure:: files/pulser_animation.webp
    :align: center
    :alt: Animated diagram depicting the execution of a multi-channel sequence as different types of laser pulses acting on an atom register
    :figclass: align-center

    Execution of a pulse sequence designed for a specific device.

Consequently, Pulser breaks free from the paradigm of digital quantum computing
and also allows the creation of **analog** quantum simulations, outside of the
scope of traditional quantum circuit approaches. Whatever the type of experiment
or paradigm, if it can be done on the device, it can be done with Pulser.

Additionally, Pulser features built-in tools for classical simulation to aid in
the development and testing of new pulse sequences.

To get started with Pulser, follow the instructions in :doc:`installation` and
check out the :doc:`programming` page to learn what mathematical objects you
are programming with Pulser and how to program them. Then, you can see examples
of quantum programs written with Pulser on :doc:`tutorials/creating`.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   programming
   tutorials/creating

.. toctree::
   :maxdepth: 2
   :caption: Fundamentals

   conventions
   hardware
   register
   pulses
   sequence
   tutorials/backends

.. toctree::
   :maxdepth: 1
   :caption: Extended Usage

   extended_usage_intro
   tutorials/reg_layouts
   tutorials/output_mod_eom
   tutorials/virtual_devices
   tutorials/paramseqs
   tutorials/composite_wfs
   tutorials/interpolated_wfs
   tutorials/serialization
   noise_model
   tutorials/dmm
   tutorials/slm_mask
   tutorials/xy_spin_chain
   tutorials/phase_shifts_vz_gates

.. toctree::
   :maxdepth: 1
   :caption: Applications

   tutorials/optimization
   tutorials/qubo
   tutorials/mwis

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   apidoc/core
   apidoc/simulation
   apidoc/pasqal


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
