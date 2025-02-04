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

.. figure:: files/pulser_animation.gif
    :align: center
    :alt: pulser_animation
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
   register
   hardware
   pulses
   tutorials/backends

.. toctree::
   :maxdepth: 2
   :caption: Classical Simulation

   tutorials/noisy_sim
   tutorials/spam
   tutorials/laser_noise
   tutorials/effective_noise

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   tutorials/phase_shifts_vz_gates
   tutorials/composite_wfs
   tutorials/paramseqs
   tutorials/reg_layouts
   tutorials/interpolated_wfs
   tutorials/serialization
   tutorials/dmm
   tutorials/slm_mask
   tutorials/output_mod_eom
   tutorials/virtual_devices


.. toctree::
   :maxdepth: 1
   :caption: Quantum Simulation & Applications

   tutorials/afm_prep
   tutorials/optimization
   tutorials/xy_spin_chain
   tutorials/qubo

.. toctree::
   :maxdepth: 3
   :caption: Documentation

   apidoc/pulser


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
