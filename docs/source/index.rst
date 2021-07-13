Pulser
==================================

**Pulser** is an open-source Python software package. It provides easy-to-use
libraries for designing and simulating pulse sequences that act on
programmable arrays of neutral atoms, a promising platform for quantum computation
and simulation.

**Online documentation**: `<https://pulser.readthedocs.io>`_

**Source code repository** (go `here <https://pulser.readthedocs.io/en/latest/>`_
for the latest docs): `<https://github.com/pasqal-io/Pulser>`_

**License**: Apache 2.0 -- see `LICENSE <https://github.com/pasqal-io/Pulser/blob/master/LICENSE>`_
for details

Overview
---------------

Pulser is designed to let users create experiments that are tailored to a
specific device. In this way, you can have maximal flexibility and control over
the behaviour of relevant physical parameters, within the bounds set by the chosen device.

.. figure:: https://pasqal.io/wp-content/uploads/2021/02/pulser_animation.gif
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
check out the :doc:`introduction_rydberg_blockade` page. For a more in-depth
introduction, consult the tutorials on :doc:`creating` and :doc:`simulating`.
To better understand neutral atom devices and how they serve as quantum
computers and simulators, check the pages in :doc:`review`.


.. toctree::
   :maxdepth: 2
   :caption: Installation and First Steps

   installation
   introduction_rydberg_blockade
   creating
   simulating

.. toctree::
   :maxdepth: 2
   :caption: Fundamental Concepts

   review

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   phase_shifts_vz_gates
   composite_wfs
   paramseqs
   interpolated_wfs
   serialization

.. toctree::
   :maxdepth: 2
   :caption: Applications

   noisy_sim
   cz_gate
   afm_prep
   1D_crystals
   optimization
   qaoa_mis
   qaoa_param_seq

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   pulser


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
