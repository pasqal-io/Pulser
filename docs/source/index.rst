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
check out the :doc:`intro_rydberg_blockade` page. For a more in-depth
introduction, consult the tutorials on :doc:`tutorials/creating` and
:doc:`tutorials/simulating`.
To better understand neutral atom devices and how they serve as quantum
computers and simulators, check the pages in :doc:`review`.


.. toctree::
   :maxdepth: 2
   :caption: Installation and First Steps

   installation
   intro_rydberg_blockade
   tutorials/creating
   tutorials/simulating

.. toctree::
   :maxdepth: 2
   :caption: Fundamental Concepts

   review

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   tutorials/phase_shifts_vz_gates
   tutorials/composite_wfs
   tutorials/paramseqs
   tutorials/interpolated_wfs
   tutorials/serialization

.. toctree::
   :maxdepth: 1
   :caption: Applications

   tutorials/noisy_sim
   tutorials/cz_gate
   tutorials/afm_prep
   tutorials/1D_crystals
   tutorials/optimization
   tutorials/qaoa_mis
   tutorials/qaoa_param_seq
   tutorials/qek

.. toctree::
   :maxdepth: 3
   :caption: Documentation

   apidoc/pulser


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
