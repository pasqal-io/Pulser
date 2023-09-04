****************************************
Conventions
****************************************

States and Bases
####################################

Bases
*******
A basis refers to a set of two eigenstates. The transition between
these two states is said to be addressed by a channel that targets that basis. Namely:

.. list-table:: 
   :align: center
   :widths: 50 35 35
   :header-rows: 1

   * - Basis
     - Eigenstates
     - ``Channel`` type
   * - ``ground-rydberg``
     - :math:`|g\rangle,~|r\rangle`
     - ``Rydberg``
   * - ``digital``
     - :math:`|g\rangle,~|h\rangle`
     - ``Raman``
   * - ``XY``
     - :math:`|0\rangle,~|1\rangle`
     - ``Microwave``



Qutrit state
******************

The qutrit state combines the basis states of the ``ground-rydberg`` and ``digital`` bases, 
which share the same ground state, :math:`|g\rangle`. This qutrit state comes into play
in the digital approach, where the qubit state is encoded in :math:`|g\rangle` and 
:math:`|h\rangle` but then the Rydberg state :math:`|r\rangle` is accessed in multi-qubit
gates.

The qutrit state's basis vectors are defined as:

.. math:: |r\rangle = (1, 0, 0)^T,~~|g\rangle = (0, 1, 0)^T, ~~|h\rangle = (0, 0, 1)^T.

Qubit states
**************

.. warning:: 
  There is no implicit relationship between a state's vector representation and its 
  associated measurement value. To see the measurement value of a state for each 
  measurement basis, see :ref:`SPAM` .

When using only the ``ground-rydberg`` or ``digital`` basis, the qutrit state is not
needed and is thus reduced to a qubit state. This reduction is made simply by tracing-out
the extra basis state, so we obtain

* ``ground-rydberg``: :math:`|r\rangle = (1, 0)^T,~~|g\rangle = (0, 1)^T`
* ``digital``: :math:`|g\rangle = (1, 0)^T,~~|h\rangle = (0, 1)^T`

On the other hand, the ``XY`` basis uses an independent set of qubit states that are 
labelled :math:`|0\rangle` and :math:`|1\rangle` and follow the standard convention:

* ``XY``: :math:`|0\rangle = (1, 0)^T,~~|1\rangle = (0, 1)^T`

Multi-partite states
*************************

The combined quantum state of multiple atoms respects their order in the ``Register``.
For a register with ordered atoms ``(q0, q1, q2, ..., qn)``, the full quantum state will be

.. math:: |q_0, q_1, q_2, ...\rangle = |q_0\rangle \otimes |q_1\rangle \otimes |q_2\rangle \otimes ... \otimes |q_n\rangle

.. note::
  The atoms may be labelled arbitrarily without any inherent order, it's only the
  order with which they are stored in the ``Register`` (as returned by 
  ``Register.qubit_ids``) that matters .

.. _SPAM:

State Preparation and Measurement
####################################

.. list-table:: Initial State and Measurement Conventions
   :align: center
   :widths: 60 40 75
   :header-rows: 1

   * - Basis
     - Initial state
     - Measurement
   * - ``ground-rydberg``
     - :math:`|g\rangle`
     - |
       | :math:`|r\rangle \rightarrow 1`
       | :math:`|g\rangle,|h\rangle \rightarrow 0`
   * - ``digital``
     - :math:`|g\rangle`
     - |
       | :math:`|h\rangle \rightarrow 1`
       | :math:`|g\rangle,|r\rangle \rightarrow 0`
   * - ``XY``
     - :math:`|0\rangle`
     - |
       | :math:`|1\rangle \rightarrow 1`
       | :math:`|0\rangle \rightarrow 0`

Measurement samples order
***************************

Measurement samples are returned as a sequence of 0s and 1s, in
the same order as the atoms in the ``Register`` and in the multi-partite state.

For example, a four-qutrit state :math:`|q_0, q_1, q_2, q_3\rangle` that's
projected onto :math:`|g, r, h, r\rangle` when measured will record a count to
sample

* ``0101``, if measured in the ``ground-rydberg`` basis
* ``0010``, if measured in the ``digital`` basis

Hamiltonians
####################################

Independently of the mode of operation, the Hamiltonian describing the system
can be written as

.. math:: H(t) = \sum_i \left (H^D_i(t) + \sum_{j<i}H^\text{int}_{ij} \right), 

where :math:`H^D_i` is the driving Hamiltonian for atom :math:`i` and
:math:`H^\text{int}_{ij}` is the interaction Hamiltonian between atoms :math:`i`
and :math:`j`. Note that, if multiple basis are addressed, there will be a 
corresponding driving Hamiltonian for each transition.


Driving Hamiltonian
*********************

The driving Hamiltonian describes the coherent excitation of an individual atom
between two energies levels, :math:`|a\rangle` and :math:`|b\rangle`, with
Rabi frequency :math:`\Omega(t)`, detuning :math:`\delta(t)` and phase :math:`\phi(t)`.

.. math:: H^D(t) / \hbar = \frac{\Omega(t)}{2} e^{-i\phi(t)} |a\rangle\langle b| + \frac{\Omega(t)}{2} e^{i\phi(t)} |b\rangle\langle a| - \delta(t) |b\rangle\langle b|

.. warning::
  In this form, the Hamiltonian is **independent of the state vector representation of each basis state**,
  but it still assumes that :math:`|b\rangle` **has a higher energy than** :math:`|a\rangle`.


Pauli matrix form
---------------------

A more conventional representation of the driving Hamiltonian uses Pauli operators 
instead of projectors. However, this form now **depends on the state vector definition**
of :math:`|a\rangle` and :math:`|b\rangle`.

Pulser's state-vector definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Pulser, we consistently define the state vectors according to their relative energy.
In this way we have, for any given basis, that

.. math:: |b\rangle = (1, 0)^T,~~|a\rangle = (0, 1)^T

Thus, the Pauli and excited state occupation operators are defined as

.. math::

  \hat{\sigma}^x = |a\rangle\langle b| + |b\rangle\langle a|, \\
  \hat{\sigma}^y = i|a\rangle\langle b| - i|b\rangle\langle a|, \\
  \hat{\sigma}^z = |b\rangle\langle b| - |a\rangle\langle a|  \\
  \hat{n} = |b\rangle\langle b| = (1 + \sigma_z) / 2 

and the driving Hamiltonian takes the form

.. math:: 
  
  H^D(t) / \hbar = \frac{\Omega(t)}{2} \cos\phi(t) \hat{\sigma}^x 
  - \frac{\Omega(t)}{2} \sin\phi(t) \hat{\sigma}^y 
  - \delta(t) \hat{n}


Alternative state-vector definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Outside of Pulser, the alternative definition for the basis state 
vectors might be taken:

.. math:: |a\rangle = (1, 0)^T,~~|b\rangle = (0, 1)^T

This changes the operators and Hamiltonian definitions, 
as rewriten below with highlighted differences.

.. math::

  \hat{\sigma}^x = |a\rangle\langle b| + |b\rangle\langle a|, \\
  \hat{\sigma}^y = \textcolor{red}{-}i|a\rangle\langle b| \textcolor{red}{+}i|b\rangle\langle a|, \\
  \hat{\sigma}^z = \textcolor{red}{-}|b\rangle\langle b| \textcolor{red}{+} |a\rangle\langle a|  \\
  \hat{n} = |b\rangle\langle b| = (1 \textcolor{red}{-} \sigma_z) / 2 

.. math:: 
  
  H^D(t) / \hbar = \frac{\Omega(t)}{2} \cos\phi(t) \hat{\sigma}^x 
  \textcolor{red}{+}\frac{\Omega(t)}{2} \sin\phi(t) \hat{\sigma}^y 
  - \delta(t) \hat{n}

.. note::
  A common case for the use of this alternative definition arises when
  trying to reconcile the  basis states of the ``ground-rydberg`` basis 
  (where :math:`|r\rangle` is the higher energy level) with the 
  computational-basis state-vector convention, thus ending up with 

  .. math:: |0\rangle = |g\rangle = |a\rangle = (1, 0)^T,~~|1\rangle = |r\rangle = |b\rangle = (0, 1)^T


Interaction Hamiltonian
*************************

The interaction Hamiltonian depends on the states involved in the sequence. 
When working with the ``ground-rydberg`` and ``digital`` bases, atoms interact
when they are in the Rydberg state :math:`|r\rangle`:

.. math:: H^\text{int}_{ij} = \frac{C_6}{R_{ij}^6} \hat{n}_i \hat{n}_j

where :math:`\hat{n}_i = |r\rangle\langle r|_i` (the projector of
atom :math:`i` onto the Rydberg state), :math:`R_{ij}^6` is the distance 
between atoms :math:`i` and :math:`j` and :math:`C_6` is a coefficient
depending on the specific Rydberg level of :math:`|r\rangle`.

On the other hand, with the two Rydberg states of the ``XY``
basis, the interaction Hamiltonian takes the form

.. math:: H^\text{int}_{ij} =  \frac{C_3}{R_{ij}^3} (\hat{\sigma}_i^{+}\hat{\sigma}_j^{-} + \hat{\sigma}_i^{-}\hat{\sigma}_j^{+})

where :math:`C_3` is a coefficient that depends on the chosen Ryberg states
and 

.. math:: \hat{\sigma}_i^{+} =  |1\rangle\langle 0|_i,~~~\hat{\sigma}_i^{-} =  |0\rangle\langle 1|_i

.. note:: The definitions given for both interaction Hamiltonians are independent of the chosen state vector convention.