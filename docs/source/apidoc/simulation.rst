*********************
Classical Simulation
*********************

Since version 0.6.0, all simulation classes (previously under the ``pulser.simulation`` module)
are in the ``pulser-simulation`` extension and should be imported from ``pulser_simulation``.

QutipEmulator
----------------------

:class:`QutipEmulator` is the class to simulate :class:`SequenceSamples`, that are samples of a :class:`Sequence`.
It is possible to simulate directly a :class:`Sequence` object by using the class method 
``QutipEmulator.from_sequence``. Since version 0.14.0, the :class:`Simulation` class is deprecated 
in favour of :class:`QutipEmulator`.

.. autoclass:: pulser_simulation.simulation.QutipEmulator
   :members:

.. autoclass:: pulser_simulation.simulation.Simulation

SimConfig
----------------------

.. automodule:: pulser_simulation.simconfig
   :members:

Simulation Results
-----------------------

Depending on the types of noise involved in a simulation, the results are returned
as an instance of :class:`SimulationResults`, namely :class:`CoherentResults`
(when each result can still be represented as a :class:`QutipResult` i.e. a state vector
or a density matrix, before being measured) or :class:`NoisyResults` (when each of them can only
be represented as :class:`SampledResult` i.e. a probability distribution over the basis states).

Both classes feature methods for processing and displaying the results stored
within them.

QutipResult
^^^^^^^^^^^^^^^^

.. autoclass:: pulser_simulation.qutip_result.QutipResult
   :members:
   :inherited-members:

CoherentResults
^^^^^^^^^^^^^^^^

.. autoclass:: pulser_simulation.simresults.CoherentResults
   :members:
   :inherited-members:


NoisyResults
^^^^^^^^^^^^^^^^

.. autoclass:: pulser_simulation.simresults.NoisyResults
  :members:
  :inherited-members:
