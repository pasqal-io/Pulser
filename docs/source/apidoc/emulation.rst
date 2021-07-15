*********************
Classical Simulation
*********************

Simulation
----------------------

.. automodule:: pulser.simulation.simulation
   :members:

SimConfig
----------------------

.. automodule:: pulser.simulation.simconfig
   :members:

Simulation Results
-----------------------

Depending on the types of noise involved in a simulation, the results are returned
as an instance of :class:`SimulationResults`, namely :class:`CoherentResults`
(when the results can still be represented as a state vector or a density matrix, before being measured)
or :class:`NoisyResults` (when they can only be represented as a probability
distribution over the basis states).

Both classes feature methods for processing and displaying the results stored
within them.

CoherentResults
^^^^^^^^^^^^^^^^

.. autoclass:: pulser.simulation.simresults.CoherentResults
   :members:
   :inherited-members:


NoisyResults
^^^^^^^^^^^^^^^^

.. autoclass:: pulser.simulation.simresults.NoisyResults
  :members:
  :inherited-members:
