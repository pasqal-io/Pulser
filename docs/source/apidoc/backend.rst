************************
Backend Interfaces
************************

QPU
----

.. autoclass:: pulser.QPUBackend
   :members:


Emulators
----------

Local
^^^^^^^
.. autoclass:: pulser_simulation.QutipBackend
   :members:

Remote
^^^^^^^^^^
.. autoclass:: pulser_pasqal.EmuTNBackend
   :members:

.. autoclass:: pulser_pasqal.EmuFreeBackend
   :members:


Remote backend connection
---------------------------

.. autoclass:: pulser_pasqal.PasqalCloud
