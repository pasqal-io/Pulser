API Reference
==============


Channels
----------------------

.. automodule:: pulser.channels
   :members:
   :show-inheritance:

Devices
---------------------

Structure of a Pasqal Device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`PasqalDevice` class sets the structure of every device instance.

.. automodule:: pulser.devices._pasqal_device
   :members:

Physical Devices
^^^^^^^^^^^^^^^^^^^
Each device instance holds the characteristics of one of Pasqal's physical devices,
which when associated with a :class:`pulser.Sequence` condition its development.

.. autodata:: pulser.devices.Chadoq2

The MockDevice
^^^^^^^^^^^^^^^^
A very permissive device that supports sequences which are currently unfeasible
on physical devices. Unlike with physical devices, its channels remain available
after declaration and can be declared again so as to have multiple channels
with the same characteristics.

.. autodata:: pulser.devices.MockDevice

Pulse
-------------------

.. automodule:: pulser.pulse
   :members:
   :undoc-members:

Register
----------------------

.. automodule:: pulser.register
   :members:
   :undoc-members:

Sequence
----------------------

.. automodule:: pulser.sequence
   :members:
   :undoc-members:

Simulation
----------------------

.. automodule:: pulser.simulation
   :members:
   :undoc-members:

Simulation Results
----------------------

.. automodule:: pulser.simresults
   :members:
   :undoc-members:


Waveforms
-----------------------

.. automodule:: pulser.waveforms
   :members:
   :undoc-members:
   :show-inheritance:
