************************
Pulse Sequence Creation
************************

Sequence
----------------------

.. automodule:: pulser.sequence
   :members:
   :undoc-members:

Register
----------------------

.. automodule:: pulser.register
  :members:
  :undoc-members:

Devices
---------------------

Structure of a Device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`Device` class sets the structure of every device instance.

.. automodule:: pulser.devices._device_datacls
   :members:

Physical Devices
^^^^^^^^^^^^^^^^^^^
Each device instance holds the characteristics of a physical device,
which when associated with a :class:`pulser.Sequence` condition its development.

.. autodata:: pulser.devices.Chadoq2

The MockDevice
^^^^^^^^^^^^^^^^
A very permissive device that supports sequences which are currently unfeasible
on physical devices. Unlike with physical devices, its channels remain available
after declaration and can be declared again so as to have multiple channels
with the same characteristics.

.. autodata:: pulser.devices.MockDevice

Channels
^^^^^^^^^^^
Each device is contains a set hardware Channels, to which the Pulses are allocated.
Each channel has specific configurations, which dictate the pulses it can execute.

.. automodule:: pulser.channels
   :members:
   :show-inheritance:


Pulse
-------------------

.. automodule:: pulser.pulse
   :members:
   :undoc-members:


Waveforms
----------------------

.. automodule:: pulser.waveforms
  :members:
  :undoc-members:
  :show-inheritance:
