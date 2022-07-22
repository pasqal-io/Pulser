************************
Pulse Sequence Creation
************************

Sequence
----------------------

.. automodule:: pulser.sequence.sequence
   :members:

Register
----------------------

Register classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The register classes allow for the creation of arbitrary registers.

.. autoclass:: pulser.register.base_register.BaseRegister
  :members:

.. autoclass:: pulser.register.register.Register
  :members:
  :show-inheritance:

.. autoclass:: pulser.register.register3d.Register3D
  :members:
  :show-inheritance:


Register layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A ``RegisterLayout`` is used to define a register from a set of traps. It is
intended to be given to the user by the hardware provided as a way of showing
which layouts are already available on a given device. In turn, the user
can create a ``Register`` by selecting the traps on which to place atoms, or
even a ``MappableRegister``, which allows for the creation of sequences whose
register can be defined at build time.

.. autoclass:: pulser.register.register_layout.RegisterLayout
  :members:

.. autoclass:: pulser.register.mappable_reg.MappableRegister
  :members:



Special cases
""""""""""""""""""

.. automodule:: pulser.register.special_layouts
  :members:
  :show-inheritance:


Pulse
-------------------

.. automodule:: pulser.pulse
   :members:


Waveforms
----------------------

.. automodule:: pulser.waveforms
  :members:
  :show-inheritance:

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

.. autodata:: pulser.devices.IroiseMVP

The MockDevice
^^^^^^^^^^^^^^^^
A very permissive device that supports sequences which are currently unfeasible
on physical devices. Unlike with physical devices, its channels remain available
after declaration and can be declared again so as to have multiple channels
with the same characteristics.

.. autodata:: pulser.devices.MockDevice

Channels
^^^^^^^^^^^
.. automodule:: pulser.channels
   :members:
   :show-inheritance:
