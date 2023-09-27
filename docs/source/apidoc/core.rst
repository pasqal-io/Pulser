************************
Core Features
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
  :inherited-members:

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
The :class:`Device` class sets the structure of a physical device, 
while :class:`VirtualDevice` is a more permissive device type which can
only be used in emulators, as it does not necessarily represent the
constraints of a physical device. 

Illustrative instances of :class:`Device` (see `Physical Devices`_) and :class:`VirtualDevice` 
(the `MockDevice`) come included in the `pulser.devices` module.

.. autoclass:: pulser.devices._device_datacls.Device
  :members:
  :inherited-members:

.. autoclass:: pulser.devices._device_datacls.VirtualDevice
  :members:
  :inherited-members:

.. _Physical Devices:

Physical Devices
^^^^^^^^^^^^^^^^^^^
Each `Device`` instance holds the characteristics of a physical device,
which when associated with a :class:`pulser.Sequence` condition its development.

.. autodata:: pulser.devices.Chadoq2

.. autodata:: pulser.devices.AnalogDevice


Channels
---------------------

Base Channel
^^^^^^^^^^^^^^^
.. automodule:: pulser.channels.base_channel
   :members:


Available Channels
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pulser.channels.channels
   :members:
   :show-inheritance:

.. automodule:: pulser.channels.dmm
   :members:
   :show-inheritance:

EOM Mode Configuration
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pulser.channels.eom
   :members:
   :show-inheritance:

Sampler
------------------
.. automodule:: pulser.sampler.sampler
   :members:

.. automodule:: pulser.sampler.samples
   :members:

Result
------------------
.. automodule:: pulser.result
   :members: