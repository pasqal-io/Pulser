``pulser``
=================

.. automodule:: pulser

Classes
-----------------

These are classes that can be imported directly from ``pulser``. They should cover the fundamental
needs for sequence creation.

.. tip::

   If the links to the classes on the table don't work, you can still access them via the sidebar.


.. autosummary::
   :toctree: _autosummary

   CompositeWaveform
   CustomWaveform
   ConstantWaveform
   RampWaveform
   BlackmanWaveform
   InterpolatedWaveform
   KaiserWaveform
   Pulse
   Register
   Register3D
   Sequence
   NoiseModel
   EmulatorConfig
   QPUBackend


Device Examples
-----------------

These are built-in :py:class:`~pulser.devices.Device` and :py:class:`~pulser.devices.VirtualDevice` instances that can be
imported directly from ``pulser``. 

.. important::

   These instances are **not** descriptions of actual devices. They are just examples that
   can be used to enforce different sets of constraints during :py:class:`~pulser.Sequence` creation.


.. autosummary::
   :toctree: _autosummary

   AnalogDevice
   DigitalAnalogDevice
   MockDevice


Modules
----------

.. autosummary::
   :toctree: _autosummary

   pulser.abstract_repr
   pulser.backend
   pulser.backends
   pulser.channels
   pulser.devices
   pulser.noise_model
   pulser.parametrized
   pulser.register
   pulser.result
   pulser.sampler
   pulser.sequence
   pulser.waveforms