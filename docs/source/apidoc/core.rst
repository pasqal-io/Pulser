``pulser``
=================

.. automodule:: pulser

Classes
-----------------

These are classes that can be imported directly from ``pulser``. They should cover the fundamental
needs for sequence creation.



.. autosummary::
   :toctree: _autosummary

   ~pulser.waveforms.CompositeWaveform
   ~pulser.waveforms.CustomWaveform
   ~pulser.waveforms.ConstantWaveform
   ~pulser.waveforms.RampWaveform
   ~pulser.waveforms.BlackmanWaveform
   ~pulser.waveforms.InterpolatedWaveform
   ~pulser.waveforms.KaiserWaveform
   ~pulser.pulse.Pulse
   ~pulser.register.Register
   ~pulser.register.Register3D
   ~pulser.sequence.Sequence
   ~pulser.noise_model.NoiseModel
   ~pulser.backend.EmulatorConfig
   ~pulser.backend.QPUBackend


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
   pulser.exceptions
   pulser.parametrized
   pulser.register
   pulser.result
   pulser.sampler
   pulser.waveforms