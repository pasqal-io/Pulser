import numpy as np
import qutip
import torch

from pulser import Register, Pulse, Sequence, BlackmanWaveform
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser import NoiseModel

natoms = 1
reg = Register.from_coordinates([(0, 0)], prefix="q")

seq = Sequence(reg, DigitalAnalogDevice)
seq.declare_channel("ch0", "rydberg_global")
duration = 500
pulse = Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi), 0.0, 0)
seq.add(pulse, "ch0")

# define an observable for the state
# |x> = |0> leakage state with computation basis: |g> =|1>, |r>=|2>
# up_qtrit =   # ground state projector
# listident_qtrit = [qutip.qeye(3)] * natoms
# listident_qtrit[0] = up_qtrit
obs_qtrit = qutip.basis(3, 0).proj()  # observable for the first atom
print(obs_qtrit)
print("observable for the first atom:", obs_qtrit)

# define noise model
eff_rate = [0.5]
eff_ops = [qutip.basis(3, 2) @ qutip.basis(3, 1).dag()]
noise_model = NoiseModel(
    eff_noise_rates=eff_rate,
    eff_noise_opers=eff_ops,
    with_leakage=True,
)

# check the noise model
print(noise_model.noise_types)

sim = QutipEmulator.from_sequence(
    seq, sampling_rate=0.1, noise_model=noise_model
)
sim.set_evaluation_times(0.5)
noisy_res = sim.run()

print("Result type:", type(noisy_res))
print("Uses pseudo density:", noisy_res._use_pseudo_dens)

final_state = noisy_res.get_final_state()
print("final state:", final_state)  # print the final state

print(noisy_res.sample_final_state(10000))

print("expectation value:", qutip.expect(obs_qtrit, final_state))

print(noisy_res.expect([obs_qtrit])[0][-1])
