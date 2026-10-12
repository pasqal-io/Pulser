# Quick Run Guide

:::{admonition} Onboarding
:class: note

This onboarding page gets you up and running with Pulser as quickly as possible: installation, running a job locally, and submitting to remote backends. It is intentionally focused and does not replace the full documentation. For an in-depth understanding, follow the full documentation in order.
:::

## Installation

Pulser requires Python 3.10 or later. Install it from [PyPI](https://pypi.org/project/pulser/) in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pulser
```

## Local Execution

Run your first experiment on a local emulator. No account or credentials are needed.

### 1. Define a pulse sequence

Create a register and a sequence on a device, then add a pulse to a channel. `MockDevice` is a virtual device for local runs.

```python
from pulser import (
    ConstantWaveform,
    MockDevice,
    Pulse,
    RampWaveform,
    Register,
    Sequence,
)

# Create the register (coordinates in µm)
register = Register.from_coordinates([(0, 6), (0, -6), (10, 0)], prefix="q")

# Define the pulse parameters
omega = 2.0  # rad/µs
delta_i = -2.0 * omega  # rad/µs
delta_f = -delta_i  # rad/µs
T = 3000  # ns

# Define the pulse
pulse = Pulse(
    amplitude=ConstantWaveform(T, omega),
    detuning=RampWaveform(T, delta_i, delta_f),
    phase=0.0,
)

# Create the sequence and add the pulse
seq = Sequence(register, MockDevice)
seq.declare_channel("rydberg_global", "rydberg_global")
seq.add(pulse, "rydberg_global")
```

### 2. Run on the local emulator

`QutipBackendV2` runs the sequence on your machine. `run()` returns the `Results` directly.

```python
from pulser.backends import QutipBackendV2

backend = QutipBackendV2(seq)
results = backend.run()
```

### 3. Choose a backend

Pulser supports several local [Pasqal emulators](https://docs.pasqal.com/qpu-emulators/emulators/), all accessible from `pulser.backends`:

- `QutipBackendV2` (default): based on QuTiP, best for up to ~15 qubits.
- `SVBackend`: PyTorch-based state-vector emulator, best for up to ~25 qubits. Requires the [`emu-sv`](https://docs.pasqal.com/qpu-emulators/emusv/) package.
- `MPSBackend`: PyTorch-based Matrix Product State emulator, best for 25 to ~80 qubits. Requires the [`emu-mps`](https://docs.pasqal.com/qpu-emulators/emumps/) package.

Install `emu-sv` and `emu-mps` with `pip install emu-sv emu-mps`, then pick a backend by swapping the class:

```python
from pulser.backends import SVBackend

backend = SVBackend(seq)
results = backend.run()
```

### 4. Inspect results

`backend.run()` returns a `Results` object. `get_result_tags()` lists what it contains, and `final_bitstrings` gives the sampled bitstrings:

```python
results.get_result_tags()   # ['bitstrings', 'state']
results.final_bitstrings    # Counter({'111': 566, '110': 155, '011': 111, ...})
```

## Remote Execution

### Connection

:::{admonition} Create a Connection
:class: note

Remote execution, whether it's on emulators or QPUs, requires a connection to submit a job. You can connect either through Pasqal Cloud or through one of our third-party cloud providers. If you haven't set up a connection yet, do so before continuing.

<div style="text-align: center">
  <a href="https://docs.pasqal.com/cloud/set-up" class="pulser-button pulser-button--primary">Pasqal Cloud</a>
  <a href="https://docs.pasqal.com/third-party-cloud-providers/#create-a-connection" class="pulser-button">Third-Party Providers</a>
</div>
:::

### Remote Execution on Cloud Emulators

#### 1. Create a connection

This example uses Pasqal Cloud: install the [`pasqal-cloud`](https://pypi.org/project/pasqal-cloud/) package with `pip install pasqal-cloud`, then create a `PasqalCloudConnection` with your username, password and project ID from the portal.

:::{tip}
**Using a third-party provider?**

If you chose a third-party cloud provider, create your connection by following that provider's instructions instead (for example, [Scaleway](https://docs.pasqal.com/third-party-cloud-providers/scaleway/#step-4-create-your-connection)), then continue from step 2, using it as `connection`.
:::

```python
from pasqal_cloud import PasqalCloudConnection

connection = PasqalCloudConnection(
    username=USERNAME,      # your user/email for the Pasqal Cloud Platform
    password=PASSWORD,      # your Pasqal Cloud Platform password
    project_id=PROJECT_ID,  # the ID of the project associated with your account
)
```

Replace `USERNAME`, `PASSWORD` and `PROJECT_ID` with your own values. Avoid hardcoding your password in code you share.

#### 2. Initialize a remote emulator and submit

`RemoteEmuFreeBackend` is free for all Pasqal Cloud accounts and needs no credits. `run()` submits the sequence and returns a `RemoteResults` object without waiting for the results.

```python
from pulser.backend import EmulationConfig
from pulser.backends import RemoteEmuFreeBackend

remote_backend = RemoteEmuFreeBackend(
    seq,
    connection=connection,
    config=EmulationConfig(default_num_shots=1000),
)
remote_results = remote_backend.run()
```

Remote emulator backends work as follows:

- Backend class: `RemoteEmuFreeBackend`, `RemoteSVBackend` or `RemoteMPSBackend`. Availability depends on your provider.
- `config`: an `EmulationConfig`, as for local emulators. See <a href="tutorials/backends.html#4.-Configuring-the-Emulation">Configuring the Emulation</a>.
- `default_num_shots` (set in `config`): number of bitstring samples to collect.

#### 3. Check status and retrieve results

Remote jobs run asynchronously, so you can check their status and fetch the results once they are done. Save the batch ID to retrieve the results later, for example from a new session.

```python
from pulser.backend import RemoteResults

# query status: PENDING, RUNNING, DONE, etc.
status = remote_results.get_batch_status()

# fetch the results (available once the batch is done)
results = remote_results[0]
results.final_bitstrings

# save the identifier for later retrieval
batch_id = remote_results.batch_id

# reconnect to the batch later
remote_results = RemoteResults(batch_id, connection)
```

### Remote Execution on a QPU

The same connection lets you run the sequence on a QPU.

#### 1. List available devices and adapt the sequence to the QPU

List the devices available through your connection, then adapt the sequence to the target QPU. QPUs require the register to be defined from a register layout, so give it one with `with_automatic_layout()` before switching to the QPU device.

```python
# list available devices
connection.fetch_available_devices()
# {'FRESNEL': ..., ...}

# adapt the sequence to the FRESNEL device
device = connection.fetch_available_devices()["FRESNEL"]
qpu_seq = seq.with_new_register(
    seq.register.with_automatic_layout(device)
).with_new_device(device)
```

#### 2. Submit to the QPU

Create a `QPUBackend` with the sequence, your connection and number of shots, then submit. Status checks and result retrieval work the same as for remote emulators.

```python
from pulser.backend import BackendConfig
from pulser.backends import QPUBackend

qpu = QPUBackend(
    qpu_seq,
    connection=connection,
    config=BackendConfig(default_num_shots=500),
)
remote_results = qpu.run()

# save the identifier for later retrieval
batch_id = remote_results.batch_id
```
