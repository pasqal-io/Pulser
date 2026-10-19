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

This example contains three atoms in a straight line 7 µm apart, and slowly sweeps the detuning from negative to positive. Adjacent atoms are inside each other's Rydberg blockade radius (here, about 10 µm), so they cannot both be excited. The two end atoms are far enough apart (14 µm) to be excited together.

```python
import numpy as np
from pulser import (
    ConstantWaveform,
    MockDevice,
    Pulse,
    RampWaveform,
    Register,
    Sequence,
)

# Create the register: 3 atoms in a line, 7 µm apart
register = Register.from_coordinates([(-7, 0), (0, 0), (7, 0)], prefix="q")

# Define the pulse parameters
omega = 2 * np.pi  # rad/µs
delta_i = -3.0 * omega  # rad/µs
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

### 2. Run on a local emulator

Pulser supports several [Pasqal emulators](https://docs.pasqal.com/qpu-emulators/emulators/):

- `QutipBackendV2`: based on QuTiP, best for up to ~15 qubits. Installed by default with `pulser`.
- `SVBackend`: PyTorch-based state-vector emulator, best for up to ~25 qubits. Requires the [`emu-sv`](https://docs.pasqal.com/qpu-emulators/emusv/) package.
- `MPSBackend`: PyTorch-based Matrix Product State emulator, best for 25 to ~80 qubits. Requires the [`emu-mps`](https://docs.pasqal.com/qpu-emulators/emumps/) package.

We use `QutipBackendV2` here since it comes packaged with `pulser` by default. `run()` executes the sequence on your machine and returns the `Results` directly:

```python
from pulser.backends import QutipBackendV2

backend = QutipBackendV2(seq)
results = backend.run()
```

To use another emulator, install it (`pip install emu-sv emu-mps`) and swap the class, for example `SVBackend(seq)`.

### 3. Inspect results

`backend.run()` returns a `Results` object. `get_result_tags()` lists what it contains, and `final_bitstrings` gives the sampled bitstrings:

```python
results.get_result_tags()   # ['bitstrings', 'state']
results.final_bitstrings    # Counter({'101': 810, '010': 61, '001': 57, ...})
```

`'101'` dominates because the middle atom blocks both of its neighbours, so the system has only the two end atoms excited. The exact counts vary between runs because the bitstrings are sampled.

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

!!! tip "Using a third-party provider?"

    If you chose a third-party cloud provider, create your connection by following that provider's instructions instead (for example, [Scaleway](https://docs.pasqal.com/third-party-cloud-providers/scaleway/#step-4-create-your-connection)), then continue from step 2, using it as `connection`.

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
    config=EmulationConfig(),
)
remote_results = remote_backend.run()
```

Remote emulator backends work as follows:

- Backend class: `RemoteEmuFreeBackend`(default), `RemoteSVBackend` or `RemoteMPSBackend`. Availability depends on your provider.
- `config`: an `EmulationConfig`, as for local emulators. See <a href="tutorials/backends.html#4.-Configuring-the-Emulation">Configuring the Emulation</a>.

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

List the devices available through your connection, then adapt the sequence to the target QPU.

```python
# list available devices
connection.fetch_available_devices()
# {'FRESNEL': ..., ...}

# adapt the sequence to the FRESNEL device
device = connection.fetch_available_devices()["FRESNEL"]
qpu_register = seq.register.with_automatic_layout(device)
qpu_seq = seq.with_new_register(qpu_register).with_new_device(device)
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
