# naqs_devices_HeliCam

## Directory structure

```text

└── naqs_devices_HeliCam/
    ├── .gitignore
    ├── pyproject.toml
    ├── README.md
    ├── LICENSE.txt
    ├── CITATION.cff
    ├── docs/
    │   ├── conf.py
    │   ├── make.bat
    │   ├── Makefile
    │   └── index.rst
    └── src/naqs_devices/
        └── HeliCam/
            ├── __init__.py
            ├── blacs_tabs.py
            ├── blacs_workers.py
            ├── labscript_devices.py
            ├── register_classes.py
            └── runviewer_parsers.py # TODO?
```

## Example Labscript usage

### Example Connection Tables

#### Normal mode

```python

from labscript import *
from naqs_devices.HeliCam.labscript_devices import HeliCam
from labscript_devices.PrawnBlaster.labscript_devices import PrawnBlaster
from labscript_devices.PrawnDO.labscript_devices import PrawnDO

prawn = PrawnBlaster(
    name="prawn",
    com_port="COM5",
    pico_board="pico2",
    num_pseudoclocks=1,
    clock_frequency=150e6,
)

prawn_do = PrawnDO(
    name="prawn_do",
    com_port="COM3",
    clock_line=prawn.clocklines[0],
    external_clock=False,
)

settings = {
    "SensTqp": 4095, # Demod freq = 2121 Hz
    "SensNFrames": 16, # Number of frames per burst
    "SensNavM2": 1, # Proportional to num demod cycles
    "CamMode": 0, # RawIQ
    "DdsGain": 2, # Gain of 1, level 2 of 3
    "BSEnable": 0, # Background subtraction
    "TrigFreeExtN": 0, # External timing
    'ExtTqp': 0, # Internal Tqp
    'EnTrigOnPos': 0, # No translation stage
    "TrigExtSrcSel": 0, # Default ext trigger source
    "AcqStop": 0, # Ensure acquisition is running
    "EnSynFOut": 1, # Provides signal at demod freq on OUT3
}
manual_settings = settings.copy()
manual_settings["TrigFreeExtN"] = 1 # Manual mode requires internal timing

camera = HeliCam(
    name="helicam",
    parent_device=prawn_do.outputs,
    connection="do0",
    serial_number="008650",
    camera_attributes=settings,
    manual_mode_camera_attributes=manual_settings,
    trigger_duration=2000e-6,
)


if __name__ == "__main__":
    start()
    stop(1)

```

#### ExtTqp mode

```python

from labscript import *
from naqs_devices.HeliCam.labscript_devices import HeliCam
from labscript_devices.PrawnBlaster.labscript_devices import PrawnBlaster
from labscript_devices.PrawnDO.labscript_devices import PrawnDO

prawn = PrawnBlaster(
    name="prawn",
    com_port="COM5",
    pico_board="pico2",
    num_pseudoclocks=1,
    clock_frequency=150e6,
)

prawn_do = PrawnDO(
    name="prawn_do",
    com_port="COM3",
    clock_line=prawn.clocklines[0],
    external_clock=False,
)

settings = {
    "SensTqp": 4095, # Demod freq = 2121 Hz
    "SensNFrames": 16, # Number of frames per burst
    "SensNavM2": 1, # Proportional to num demod cycles
    "CamMode": 0, # RawIQ
    "DdsGain": 2, # Gain of 1, level 2 of 3
    "BSEnable": 0, # Background subtraction
    "TrigFreeExtN": 0, # External timing
    'ExtTqp': 1, # External Tqp
    'ExtTqpPuls': 1, # Enable PhiA and PhiB on Encoder
    'EnTrigOnPos': 0, # No translation stage
    "TrigExtSrcSel": 0, # Default ext trigger source
    "AcqStop": 0, # Ensure acquisition is running
    "EnSynFOut": 0, # Has to be off for ExtTqp mode
}
manual_settings = settings.copy()
manual_settings["TrigFreeExtN"] = 1 # Manual mode requires internal timing

camera = HeliCam(
    name="helicam",
    parent_device=prawn_do.outputs,
    connection="do0",
    serial_number="008650",
    camera_attributes=settings,
    manual_mode_camera_attributes=manual_settings,
    trigger_duration=2000e-6,
)


if __name__ == "__main__":
    start()
    stop(1)

```

### Example Experiment script

In order to not have to manually find the `SensTqp` register to correspond to
your desired demodulation frequency, use a runmanager global and the
`frequency_to_tqp` method. The worker will then set the settings before the
shot runs.

```python

# with the same contents as the above connection table(s), as per the labscript paradigm

if __name__ == "__main__":
    # Capital variables are runmanager globals
    start()

    camera.camera_attributes["SensTqp"] = camera.frequency_to_tqp(DEMOD_FREQ)
    camera.camera_attributes["SensNFrames"] = N_FRAMES
    
    print(f"RUNMANAGER says: {camera.camera_attributes}")
    t = 0
    for i in range(N_ACQUISITIONS):
        t += DELTA_T
        camera.expose(t, f"snap{i}")
        add_time_marker(t, f"snap{i}", verbose=True)

    t += 1
    stop(t)

```

### Example analysis script

```python
# view_heli_images.py
import lyse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

df = lyse.data()
h5_path = df.filepath.iloc[-1]

run = lyse.Run(h5_path)

run_globals = run.get_globals()
N = run_globals.get("N_ACQUISITIONS", 1)

cols = int(np.ceil(np.sqrt(N)))
rows = int(np.ceil(N / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
axes = axes.flatten()

for i in range(N):
    ims = run.get_image('helicam', f'snap{i}', 'frame')
    axes[i].imshow(ims, cmap='gray')
    axes[i].set_title(f'frame {i + 1}')
    axes[i].axis('off')

for j in range(N, len(axes)):
    axes[j].axis('off')
    
plt.tight_layout()

```

## The Device Driver

This labscript device driver is made for the Heliotis HeliCam C3 Lock-In Camera.
The important user controllable parameters in the GUI are:

| Parameter | Explanation | Range |
| --------- | ----------- | ----- |
| `SensNFrames` | The number of frames in a fully integrated image, called a volume | (10, 511) |
| `SensNavM2` | The number of demodulation cycles per frame $N_C$ is `SensNavM2`*2 + 2 | (0, 255) |
| `SensTqp` | The time quarter period of the sensor demodulation stage | (0, 4095) |

Note that the demodulation frequency $f_d$ is found from $f_d = \frac{f_{sensor}}{8(SensTqp + 30)}$ where the sensor frequency $f_{sensor}$ is 70 MHz.

The BLACS tab provides `snap` and `continuous` acquisitions. The logs will report
many calls to the LibHeLIC's `Acquire()`, which returns either a negative value (usually -116)
or the size of the buffer allocated.

## Device specs testing

In testing, we have found that the C3's purported framerate upper limit of
3800 fps is not always achievable given settings that follow the stated
specification of:

$FPS = \frac{1}{\Delta_{T_{frame}}} = \frac{f_{demod}}{N_C} + T_{offset}$,

where $T_{offset}$ is set by the background subtraction registers. If background
subtraction is not used, $T_{offset} = 0$. A [testing script](testing/find_framerate.py)
is provided that explores the outer product of the parameter space of `SensNFrames`,
`SensNavM2`, and `SensTqp` in an attempt to characterize the limits to both
framerate and acquisition rate. The method employs a binary search algorithm and
reports the fastest successful framerate and acquisition rate. Due to the nature
of the algorithm's convergence, one may possibly find faster parameters by increasing
either the tolerance or number of desired iterations.

## How to document your device

To work within the labscript paradigm, we enforce that you write all
specification related documentation in the top-level README.md (here). Then,
any API related documentation should go in the `docs/index.rst`. The project
is structured and has the machinery so that the device can be hosted on Read
The Docs. This is optional and requires extra configuration steps as outlined
in the [RTD Docs](https://docs.readthedocs.com/platform/latest/tutorial/index.html).
