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

## Example Connection Table

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
    "SensTqp": 4095,
    "SensNFrames": 16,
    "SensNavM2": 1,
    "CamMode": 0,
    "DdsGain": 2,
    "BSEnable": 0,
    "TrigFreeExtN": 0,
    'ExtTqp': 0,
    'EnTrigOnPos': 0,
    "TrigExtSrcSel": 0,
    "AcqStop": 0,
    "EnSynFOut": 1,
}
manual_settings = settings.copy()
manual_settings["TrigFreeExtN"] = 1

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

## The Device Driver

This labscript device driver is made for the Heliotis HeliCam C3 Lock-In Camera.
The important user controllable parameters in the GUI are:

| Parameter | Explanation | Range |
| --------- | ----------- | ----- |
| `SensNFrames` | The number of frames in a fully integrated image, called a volume | (10, 511) |
| `SensNavM2` | The number of demodulation cycles per frame is `SensNavM2`*2 + 2 | (0, 255) |
| `SensTqp` | The time quarter period of the sensor demodulation stage | (0, 4095) |

Note that the demodulation frequency $f_d$ is found from $f_d = \frac{f_{sensor}}{8(SensTqp + 30)}$ where the sensor frequency $f_{sensor}$ is 70 MHz.

The BLACS tab provides `snap` and `continuous` acquisitions. The logs will report
many calls to the LibHeLIC's `Acquire()`, which returns either a negative value (usually -116)
or the size of the buffer allocated.s

## How to document your device

To work within the labscript paradigm, we enforce that you write all
specification related documentation in the top-level README.md (here). Then,
any API related documentation should go in the `docs/index.rst`. The project
is structured and has the machinery so that the device can be hosted on Read
The Docs. This is optional and requires extra configuration steps as outlined
in the [RTD Docs](https://docs.readthedocs.com/platform/latest/tutorial/index.html).
