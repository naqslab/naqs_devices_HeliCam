# naqs_devices_HeliCam

## Directory structure

```text

└── naqs_devices_HeliCam/
    ├── .gitignore
    ├── pyproject.toml
    ├── README.md
    ├── LICENSE.txt # Choose a license, labscript uses BSD
    ├── CITATION.cff # Optional to define citation for citing the device repository
    ├── docs/
    │   ├── conf.py
    │   ├── make.bat
    │   ├── Makefile
    │   └── index.rst
    └── src/naqs_devices/ # note: must be same as in the parent naqs_devices repo to be in the same namespace
        └── HeliCam/
            ├── __init__.py
            ├── blacs_tabs.py
            ├── blacs_workers.py
            ├── labscript_devices.py
            ├── register_classes.py
            └── runviewer_parsers.py
```

## Example Connection Table

```python

from labscript import *
from labscript_devices.HeliCam.labscript_devices import HeliCam
from labscript_devices.DummyPseudoclock.labscript_devices import DummyPseudoclock
from labscript_devices.DummyIntermediateDevice import DummyIntermediateDevice


dummy_clock = DummyPseudoclock(name="dummy_clock", BLACS_connection="dummy")
dummy_daq = DummyIntermediateDevice(
    name="dummy_device", BLACS_connection="dummy2", parent_device=dummy_clock.clockline
)

settings={
    "SensTqp": 4095,
    "SensNFrames": 16,
    "SensNavM2": 255,
    "CamMode": 0,
    "DdsGain": 2,
    "BSEnable": 0,
    "TrigFreeExtN": 1,
    "TrigExtSrcSel": 0,
    "AcqStop": 0,
    "EnSynFOut": 1,
}

camera = HeliCam(
    name="helicam",
    parent_device=dummy_daq,
    connection="c3cam_s170",
    serial_number="008650",
    camera_attributes=settings,
    manual_mode_camera_attributes=settings,
    trigger_duration=2e-6
)

if __name__ == "__main__":
    start()

    stop(1)


```

## How to document your device

To work within the labscript paradigm, we enforce that you write all
specification related documentation in the top-level README.md (here). Then,
any API related documentation should go in the `docs/index.rst`. The project
is structured and has the machinery so that the device can be hosted on Read
The Docs. This is optional and requires extra configuration steps as outlined
in the [RTD Docs](https://docs.readthedocs.com/platform/latest/tutorial/index.html).
