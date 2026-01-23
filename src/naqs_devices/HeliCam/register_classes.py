"""
Registration of HeliCam device classes with the labscript framework.

This module registers the HeliCam device driver and its BLACS GUI tab
with the labscript suite, making them available for use in connection tables
and the BLACS control interface.
"""

import labscript_devices

labscript_device_name = 'HeliCam'
blacs_tab = 'naqs_devices.HeliCam.blacs_tabs.HeliCamTab'

labscript_devices.register_classes(
    labscript_device_name=labscript_device_name,
    BLACS_tab=blacs_tab,
)