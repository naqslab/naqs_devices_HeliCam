"""
BLACS GUI tab for Heliotis HeliCam C3 camera control.

This module provides the graphical user interface for controlling the HeliCam
camera within BLACS. Features include:

- Live image display with pyqtgraph ImageView (zoom, pan, level adjustment)
- Continuous acquisition mode for manual preview during experiment setup
- Single-shot "snap" mode for quick images
- Attribute editor dialog for viewing and modifying camera settings
- Frame rate indicator for monitoring acquisition performance
- Mode switching between buffered (experiment) and continuous (preview)

Components:
- ImageReceiver: ZMQ server for receiving images from worker process
- HeliCamTab: Main BLACS tab for user interaction

The tab communicates with the worker process via:
- ZMQ sockets for image transfer
- BLACS property system for device state

Camera attributes can be configured both in the connection table and via
the interactive attributes dialog during a shot. The dialog provides visibility
level filtering (simple/intermediate/advanced) for attribute discovery.

.. :no-index:

"""

import os
import json
from time import perf_counter
import ast
from queue import Empty

import labscript_utils.h5_lock
import h5py

import numpy as np

from qtutils import UiLoader, inmain_decorator
import qtutils.icons
from qtutils.qt import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from blacs.tab_base_classes import define_state, MODE_MANUAL
from blacs.device_base_class import DeviceTab

import labscript_utils.properties
from labscript_utils.ls_zprocess import ZMQServer
import logging


def exp_av(av_old, data_new, dt, tau):
    """Compute the new value of an exponential moving average based on the previous
    average av_old, a new value data_new, a time interval dt and an averaging timescale
    tau. Returns data_new if dt > tau"""
    if dt > tau:
        return data_new
    k = dt / tau
    return k * data_new + (1 - k) * av_old


class ImageReceiver(ZMQServer):
    """ZMQ server for receiving and displaying images from camera worker process.
    
    This server runs in the main BLACS GUI process and receives images from the
    camera worker process via ZMQ. When an image is received:
    
    1. It immediately replies 'ok' to the worker (allowing next acquisition)
    2. Updates the PyQtGraph ImageView with the new image
    3. Calculates and displays the frame rate
    
    The immediate reply enables pipelining - the worker can begin acquiring the
    next frame while the GUI renders the current one, without allowing a backlog
    to accumulate.
    
    Attributes:
        image_view: PyQtGraph ImageView widget to update with images
        label_fps: QLabel widget for displaying frame rate
        frame_rate: Current exponential moving average of frame rate
        last_frame_time: Timestamp of the last received frame
    """
    def __init__(self, image_view, label_fps):
        ZMQServer.__init__(self, port=None, dtype='multipart')
        self.image_view = image_view
        self.label_fps = label_fps
        self.last_frame_time = None
        self.frame_rate = None
        self.update_event = None

    @inmain_decorator(wait_for_return=True)
    def handler(self, data):
        # Acknowledge immediately so that the worker process can begin acquiring the
        # next frame. This increases the possible frame rate since we may render a frame
        # whilst acquiring the next, but does not allow us to accumulate a backlog since
        # only one call to this method may occur at a time.
        self.send([b'ok'])
        md = json.loads(data[0])
        image = np.frombuffer(memoryview(data[1]), dtype=md['dtype'])
        image = image.reshape(md['shape'])
        if len(image.shape) == 3 and image.shape[0] == 1:
            # If only one image given as a 3D array, convert to 2D array:
            image = image.reshape(image.shape[1:])
        this_frame_time = perf_counter()
        if self.last_frame_time is not None:
            dt = this_frame_time - self.last_frame_time
            if self.frame_rate is not None:
                # Exponential moving average of the frame rate over 1 second:
                self.frame_rate = exp_av(self.frame_rate, 1 / dt, dt, 1.0)
            else:
                self.frame_rate = 1 / dt
        self.last_frame_time = this_frame_time
        if self.image_view.image is None:
            # First time setting an image. Do autoscaling etc:
            self.image_view.setImage(image, autoRange=True, autoLevels=True)
            # Ensure histogram is properly initialized
            self.image_view.ui.histogram.setImageItem(self.image_view.getImageItem())
        else:
            # Updating image. Keep zoom/pan/levels/etc settings.
            self.image_view.setImage(
                image, autoRange=False, autoLevels=False
            )
        # Update fps indicator:
        if self.frame_rate is not None:
            self.label_fps.setText(f"{self.frame_rate:.01f} fps")

        # Tell Qt to send posted events immediately to prevent a backlog of paint events
        # and other low-priority events. It seems that we cannot make our qtutils
        # CallEvents (which are used to call this method in the main thread) low enough
        # priority to ensure all other occur before our next call to self.handler()
        # runs. This may be because the CallEvents used by qtutils.invoke_in_main have
        # their own event handler (qtutils.invoke_in_main.Caller), perhaps posted event
        # priorities are only meaningful within the context of a single event handler,
        # and not for the Qt event loop as a whole. In any case, this seems to fix it.
        # Manually calling this is usually a sign of bad coding, but I think it is the
        # right solution to this problem. This solves issue #36.
        QtWidgets.QApplication.instance().sendPostedEvents()
        return self.NO_RESPONSE


class HeliCamTab(DeviceTab):
    """BLACS GUI tab for HeliCam camera control.
    
    This tab provides the user interface for controlling a HeliCam camera within
    the BLACS framework via:
    
    - Live image preview in continuous acquisition mode
    - Single-shot "snap" image acquisition
    - Camera attribute configuration and inspection
    - Frame rate monitoring
    
    Layout is set by the blacs_tab.ui
    
    Attributes:
        image: PyQtGraph ImageView widget for image display
        image_receiver: ZMQServer instance for receiving images
        attributes_dialog: Dialog for viewing/editing camera attributes
        continuous_thread: Thread for continuous acquisition
        worker_class: Path to the BLACS worker class
        use_smart_programming: Whether to cache and skip unchanged attributes
    """
    # Subclasses may override this if all they do is replace the worker class with a
    # different one:
    worker_class = 'naqs_devices.HeliCam.blacs_workers.HeliCamWorker' 
    # Subclasses may override this to False if camera attributes should be set every
    # shot even if the same values have previously been set:
    use_smart_programming = False

    def initialise_GUI(self):
        # Populate line edits with initial values from camera attributes
        device = self.settings['connection_table'].find_by_name(self.device_name)
        
        init_freq = device.properties['manual_mode_camera_attributes']['SensTqp']
        
        print(f"{init_freq=}")
        
        
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'blacs_tab.ui'
        )
        attributes_ui_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'attributes_dialog.ui'
        )
        self.ui = UiLoader().load(ui_filepath)
        self.ui.pushButton_continuous.clicked.connect(self.on_continuous_clicked)
        self.ui.pushButton_stop.clicked.connect(self.on_stop_clicked)
        self.ui.pushButton_snap.clicked.connect(self.on_snap_clicked)
        self.ui.pushButton_attributes.clicked.connect(self.on_attributes_clicked)
        self.ui.toolButton_nomax.clicked.connect(self.on_reset_rate_clicked)
        

        self.attributes_dialog = UiLoader().load(attributes_ui_filepath)
        self.attributes_dialog.setParent(self.ui.parent())
        self.attributes_dialog.setWindowFlags(QtCore.Qt.Tool)
        self.attributes_dialog.setWindowTitle("{} attributes".format(self.device_name))
        self.attributes_dialog.pushButton_copy.clicked.connect(self.on_copy_clicked)
        self.attributes_dialog.comboBox.currentIndexChanged.connect(
            self.on_attr_visibility_level_changed
        )
        self.ui.doubleSpinBox_maxrate.valueChanged.connect(self.on_max_rate_changed)

        layout.addWidget(self.ui)
        self.image = pg.ImageView()
        self.image.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.ui.gridLayout.addWidget(self.image)
        self.ui.pushButton_stop.hide()
        self.ui.doubleSpinBox_maxrate.hide()
        self.ui.toolButton_nomax.hide()
        self.ui.label_fps.hide()
        
        # use AO widgets to mimick input field functionality
        analog_properties = {
            'SensTqp':{
                'base_unit':'arb',
                'min':int(1),
                'max':int(4095),
                'step':int(1),
                'decimals':0
                },
            'SensNavM2':{
                'base_unit':'arb',
                'min':int(1),
                'max':int(255),
                'step':int(1),
                'decimals':0
                },
            'SensNFrames':{
                'base_unit':'arb',
                'min':int(1),
                'max':int(512),
                'step':int(1),
                'decimals':0
                },
            }

        self.create_analog_outputs(analog_properties)
        ao_widgets = self.create_analog_widgets(analog_properties)
        self.auto_place_widgets(('Lock-in Settings', ao_widgets))

        # Ensure the GUI reserves space for these widgets even if they are hidden.
        # This prevents the GUI jumping around when buttons are clicked:
        for widget in [
            self.ui.pushButton_stop,
            self.ui.doubleSpinBox_maxrate,
            self.ui.toolButton_nomax,
        ]:
            size_policy = widget.sizePolicy()
            if hasattr(size_policy, 'setRetainSizeWhenHidden'): # Qt 5.2+ only
                size_policy.setRetainSizeWhenHidden(True)
                widget.setSizePolicy(size_policy)

        # Start the image receiver ZMQ server:
        self.image_receiver = ImageReceiver(self.image, self.ui.label_fps)
        self.acquiring = False

        self.supports_smart_programming(self.use_smart_programming) 

    def get_save_data(self):
        return {
            'attribute_visibility': self.attributes_dialog.comboBox.currentText(),
            'acquiring': self.acquiring,
            'max_rate': self.ui.doubleSpinBox_maxrate.value(),
            'colormap': repr(self.image.ui.histogram.gradient.saveState())
        }

    def restore_save_data(self, save_data):
        self.attributes_dialog.comboBox.setCurrentText(
            save_data.get('attribute_visibility', 'simple')
        )
        self.ui.doubleSpinBox_maxrate.setValue(save_data.get('max_rate', 0))
        if save_data.get('acquiring', False):
            # Begin acquisition
            self.on_continuous_clicked(None)
        if 'colormap' in save_data:
            try:
                self.image.ui.histogram.gradient.restoreState(
                    ast.literal_eval(save_data['colormap'])
                )
            except (ValueError, SyntaxError) as e:
                # Saved colormap may be incompatible or malformed (e.g., contains
                # Color() objects). Just use the default grayscale colormap.
                logger = logging.getLogger(__name__)
                logger.debug('Could not restore saved colormap state: %s', str(e))


    def initialise_workers(self):
        table = self.settings['connection_table']
        connection_table_properties = table.find_by_name(self.device_name).properties
        # The device properties can vary on a shot-by-shot basis, but at startup we will
        # initially set the values that are configured in the connection table, so they
        # can be used for manual mode acquisition:
        with h5py.File(table.filepath, 'r') as f:
            device_properties = labscript_utils.properties.get(
                f, self.device_name, "device_properties"
            )
        worker_initialisation_kwargs = {
            'serial_number': connection_table_properties['serial_number'],
            'orientation': connection_table_properties['orientation'],
            'camera_attributes': device_properties['camera_attributes'],
            'manual_mode_camera_attributes': connection_table_properties[
                'manual_mode_camera_attributes'
            ],
            'mock': connection_table_properties['mock'],
            'image_receiver_port': self.image_receiver.port,
        }
        self.create_worker(
            'main_worker', self.worker_class, worker_initialisation_kwargs
        )
        self.primary_worker = "main_worker"

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def update_attributes(self):
        attributes_text = yield (
            self.queue_work(
                self.primary_worker,
                'get_attributes_as_text',
                self.attributes_dialog.comboBox.currentText(),
            )
        )
        self.attributes_dialog.plainTextEdit.setPlainText(attributes_text)

    def on_attributes_clicked(self, button):
        self.attributes_dialog.show()
        self.on_attr_visibility_level_changed(None)

    def on_attr_visibility_level_changed(self, value):
        self.attributes_dialog.plainTextEdit.setPlainText("Reading attributes...")
        self.update_attributes()

    def on_continuous_clicked(self, button):
        self.ui.pushButton_snap.setEnabled(False)
        self.ui.pushButton_attributes.setEnabled(False)
        self.ui.pushButton_continuous.hide()
        self.ui.pushButton_stop.show()
        self.ui.doubleSpinBox_maxrate.show()
        self.ui.toolButton_nomax.show()
        self.ui.label_fps.show()
        self.ui.label_fps.setText('? fps')
        self.acquiring = True
        max_fps = self.ui.doubleSpinBox_maxrate.value()
        dt = 1 / max_fps if max_fps else 0
        self.start_continuous(dt)

    def on_stop_clicked(self, button):
        self.ui.pushButton_snap.setEnabled(True)
        self.ui.pushButton_attributes.setEnabled(True)
        self.ui.pushButton_continuous.show()
        self.ui.doubleSpinBox_maxrate.hide()
        self.ui.toolButton_nomax.hide()
        self.ui.pushButton_stop.hide()
        self.ui.label_fps.hide()
        self.acquiring = False
        self.stop_continuous()

    def on_copy_clicked(self, button):
        text = self.attributes_dialog.plainTextEdit.toPlainText()
        clipboard = QtWidgets.QApplication.instance().clipboard()
        clipboard.setText(text)

    def on_reset_rate_clicked(self):
        self.ui.doubleSpinBox_maxrate.setValue(0)

    def on_max_rate_changed(self, max_fps):
        if self.acquiring:
            self.stop_continuous()
            dt = 1 / max_fps if max_fps else 0
            self.start_continuous(dt)

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def on_snap_clicked(self, button):
        yield (self.queue_work(self.primary_worker, 'snap'))
        
    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def start_continuous(self, dt):
        yield (self.queue_work(self.primary_worker, 'start_continuous', dt))

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def stop_continuous(self):
        yield (self.queue_work(self.primary_worker, 'stop_continuous'))

    def restart(self, *args, **kwargs):
        # Must manually stop the receiving server upon tab restart, otherwise it does
        # not get cleaned up:
        try:
            self.image_receiver.shutdown()
        except TypeError as e:
            # zprocess/zmq version incompatibility: sock.close(linger=True) fails
            # because zmq expects int, not bool. This is just cleanup, so we can safely ignore it.
            import logging
            logger = logging.getLogger(__name__)
            logger.warning('TypeError during image_receiver shutdown (zmq/zprocess version incompatibility): %s', str(e))
        return DeviceTab.restart(self, *args, **kwargs)
