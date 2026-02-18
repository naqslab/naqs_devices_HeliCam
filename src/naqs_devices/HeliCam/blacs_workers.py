"""
BLACS worker for Heliotis HeliCam C3 camera control.

This module provides the BLACS worker implementation for controlling the HeliCam
camera via hardware triggers. It includes:

- HeliCamInterface: Interface class for the HeliCam hardware using libHeLIC
- MockCamera: Mock camera implementation for testing without hardware
- HeliCamWorker: BLACS worker that manages camera lifecycle and image acquisition

The worker handles:

- Continuous and buffered acquisition modes
- Manual mode snapping for live viewing
- Automatic image saving to HDF5 files
- Smart attribute caching to minimize camera reprogramming
- Thread-based image acquisition with timeout handling

Dependencies:

- libHeLIC (Heliotis HeliCam C3 library and Python wrapper)
- h5py, numpy, threading, zmq, ctypes

.. :no-index:

"""

import sys
from time import perf_counter
from blacs.tab_base_classes import Worker
import threading
import numpy as np
from labscript_utils import dedent
from labscript import LabscriptError
import labscript_utils.h5_lock
import h5py
import labscript_utils.properties
import zmq
import ctypes as ct
import os
from enum import IntEnum
from labscript_utils.ls_zprocess import Context
from labscript_utils.shared_drive import path_to_local
from labscript_utils.properties import set_attributes
# Import libHeLIC at module level so it's available for all classes
try:
    prgPath = os.environ["PROGRAMFILES"]

    sys.path.insert(0, prgPath + r"\Heliotis\heliCam\Python\wrapper")
    from libHeLIC import LibHeLIC  # noqa: E402
except ImportError:
    raise LabscriptError("Could not find LibHeLIC, is HeliSDK properly installed?")

# A file local enum that should probably be moved to a class attr
class CamMode(IntEnum):
    RAW_IQ = 0
    AMPLITUDE = 1
    SMOOTH_AMPLITUDE = 2
    INTENSITY = 3 # Manual calls this "intensity", what they mean is HDR
    SIMPLE_MAX = 4
    MIN_ENERGY = 7
    
class MockCamera(object):
    """Mock camera class that returns fake image data."""

    def __init__(self):
        print("Starting device worker as a mock device")
        self.attributes = {}
        self.exception_on_failed_shot = True
        self.heSys = None

    def set_attributes(self, attributes):
        self.attributes.update(attributes)

    def get_attribute(self, name):
        return self.attributes[name]

    def get_attribute_names(self, visibility_level=None):
        return list(self.attributes.keys())

    def configure_acquisition(self, continuous=False, bufferCount=5):
        pass

    def grab(self):
        return self.snap(integrated_frames=1, settings=self.attributes)

    def grab_multiple(self, n_images, images, waitForNextBuffer=True):
        print(f"Attempting to grab {n_images} (mock) images.")
        for i in range(n_images):
            images.append(self.grab())
            print(f"Got (mock) image {i+1} of {n_images}.")
        print(f"Got {len(images)} of {n_images} (mock) images.")

    def snap(self, integrated_frames, settings):
        N = 300
        A = 300
        x = np.linspace(-5, 5, 300)
        y = x.reshape((N, 1))
        clean_image = A * (1 - 0.5 * np.exp(-(x ** 2 + y ** 2)))

        # Write text on the image that says "NOT REAL DATA"
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default()
        canvas = Image.new('L', [N // 5, N // 5], (0,))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 20), "NOT REAL DATA", font=font, fill=1)
        clean_image += 0.2 * A * np.asarray(canvas.resize((N, N)).rotate(20))
        return np.random.poisson(clean_image)
    
    def stop_acquisition(self):
        pass

    def abort_acquisition(self):
        pass

    def close(self):
        pass


class HeliCamInterface(object):
    """Interface to the Heliotis HeliCam C3 camera via libHeLIC.
    
    This class wraps the libHeLIC library to control the HeliCam, providing
    methods for attribute configuration, image acquisition, and camera control.
    
    The HeliCam is controlled via the libHeLIC Python library which communicates
    with the camera's firmware via USB. The most important registers are:
    
    - SensNFrames: Number of frames to acquire
    - SensTqp: Time quarter period (to convert to demodulation frequency)
    - CamMode: Currently supported: 0: Raw IQ, 3: Intensity (HDR)
    - SensNavM2: Number of demodulation cycles per frame
    - DdsGain: Analogue signal gain of the sensor. 
    Heliotis claims through testing DdsGain=2 (Effective gain of 1) has yielded the best SNR.
    - BSEnable: Background subtraction
    - TrigExtSrcSel: External trigger source selection
    
    Attributes:
        heSys: The libHeLIC camera instance
        exception_on_failed_shot: Whether to raise on acquisition timeout
        _abort_acquisition: Flag to abort ongoing acquisition
        attributes: Dictionary tracking current attribute values
    
    Note:
        The HeliCam C3 sensor has a maximum resolution of 300x300 pixels.
    """
    # Maximum sensor resolution for HeliCam C3
    MAX_SENSOR_WIDTH = 300
    MAX_SENSOR_HEIGHT = 300
    
    def __init__(self, serial_number, initial_attributes):
        if LibHeLIC is None:
            raise RuntimeError(
                "libHeLIC not found. Please install the Heliotis HeliCam Python library."
            )
        
        print("Initializing HeliCam via libHeLIC...")
        self.heSys = LibHeLIC()
        
        # Open camera - note: serial_number argument is for future compatibility
        # Current libHeLIC implementation uses device index
        print("Opening camera...")
        try:
            self.heSys.Open(0, sys="c3cam_sl70")
        except Exception as e:
            msg = f"Failed to open camera: {e}"
            raise RuntimeError(msg) from e
        
        res = self.heSys.Acquire()
        print(f"Initial Acquire returned: {res}")
        
        self.exception_on_failed_shot = True
        self._abort_acquisition = False
        # self.attributes = {}
        self.attributes = dict(initial_attributes)
        # self.configure_acquisition()
        print("HeliCam connected successfully")
        
        # print(self.heSys.CamDataHdr.nVolume.size)
        # print(self.heSys.CamDataHdr.nVolume.offset)
        # print(self.heSys.CamDataHdr.nFrames.size)
        # print(self.heSys.CamDataHdr.nFrames.offset)

    def set_attributes(self, attr_dict):
        """Set multiple camera attributes from a dictionary"""
        for k, v in attr_dict.items():
            self.set_attribute(k, v)

    def set_attribute(self, name, value):
        """Set a single camera attribute by name"""
        try:
            setattr(self.heSys.map, name, value)
            self.attributes[name] = value
        except Exception as e:
            msg = f"Failed to set attribute {name} to {value}: {e}"
            raise RuntimeError(msg) from e

    def get_attribute(self, name):
        """Get the current value of a camera attribute"""
        try:
            value = getattr(self.heSys.map, name, None)
            if value is None:
                # Try to read from stored attributes
                value = self.attributes.get(name)
            return value
        except Exception as e:
            raise RuntimeError(f"Failed to get attribute {name}: {e}") from e

    def get_attribute_names(self, visibility_level=None):
        """Return list of available attribute names"""
        rd = self.heSys.GetRegDesc()

        attribute_dict = self.heSys.MapByName(rd, self.heSys).as_dict()
        if visibility_level=="Advanced":
            return [str(x.decode()) for x in attribute_dict.keys()]
        else:
            return list(self.attributes.keys())
    
    def get_demod_freq(self, tqp):
        f = (70e6 / 8) * (1 / (tqp + 30))
        return f # Hz
    
    def get_tqp(self, demod_freq):
        tqp = 70e6 / (8 * demod_freq) - 30
        return tqp
    
    def snap(self, integrated_frames, settings: tuple | dict):
        """Acquire a single image and return it as numpy array"""

        self.configure_acquisition(continuous=False, bufferCount=1)
        res = self.heSys.Acquire()
        print("Acquire returned:", res)
        cd = self.heSys.ProcessCamData(idx=1, mode=0, param=0)
        print("ProcessCamData returned", cd.contents.data)
        img = self.heSys.GetCamData(idx=1, addRef=0, meta=0)
        
        data = img.contents.data
        data = LibHeLIC.Ptr2Arr(data, (integrated_frames, 300, 300, 2), ct.c_int16)
        amplitude = data[1:,:,:,:].sum(axis=3, dtype=np.int16)
        return amplitude

    def configure_acquisition(self, continuous=True, bufferCount=1):
        """Configure camera for acquisition (buffered or continuous)"""
        settings = self.attributes.copy()
        if isinstance(settings, tuple):
            settings = dict(settings)
        
        print('Initial setting of attributes in configure')
        self.set_attributes(settings)

        print(f'Allocating data for CamMode : {settings["CamMode"]}')
        if settings["CamMode"] == 0:
            self.heSys.AllocCamData(
                idx=1,
                format=LibHeLIC.CamDataFmt["DF_I16Q16"],
                prop=0,
                extData=0,
                extDataSz=0,
            )
        elif settings["CamMode"] == 3:
            self.heSys.AllocCamData(
                idx=1,
                format=LibHeLIC.CamDataFmt["DF_Hf"],
                prop=0,
                extData=0,
                extDataSz=0,
            )
        else:
            raise LabscriptError(
                "Invalid CamMode, supported are: 0 - RAW_IQ, 3 - INTENSITY (HDR)"
            )

        print(
            f"Configured for acquisition (continuous={continuous}, buffers={bufferCount})"
        )

    def grab(self, waitForNextBuffer=True, skipAcquire=False):
        """Acquire a single frame, returns image as np.array
        """
        try:
            if self._abort_acquisition:
                raise RuntimeError("Acquisition aborted")
            
            num_frames = self.attributes.get('SensNFrames')
            num_frames = int(num_frames)

            if not skipAcquire:
                res = self.heSys.Acquire()
                print(f"Acquire in grab() returned: {res}")
            cd = self.heSys.ProcessCamData(1, 0, 0)
            img = self.heSys.GetCamData(1, 0, 0)

            data = img.contents.data
            
            data_array = LibHeLIC.Ptr2Arr(
                data, (num_frames, 300, 300, 2), ct.c_int16
            )
            
            # This is distinctly different from snap since all frames are to
            # be integrated over
            amplitude = data_array.sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)

            return amplitude
        
        except Exception as e:
            raise RuntimeError(f"Failed to grab frame: {e}") from e

    def grab_multiple(self, n_images, images, exposures=None, waitForNextBuffer=True):
        """Acquire multiple frames and append to images list"""
        print(f"Attempting to grab {n_images} images.")
        
        # Note: the following logic reimplements Imaqdx's waitForNextBuffer
        # I should make sure that it is True for external timing, but not for 
        # internal
        
        # force -1 since Acquire's "error" state will always be -116
        res = -1
        for i in range(n_images):
            if self._abort_acquisition:
                print("Abort during acquisition.")
                self._abort_acquisition = False
                break
            try:
                
                # catch both cases where we're not triggered and not ready to 
                # return an image
                while res < 1:
                    print(f'waiting for trigger, got {res}')
                    res = self.heSys.Acquire()
                    
                # exiting the while means Acquire should be responding with 
                # the size of the buffer *with* an image inside
                print(f'Trigger received, got {res}')
                images.append(self.grab(waitForNextBuffer, skipAcquire=True))
                print(f"Got image {i+1} of {n_images}.")
                
                # Reset our error handler variable
                res = -1
            except Exception as e:
                if self.exception_on_failed_shot:
                    raise
                else:
                    print(f"Warning: Failed to grab image {i+1}: {e}", file=sys.stderr)
                    break
        
        print(f"Got {len(images)} of {n_images} images.")

    def stop_acquisition(self):
        """Stop image acquisition"""
        try:
            # I have not found anything in the SDK that will actually "Stop"
            # an acquisition, but it's mostly frame by frame anyways
            
            print("Acquisition stopped")
        except Exception as e:
            raise RuntimeError(f"Failed to stop acquisition: {e}") from e

    def abort_acquisition(self):
        """Abort ongoing acquisition"""
        self._abort_acquisition = True

    def _ensure_numpy(self, img):
        """Convert image to numpy array if needed"""
        if isinstance(img, np.ndarray):
            return img.copy()
        # If it's a pointer or other type, try conversion
        try:
            return np.array(img)
        except Exception:
            raise RuntimeError(f"Cannot convert image to numpy array: {type(img)}")

    def close(self):
        """Close the camera connection"""
        try:
            if hasattr(self, 'heSys') and self.heSys is not None:
                self.heSys.Close()
                print("Camera closed")
        except Exception as e:
            print(f"Warning: Error closing camera: {e}", file=sys.stderr)


class HeliCamWorker(Worker):
    """BLACS worker for HeliCam camera control and image acquisition.
    
    This worker manages the complete lifecycle of a HeliCam camera including:
    
    - Hardware initialization and attribute configuration
    - Continuous live preview acquisition in manual mode
    - Buffered acquisition synchronized with hardware triggers
    - Automatic HDF5 file storage of acquired images
    - Smart attribute caching to minimize unnecessary reprogramming
    
    The worker runs in a separate process (via BLACS) and communicates with
    the GUI via:
    - ZMQ sockets for image transfer to the parent GUI process
    - HDF5 files for experiment configuration and image storage
    - Property dictionaries for camera attribute settings
    
    Attributes:
        camera: Instance of the HeliCamInterface interface class
        interface_class: The camera interface class to instantiate
        smart_cache: Dictionary tracking which attributes have been set
        image_socket: ZMQ socket for sending images to parent GUI
        continuous_thread: Thread for continuous manual mode acquisition
        acquisition_thread: Thread for buffered acquisition
    """
    # Subclasses may override this if their interface class takes only the serial number
    # as an instantiation argument, otherwise they may reimplement get_camera():
    interface_class = HeliCamInterface

    def init(self):
        self.camera = self.get_camera()
        self.logger.info("Setting attributes...")
        self.smart_cache = {}
        self.set_attributes_smart(self.camera_attributes)
        self.set_attributes_smart(self.manual_mode_camera_attributes)
        self.images = None
        self.n_images = None
        self.attributes_to_save = None
        self.exposures = None
        self.acquisition_thread = None
        self.h5_filepath = None
        self.stop_acquisition_timeout = None
        self.exception_on_failed_shot = None
        self.continuous_stop = threading.Event()
        self.continuous_thread = None
        self.continuous_dt = None
        self.image_socket = Context().socket(zmq.REQ)
        self.image_socket.connect(
            f'tcp://{self.parent_host}:{self.image_receiver_port}'
        )
        self.logger.info("Initialisation complete")

    def get_camera(self):
        """Return an instance of the camera interface class. Subclasses may override
        this method to pass required arguments to their class if they require more
        than just the serial number."""
        if self.mock:
            return MockCamera()
        else:
            return self.interface_class(
                serial_number=self.serial_number,
                initial_attributes=self.camera_attributes,
            )

    def set_attributes_smart(self, attributes):
        """Call self.camera.set_attributes() to set the given attributes, only setting
        those that differ from their value in, or are absent from self.smart_cache.
        Update self.smart_cache with the newly-set values"""
        uncached_attributes = {}
        for name, value in attributes.items():
            self.logger.info(f'Smart setting attr: {name} to {value}')
            if name not in self.smart_cache or self.smart_cache[name] != value:
                uncached_attributes[name] = value
                self.smart_cache[name] = value
                
        self.camera.set_attributes(uncached_attributes)

    def get_attributes_as_dict(self, visibility_level):
        """Return a dict of the attributes of the camera for the given visibility
        level"""
        names = self.camera.get_attribute_names(visibility_level)
        attributes_dict = {name: self.camera.get_attribute(name) for name in names}
        return attributes_dict

    def get_attributes_as_text(self, visibility_level):
        """Return a string representation of the attributes of the camera for
        the given visibility level"""
        attrs = self.get_attributes_as_dict(visibility_level)
        # Format it nicely:
        lines = [f'    {repr(key)}: {repr(value)},' for key, value in attrs.items()]
        dict_repr = '\n'.join(['{'] + lines + ['}'])
        return self.device_name + '_camera_attributes = ' + dict_repr

    def snap(self):
        """Acquire one frame in manual mode. Send it to the parent via
        self.image_socket. Wait for a response from the parent."""
        n_frames = self.manual_mode_camera_attributes['SensNFrames']
        self.logger.info(f"Snapping {n_frames} frames together")
        image = self.camera.snap(
            # casting to int here, there may be a better location to do this
            integrated_frames=int(n_frames),
            settings=self.manual_mode_camera_attributes,
        )
        self._send_image_to_parent(image)
        
    def _send_image_to_parent(self, image):
        """Send the image to the GUI to display. This will block if the parent process
        is lagging behind in displaying frames, in order to avoid a backlog."""
        metadata = dict(dtype=str(image.dtype), shape=image.shape)
        self.image_socket.send_json(metadata, zmq.SNDMORE)
        self.image_socket.send(image, copy=False)
        response = self.image_socket.recv()
        assert response == b'ok', response

    def continuous_loop(self, dt):
        """Acquire continuously in a loop, with minimum repetition interval dt"""
        self.camera.configure_acquisition(continuous=True, bufferCount=1)
        while True:
            if dt is not None:
                t = perf_counter()
            try:
                image = self.camera.grab()  # Ensure grab() is configured for continuous mode
                self._send_image_to_parent(image)
            except Exception as e:
                self.logger.error(f"Error during continuous acquisition: {e}")
                break

            if dt is None:
                timeout = 0
            else:
                timeout = t + dt - perf_counter()
            if self.continuous_stop.wait(timeout):
                self.continuous_stop.clear()
                break

    def start_continuous(self, dt):
        """Begin continuous acquisition in a thread with minimum repetition interval dt.
        
        Applies manual mode camera attributes before starting acquisition.
        """
        assert self.continuous_thread is None, "Continuous acquisition is already running."
        # Apply manual mode attributes for responsive live view
        # I may force TrigFreeExtN = 1
        self.set_attributes_smart(self.manual_mode_camera_attributes)
        self.camera.configure_acquisition(continuous=True)  # Ensure continuous mode is enabled
        self.continuous_stop.clear()  # Reset the stop event
        self.continuous_thread = threading.Thread(
            target=self.continuous_loop, args=(dt,), daemon=True
        )
        self.continuous_thread.start()
        self.continuous_dt = dt

    def stop_continuous(self, pause=False):
        """Stop the continuous acquisition thread"""
        assert self.continuous_thread is not None, "Continuous acquisition is not running."
        self.continuous_stop.set()
        self.continuous_thread.join()
        self.continuous_thread = None
        self.camera.stop_acquisition()
        # If we're just 'pausing', then do not clear self.continuous_dt. That way
        # continuous acquisition can be resumed with the same interval by calling
        # start(self.continuous_dt), without having to get the interval from the parent
        # again, and the fact that self.continuous_dt is not None can be used to infer
        # that continuous acquisiton is paused and should be resumed after a buffered
        # run is complete:
        if not pause:
            self.continuous_dt = None

    def transition_to_buffered(self, device_name, h5_filepath, initial_values, fresh):
        
        print(f"{initial_values=}")
        if getattr(self, 'is_remote', False):
            h5_filepath = path_to_local(h5_filepath)
        if self.continuous_thread is not None:
            # Pause continuous acquistion during transition_to_buffered:
            self.stop_continuous(pause=True)
        with h5py.File(h5_filepath, 'r') as f:
            group = f['devices'][self.device_name]
            if 'EXPOSURES' not in group:
                return {}
            self.h5_filepath = h5_filepath
            self.exposures = group['EXPOSURES'][:]
            self.n_images = len(self.exposures)

            # Get the camera_attributes from the device_properties
            properties = labscript_utils.properties.get(
                f, self.device_name, 'device_properties'
            )
            camera_attributes = properties['camera_attributes']
            self.stop_acquisition_timeout = properties['stop_acquisition_timeout']
            self.exception_on_failed_shot = properties['exception_on_failed_shot']
            saved_attr_level = properties['saved_attribute_visibility_level']
            self.camera.exception_on_failed_shot = self.exception_on_failed_shot
        print(f'EXPOSURES: {self.exposures=}')
        # Only reprogram attributes that differ from those last programmed in, or all of
        # them if a fresh reprogramming was requested:
        if fresh:
            self.smart_cache = {}
        self.set_attributes_smart(camera_attributes)
        # Get the camera attributes, so that we can save them to the H5 file:
        if saved_attr_level is not None:
            self.attributes_to_save = self.get_attributes_as_dict(saved_attr_level)
        else:
            self.attributes_to_save = None
        print(f"Configuring camera for {self.n_images} images.")
        self.camera.configure_acquisition(continuous=False, bufferCount=self.n_images)
        self.images = []
        self.acquisition_thread = threading.Thread(
            target=self.camera.grab_multiple,
            args=(self.n_images, self.images),
            kwargs={'exposures': self.exposures},
            daemon=True,
        )
        self.acquisition_thread.start()
        return {}

    def transition_to_manual(self):

        self.logger.info('Got to transition_to_manual')
        if self.h5_filepath is None:
            print('No camera exposures in this shot.\n')
            return True
        assert self.acquisition_thread is not None
        self.acquisition_thread.join(timeout=self.stop_acquisition_timeout)
        if self.acquisition_thread.is_alive():
            msg = """Acquisition thread did not finish. Likely did not acquire expected
                number of images. Check triggering is connected/configured correctly"""
            if self.exception_on_failed_shot:
                self.abort()
                raise RuntimeError(dedent(msg))
            else:
                self.camera.abort_acquisition()
                self.acquisition_thread.join()
                print(dedent(msg), file=sys.stderr)
        self.acquisition_thread = None

        print("Stopping acquisition.")
        self.camera.stop_acquisition()

        print(f"Saving {len(self.images)}/{len(self.exposures)} images.")

        with h5py.File(self.h5_filepath, 'r+') as f:
            # Use orientation for image path, device_name if orientation unspecified
            if self.orientation is not None:
                image_path = 'images/' + self.orientation
            else:
                image_path = 'images/' + self.device_name
            image_group = f.require_group(image_path)
            image_group.attrs['camera'] = self.device_name

            # Save camera attributes to the HDF5 file:
            if self.attributes_to_save is not None:
                set_attributes(image_group, self.attributes_to_save)

            # Whether we failed to get all the expected exposures:
            image_group.attrs['failed_shot'] = len(self.images) != len(self.exposures)

            # key the images by name and frametype. Allow for the case of there being
            # multiple images with the same name and frametype. In this case we will
            # save an array of images in a single dataset.
            images = {
                (exposure['name'], exposure['frametype']): []
                for exposure in self.exposures
            }

            # Iterate over expected exposures, sorted by acquisition time, to match them
            # up with the acquired images:
            self.exposures.sort(order='t')
            for image, exposure in zip(self.images, self.exposures):
                images[(exposure['name'], exposure['frametype'])].append(image)

            # Save images to the HDF5 file:
            for (name, frametype), imagelist in images.items():
                data = imagelist[0] if len(imagelist) == 1 else np.array(imagelist)
                print(f"Saving frame(s) {name}/{frametype}.")
                group = image_group.require_group(name)
                dset = group.create_dataset(
                    frametype, data=data, dtype='uint16', compression='gzip'
                )
                # Specify this dataset should be viewed as an image
                dset.attrs['CLASS'] = np.bytes_('IMAGE')
                dset.attrs['IMAGE_VERSION'] = np.bytes_('1.2')
                dset.attrs['IMAGE_SUBCLASS'] = np.bytes_('IMAGE_GRAYSCALE')
                dset.attrs['IMAGE_WHITE_IS_ZERO'] = np.uint8(0)

        # If the images are all the same shape, send them to the GUI for display:
        try:
            image_block = np.stack(self.images)
        except ValueError:
            print("Cannot display images in the GUI, they are not all the same shape")
        else:
            self._send_image_to_parent(image_block)

        self.images = None
        self.n_images = None
        self.attributes_to_save = None
        self.exposures = None
        self.h5_filepath = None
        self.stop_acquisition_timeout = None
        self.exception_on_failed_shot = None
        self.logger.info("Setting manual mode camera attributes.\n")
        self.set_attributes_smart(self.manual_mode_camera_attributes)
        if self.continuous_dt is not None:
            # If continuous manual mode acquisition was in progress before the bufferd
            # run, resume it:
            self.start_continuous(self.continuous_dt)
        return True

    def abort(self):
        if self.acquisition_thread is not None:
            self.camera.abort_acquisition()
            self.acquisition_thread.join()
            self.acquisition_thread = None
            self.camera.stop_acquisition()
        self.camera._abort_acquisition = False
        self.images = None
        self.n_images = None
        self.attributes_to_save = None
        self.exposures = None
        self.acquisition_thread = None
        self.h5_filepath = None
        self.stop_acquisition_timeout = None
        self.exception_on_failed_shot = None
        # Resume continuous acquisition, if any:
        if self.continuous_dt is not None and self.continuous_thread is None:
            self.start_continuous(self.continuous_dt)
        return True

    def abort_buffered(self):
        return self.abort()

    def abort_transition_to_buffered(self):
        return self.abort()

    def program_manual(self, values):
        
        # It may be better to use set_attributes_smart() here
        # self.logger.info(f"Before: {self.manual_mode_camera_attributes}")
        self.camera.set_attributes(values)
        
        for k, v in values.items():
            self.manual_mode_camera_attributes[k] = v
            
        # self.logger.info(f"After: {self.manual_mode_camera_attributes}")
        
        print(self.camera.heSys.Acquire())
        return {}

    def shutdown(self):
        if self.continuous_thread is not None:
            self.stop_continuous()
        self.camera.close()
