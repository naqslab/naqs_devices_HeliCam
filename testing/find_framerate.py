import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import ctypes as ct
import numpy as np
from enum import Enum
from enum import IntEnum
import serial
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="find_fps.log",
    filemode="w",  # a for append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
prgPath = os.environ["PROGRAMFILES"]

sys.path.insert(0, prgPath + r"\Heliotis\heliCam\Python\wrapper")
from libHeLIC import LibHeLIC  # noqa: E402

MHZ = 1000000
PORT = "COM3"
# Import libHeLIC at module level so it's available for all classes
try:
    prgPath = os.environ["PROGRAMFILES"]

    sys.path.insert(0, prgPath + r"\Heliotis\heliCam\Python\wrapper")
    from libHeLIC import LibHeLIC  # noqa: E402
except ImportError as e:
    raise e("Could not find LibHeLIC, is HeliSDK properly installed?") # type: ignore


def get_demod_freq(SensTqp: int):
    freq_demod = (70 * MHZ / 8) * (1 / (SensTqp + 30))
    return freq_demod  # Hz


def get_tqp(demod_freq):
    tqp = 70 * MHZ / (8 * demod_freq) - 30
    return int(tqp)


class CamMode(IntEnum):
    RAW_IQ = 0
    AMPLITUDE = 1
    SMOOTH_AMPLITUDE = 2
    INTENSITY = 3  # Manual calls this "intensity", what they mean is HDR
    SIMPLE_MAX = 4
    # Have to find Addrs 5,6
    MIN_ENERGY = 7


class AcquisitionMode(Enum):
    CONTINUOUS = "continuous"
    SINGLE = "single"
    N_FRAMES = "n_frames"


# convenience send command for one-off sends to the PrawnDO
def send(command: str, echo=True, PORT=PORT) -> str:
    # pico expecting a newline termination for each command

    if command[-1] != "\n":
        command += "\n"

    resp = ""
    conn = None
    try:
        conn = serial.Serial(PORT, baudrate=152000, timeout=0.1)
        conn.write(command.encode())
        if echo:
            resp = conn.readlines()
            resp = "".join(s.decode() for s in resp)
    except Exception as e:
        logger.info(f"Encountered Error: {e}")

    finally:
        conn.close()

    return resp


class HeliCamInterface:
    def __init__(self):
        self.heSys = LibHeLIC()
        self.heSys.Open(0, sys="c3cam_sl70")
        self.heSys.Acquire()
        self.settings = None

    def refresh(self):
        self.close()
        self.__init__()

    def set_settings(self, settings):
        self.settings = settings
        if not isinstance(settings, dict):
            settings = dict(settings)

        for k, v in settings.items():
            try:
                print("Setting: ", k, v)
                setattr(self.heSys.map, k, v)
            except RuntimeError:
                logger.info(f"Could not set map property {k} to {v}")

    def get_settings(self):
        current_regs = self.heSys.MapByName(
            self.heSys.GetRegDesc(), self.heSys
        ).as_dict()
        return current_regs

    def view_setting(self, setting: str):
        return self.get_settings().get(setting.encode())

    def alloc_data(self, n_images):
        for n in range(n_images):
            alloc = self.heSys.AllocCamData(
                n, LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0
            )
            logger.info(f"Alloc {n} returned {alloc}")
        # alloc = self.heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_I16Q16'], 0, 0, 0)

    # def get_data(self, n_images):
    #     frames = self.view_setting("SensNFrames")

    #     for n in range(n_images):
    #         # Process the final image
    #         res = self.heSys.Acquire()
    #         logger.info(f"Acquire {n} returned {res}")
    #         cd = self.heSys.ProcessCamData(1, 0, 0)
    #         img = self.heSys.GetCamData(idx=1, addRef=0, meta=0)

    #         if res > 0:
    #             data = img.contents.data
    #             data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)

    #             I = data[:, :, :, 0]
    #             Q = data[:, :, :, 1]

    #             # amplitude = data[1:,:,:,:].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    #             amplitude = data.sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    #             # amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))

    #             fig, axs = plt.subplots(1, 3, sharey=True)
    #             fig.suptitle(f"Single Acquisition {n}, integrated over {frames} frames")
    #             axs[0].imshow(I.sum(axis=0), cmap="gray")
    #             axs[0].set_title("I")
    #             # axs[0].colorbar()
    #             axs[1].imshow(Q.sum(axis=0), cmap="gray")
    #             axs[1].set_title("Q")
    #             axs[2].imshow(amplitude, cmap="gray")
    #             axs[2].set_title("Amplitude")
    #             # fig.colorbar()
    #         else:
    #             logger.info("negative value in acquire")
    #             continue
    #         logger.info("Single acquisition complete")

    # def _get_image(self):

    #     # self.heSys = LibHeLIC()
    #     # self.heSys.Open(0, sys="c3cam_sl70")
    #     # self.heSys.Acquire()

    #     frames = 16

    #     SensTqp = get_tqp(demod_freq=2121)
    #     SensNavM2 = 1
    #     CalDur1Cyc = 1
    #     demod_freq = get_demod_freq(SensTqp=SensTqp)
    #     settings = (
    #         ## Required:
    #         ("CamMode", CamMode.RAW_IQ),
    #         ("SensNFrames", frames),
    #         ("BSEnable", 0),
    #         # required for external acquisition trigger
    #         ("TrigFreeExtN", 1),  # 0: external triggering, 1: internal
    #         ("ExtTqp", 0),
    #         ("EnTrigOnPos", 0),
    #         ("SingleVolume", 0),
    #         # ('TrigFreeExtN', 1),
    #         ## Optional:
    #         # ('SingleVolume', 1),
    #         ("SensTqp", SensTqp),  # (1, 4095)
    #         ("SensNavM2", SensNavM2),  # (1, 255)
    #         (
    #             "CalDur1Cyc",
    #             CalDur1Cyc,
    #         ),  # 1: offset compensation T_offset = 1 cycle, 0 uses SensCaldur
    #         ("DdsGain", 2),  # (1, 3), default was 2, manual says best SNR is 2
    #         (
    #             "TrigExtSrcSel",
    #             0,
    #         ),  # selects triggering source but there's only one on the Helidriver
    #         ("AcqStop", 0),  # 0: Acquisition running, 1: Acquisition stopped
    #         ("EnSynFOut", 0),  # Provides the sync freq on OUT3 (and encoder??)
    #     )

    #     for k, v in settings:
    #         try:
    #             logger.info("Setting: ", k, v)
    #             setattr(self.heSys.map, k, v)
    #         except RuntimeError:
    #             logger.info(f"Could not set map property {k} to {v}")

    #     logger.info(f"{demod_freq =:.4f}Hz")

    #     alloc = self.heSys.AllocCamData(1, LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0)
    #     logger.info(f"Alloc: {alloc}")

    #     f = plt.figure(figsize=(8, 8))

    #     if mode == AcquisitionMode.SINGLE:
    #         cd = self.acquire_single(frames, f)
    #         plt.show()

    #     elif mode == AcquisitionMode.N_FRAMES:
    #         cd = self.acquire_n_frames(frames, f, n_acquisitions, early_stop)
    #         plt.show()

    #     else:
    #         cd = self.acquire_continuous(frames, f)

    #     return cd

    # def acquire_single(self, frames, fig):
    #     """Acquire and display a single image."""
    #     ignore = 0

    #     # Acquire images until we get past the ignore phase
    #     for i in range(ignore + 1):
    #         res = self.heSys.Acquire()
    #         logger.info(f"Acquire {i} returned {res}")

    #         if i < ignore:
    #             logger.info("Ignoring image")
    #             continue

    #         else:
    #             # Process the final image
    #             cd = self.heSys.ProcessCamData(1, 0, 0)
    #             # img = self.heSys.GetCamData(idx=1, addRef=1, meta=0)
    #             img = self.heSys.GetCamData(idx=1, addRef=0, meta=0)
    #             data = img.contents.data
    #             data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)

    #             I = data[:, :, :, 0]
    #             Q = data[:, :, :, 1]

    #             # amplitude = data[1:,:,:,:].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    #             amplitude = data.sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    #             amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))

    #             fig, axs = plt.subplots(1, 3, sharey=True)
    #             fig.suptitle(f"Single Acquisition, integrated over {frames} frames")
    #             axs[0].imshow(I.sum(axis=0), cmap="gray")
    #             axs[0].set_title("I")
    #             # axs[0].colorbar()
    #             axs[1].imshow(Q.sum(axis=0), cmap="gray")
    #             axs[1].set_title("Q")
    #             axs[2].imshow(amplitude_disp, cmap="gray")
    #             axs[2].set_title("Amplitude")
    #             # axs[1].colorbar()
    #             logger.info("Single acquisition complete")
    #     return cd

    # def acquire_n_frames(self, frames, fig, n_acquisitions, early_stop):
    #     """Acquire N images and stop."""

    #     ignore = 0
    #     for i in range(ignore + n_acquisitions):
    #         res = self.heSys.Acquire()
    #         logger.info(f"Acquire {i}: {res}")
    #         alloc = self.heSys.AllocCamData(
    #             i, LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0
    #         )
    #         logger.info(f"Alloc {i}: {alloc}")

    #         if i < ignore:
    #             logger.info("Ignoring image")
    #             continue
    #         else:
    #             if i > early_stop:
    #                 continue
    #             # Process
    #             cd = self.heSys.ProcessCamData(i, 0, 0)
    #             img = self.heSys.GetCamData(i, 0, 0)
    #             data = img.contents.data
    #             data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)
    #             I = data[:, :, :, 0]
    #             Q = data[:, :, :, 1]
    #             amplitude = (
    #                 data[1:, :, :, :]
    #                 .sum(axis=0, dtype=np.int16)
    #                 .sum(axis=2, dtype=np.int16)
    #             )
    #             # amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))

    #             fig, axs = plt.subplots(1, 3, sharey=True)
    #             fig.suptitle(f"Acquisition {i}, integrated over {frames} frames")
    #             axs[0].imshow(np.rot90(I.sum(axis=0)), cmap="gray")
    #             axs[0].set_title("I")
    #             axs[1].imshow(np.rot90(Q.sum(axis=0)), cmap="gray")
    #             axs[1].set_title("Q")
    #             axs[2].imshow(np.rot90(amplitude), cmap="gray")
    #             axs[2].set_title("Amplitude")
    #             logger.info(f"Acquisition {i} complete")
    #             # setattr(self.heSys.map, "AcqStop", 1)

    # def acquire_continuous(self, frames, fig):
    #     """Continuous streaming acquisition."""

    #     def on_timer():
    #         """This gets called every update of the plot window"""
    #         if on_timer.cnt < 0:
    #             logger.info("Stopping continuous acquisition")
    #             return

    #         res = self.heSys.Acquire()
    #         logger.info(f"Acquire {on_timer.cnt} returned {res}")
    #         cd = self.heSys.ProcessCamData(1, 0, 0)

    #         img = self.heSys.GetCamData(1, 0, 0)
    #         data = img.contents.data
    #         data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)
    #         amplitude = (
    #             data[1:, :, :, :]
    #             .sum(axis=0, dtype=np.int16)
    #             .sum(axis=2, dtype=np.int16)
    #         )

    #         # Rotate counterclockwise once after processing
    #         amplitude = np.rot90(amplitude)

    #         ignore = 1
    #         if on_timer.cnt < ignore:
    #             logger.info("Ignoring image")
    #         elif on_timer.cnt == ignore:
    #             logger.info("make fixpattern image")
    #             on_timer.f.suptitle("Continuous Streaming")
    #             on_timer.fixPtrn = amplitude.copy()

    #             amplitude_disp = np.uint8(
    #                 np.clip(amplitude - on_timer.fixPtrn + 128, 0, 255)
    #             )

    #             plt.subplot(1, 1, 1)
    #             on_timer.imHdl1 = plt.imshow(
    #                 amplitude_disp, vmin=0, vmax=255, cmap="gray"
    #             )
    #             plt.colorbar()
    #         else:
    #             amplitude_diff = amplitude - on_timer.fixPtrn

    #             amplitude_disp = np.uint8(np.clip(amplitude_diff + 128, 0, 255))

    #             on_timer.imHdl1.set_array(amplitude_disp)

    #         on_timer.f.canvas.draw_idle()
    #         on_timer.cnt += 1

    #     def on_close(event):
    #         """Sends a -1 signal to close the plot window"""
    #         on_timer.cnt = -1

    #     on_timer.f = fig
    #     on_timer.cnt = 0
    #     fig.canvas.mpl_connect("close_event", on_close)

    #     # timer = fig.canvas.new_timer(interval=1)
    #     timer = fig.canvas.new_timer(interval=200)
    #     timer.add_callback(on_timer)
    #     timer.start()
    #     plt.show()

    def test_connected_cameras(self):
        logger.info("-" * 20)
        self.heSys = LibHeLIC()
        [count, serials] = LibHeLIC.GetSerials()
        logger.info("No of installed heliCams: " + str(count))
        for item in serials:
            logger.info(item)

    def testz(self):
        logger.info("-" * 20)
        logger.info("Used Python version:")
        logger.info(sys.version)
        logger.info("-" * 20)
        logger.info("Search libHeLIC in:")
        for ImportPath in sys.path:
            logger.info(ImportPath)
        logger.info("-" * 20)
        logger.info("Numpy version:")
        logger.info(np.version.version)
        logger.info("Matplotlib version:")
        logger.info(mpl.__version__)
        logger.info("libHeLIC version:")
        self.heSys = LibHeLIC()
        logger.info(
            str(LibHeLIC.GetVersion()[1][3])
            + "."
            + str(LibHeLIC.GetVersion()[1][2])
            + "."
            + str(LibHeLIC.GetVersion()[1][1])
            + "."
            + str(LibHeLIC.GetVersion()[1][0])
        )

    def close(self):
        try:
            self.heSys.Close()
        except Exception as e:
            logger.info(f"Warning: Error closing camera: {e}", file=sys.stderr)


def image_from_buffer(cam):
    # aq = cam.heSys.Acquire()
    # logger.info(f'image_from_buffer acquire: {aq}')
    frames = int(cam.view_setting("SensNFrames"))

    cd = cam.heSys.ProcessCamData(1, 0, 0)
    # img = cam.heSys.GetCamData(idx=1, addRef=1, meta=0)
    img = cam.heSys.GetCamData(idx=1, addRef=0, meta=0)
    data = img.contents.data
    data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)
    timestamp = time.perf_counter_ns()  # before doing np processing

    I = data[:, :, :, 0]
    Q = data[:, :, :, 1]

    amplitude = (
        data[1:, :, :, :].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    )
    amplitude = data.sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    # amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))
    # return I, Q

    return amplitude, timestamp


def transfer_data_no_processing(cam):
    """Probably replaceable function that just facilitates data transfer for profiling"""
    # aq = cam.heSys.Acquire()
    # logger.info(f'image_from_buffer acquire: {aq}')
    frames = int(cam.view_setting("SensNFrames"))

    cd = cam.heSys.ProcessCamData(1, 0, 0)
    # img = cam.heSys.GetCamData(idx=1, addRef=1, meta=0) If addRef==1 a reference is added to that buffer and it is not freed if the systems releases it.
    img = cam.heSys.GetCamData(idx=1, addRef=0, meta=0)
    data = img.contents.data
    data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)

    I = data[:, :, :, 0]
    Q = data[:, :, :, 1]

    # amplitude = data[1:,:,:,:].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16) # examples drop the very first frame
    # amplitude = data.sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
    # amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))
    # return amplitude


def view_images(images, n=10):
    """Helper to visually check images for junk data"""
    nrows = 2 if n <= 10 else n // 4

    ncols = int(np.ceil(n / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes_flat = axes.flatten()

    for i in range(len(axes_flat)):
        if i < len(images) and i < n:
            axes_flat[i].imshow(images[i], cmap="Grays")
            axes_flat[i].set_title(f"Image {i + 1}")
        else:
            axes_flat[i].axis("off")
            continue

        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.show()


def program_prawndo(duration, reps):
    """Puts the pulse instructions into the PrawnDO's memory"""

    # For HeliCam, trigger_width is > 10us. 1100 cycles at 100MHz = 11us
    trigger_width = 1100

    bits = [1, 0] * reps + [1, 1]
    cycles = [duration, trigger_width] * reps + [0, 0]

    send("add\n")
    for bit, cycle in zip(bits, cycles):
        send(f"{bit:x} {cycle:x}\n")
    send("end\n")
    logger.info(f"Programmed for {duration / (100 * MHZ):e} s b/t acquisitions")


def start_prawn():
    """
    Merely sends the software start command.
    Currently with a fudged sleep time to ensure we don't start before the
    camera polling loop is ready, will change
    """
    time.sleep(5)  # warmup time to ensure polling is in fact on - can this be checked?
    logger.info(send("swr"))
     
def automated_checking_binary_search(
    SensNFrames: int,
    SensNavM2: int,
    SensTqp: int,
    times_to_test=None,
    tolerance=0.001,
    max_iters=15,
):
    """
    Given the parameter triple of (`SensNFrames`,`SensNavM2`, `SensTqp`),
    tests for the fastest achievable duration between pictures via a binary search.

    Args:
        SensNFrames (int): Number of Frames
        SensNavM2 (int): Register for (number of demod cycles - 2 )/ 2
        SensTqp (int): Time quarter period
        times_to_test (_type_, optional): Option to pass in time grid. 
        Defaults to None so that later initializes to powers of ten in arange(5, 9).
        tolerance (float, optional): Binary search tolerance in seconds.
        max_iters (int, optional): Binary search max iterations.

    Raises:
        TimeoutError: Uses the TimeoutError in binary search to check if we need to slow down.
        Will otherwise speed up.
    """

    cam = HeliCamInterface()
    settings = (
        ## Required:
        ("CamMode", CamMode.RAW_IQ),
        ("SensNFrames", SensNFrames),
        ("BSEnable", 0),
        # required for external acquisition trigger
        ("TrigFreeExtN", 0),  # 0: external triggering, 1: internal
        # ('TrigFreeExtN', 1),
        ("ExtTqp", 0),
        ("EnTrigOnPos", 0),
        ("SingleVolume", 0),
        ## Optional:
        # ('SingleVolume', 1),
        ("SensTqp", SensTqp),  # (1, 4095)
        ("SensNavM2", SensNavM2),  # (1, 255)
        ("DdsGain", 2),  # (1, 3), default was 2, manual says best SNR is 2
        ("TrigExtSrcSel", 0),  # selects triggering source (only one)
        ("AcqStop", 0),  # 0: Acquisition running, 1: Acquisition stopped
        ("EnSynFOut", 1),  # Provides the sync freq on OUT3 (and encoder??)
    )
    settings = dict(settings)

    cam.set_settings(settings)
    n_frames = int(cam.view_setting("SensNFrames"))
    n_demod_cycles = int(cam.view_setting("SensNavM2")) * 2 + 2
    tqp = int(cam.view_setting("SensTqp"))

    cam.heSys.AllocCamData(1, LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0)

    n_acquisitions = 3
    
    # if you pass in `times_to_test` then we'll defer to that but otherwise
    # its set here
    if times_to_test is None:
        # descending, but should be at worst case 10 seconds
        # define a coarse grid over powers of ten
        powers_of_ten = 10 ** np.arange(5, 10)
        times_to_test = powers_of_ten
        # times_to_test = np.array([2 * 10**8, 10**8, 5 * 10**7])
    # failed_arr = np.zeros_like(times_to_test, dtype=bool)
    
    # initialize a binary search / bisect method over the times
    # outer loop searches over durations
    left = times_to_test[0]
    right = times_to_test[-1]
    fastest_time = None
    # step = mid - left
    
    ind = 0
    # don't forget we're actually in clock cycles, but the tolerance is in s
    while (right - left) / 100e6 > tolerance and ind < max_iters: 
        
        mid = (right + left) // 2
        duration = mid
        
        ## <<<<< make this a function??
        images = np.empty((n_acquisitions, 300, 300))
        send("cls")  # clear DO memory
        # program_prawndo(duration=duration, reps=n_acquisitions)
        ready = False
        ready = check_ready()
        while not ready:
            logger.info("WARNING: Something went wrong trying to program PrawnDO")
            logger.info(send("abt"))
            logger.info(send("cls"))
            time.sleep(3)
            ready = check_ready()
            logger.info(f'{ready=}')
        program_prawndo(duration=duration, reps=n_acquisitions)

        # call start from a different thread than the while loop
        t = threading.Thread(target=start_prawn, daemon=True)
        t.start()

        try:
            acquisition_times = []
            res = -1
            # fudge how long the acquisition should never be longer than?
            # This includes the time that the DO waits for the polling loop to start
            # inner loop over each acquisition
            for i in range(n_acquisitions):
                deadline = (
                    5 + time.perf_counter() + duration / (100 * MHZ) * 3
                )  # 3x margin?? TODO: Make this a DO check
                while res < 1:
                    if time.perf_counter() > deadline:
                        raise TimeoutError(
                            f"Acquire() timeout with duration {duration}"
                        )
                    logger.info(f"[Camera polling] waiting, got {res}")
                    t_before = time.perf_counter_ns()
                    res = cam.heSys.Acquire()
                logger.info(f"[Camera polling] Triggered, got {res}")
                images[i], t_after = image_from_buffer(cam)
                acquisition_times.append((t_after - t_before))  # should be T_acquisition
                res = -1

            # t.join()
            logger.info(f"Measured acquisition {acquisition_times=}")
            ## >>>>> end function???
            # go faster
            logger.info(f'[Binary Search] iter: {ind} -- dur: {duration/100e6} s succeeded -- Going faster')
            right = mid
            # mid = abs(right - left) // 2
            # step = abs(mid - left)
            # ind += 1
            fastest_time = duration/100e6
            
        except Exception as e:
            # go slower
            logger.info(f'[Binary Search] iter: {ind} -- dur: {duration/100e6} s failed -- Going slower')
            left = mid 
            # mid = abs(right - left) // 2
            # step = abs(mid - right)
            # ind += 1
            # continue
        finally:
            t.join()
        ind += 1
    
    # logger.info(f'{len(images) == n_acquisitions=}')
    supposed_frm_dur = n_demod_cycles / get_demod_freq(tqp)
    supposed_acq_dur = n_frames * n_demod_cycles / get_demod_freq(tqp)
    logger.info("=" * 79)
    logger.info("============= RUN FINISHED =============")
    logger.info("=" * 79)
    # logger.info(f'[Binary Search] fastest achieved: {fastest_time:e}s in {ind} iterations')
    logger.info(f'[Binary Search] fastest achieved: {fastest_time:.3e}s in {ind} iterations' if fastest_time else 'No successful acquisition found')
    logger.info(f"{supposed_frm_dur*10**6=:.4f} microseconds")
    logger.info(f"{supposed_acq_dur*10**6=:.4f} microseconds")
    logger.info(f"supposed_fps={1 / supposed_frm_dur} fps")
    logger.info(f"{n_frames=}")
    logger.info(f"{n_demod_cycles=}")
    logger.info(f"demod freq = {get_demod_freq(tqp)}")
    logger.info(
        f"{cam.heSys.Acquire()=}, check: frames*300*300*2*2={n_frames * 300 * 300 * 2 * 2} bytes"
    )

    # finally ? close.
    logger.info(f"{cam.heSys.Close()=}")
    # view_images(images,n=n_acquisitions)

    if fastest_time is None:
        fastest_time = 'None' # coerce to str for safe saving to npz
    # save after loop to aggregate
    if SAVE:
        np.savez(
            file=os.path.join("results", "binary_search", f"f{n_frames}_d{n_demod_cycles}_t{tqp}.npz"),
            images=images,
            fastest_time=fastest_time,
            iters=ind,
            tolerance=tolerance,
            acquisition_times=acquisition_times,
            duration=duration,
            times_to_test=times_to_test,
            settings=np.array((n_frames, n_demod_cycles, tqp)),
        )


def automated_checking(SensNFrames: int, SensNavM2: int, SensTqp: int, times_to_test=None):

    cam = HeliCamInterface()
    settings = (
        ## Required:
        ("CamMode", CamMode.RAW_IQ),
        ("SensNFrames", SensNFrames),
        ("BSEnable", 0),
        # required for external acquisition trigger
        ("TrigFreeExtN", 0),  # 0: external triggering, 1: internal
        # ('TrigFreeExtN', 1),
        ("ExtTqp", 0),
        ("EnTrigOnPos", 0),
        ("SingleVolume", 0),
        ## Optional:
        # ('SingleVolume', 1),
        ("SensTqp", SensTqp),  # (1, 4095)
        ("SensNavM2", SensNavM2),  # (1, 255)
        ("DdsGain", 2),  # (1, 3), default was 2, manual says best SNR is 2
        ("TrigExtSrcSel", 0),  # selects triggering source (only one)
        ("AcqStop", 0),  # 0: Acquisition running, 1: Acquisition stopped
        ("EnSynFOut", 1),  # Provides the sync freq on OUT3 (and encoder??)
    )
    settings = dict(settings)

    cam.set_settings(settings)
    n_frames = int(cam.view_setting("SensNFrames"))
    n_demod_cycles = int(cam.view_setting("SensNavM2")) * 2 + 2
    tqp = int(cam.view_setting("SensTqp"))

    cam.heSys.AllocCamData(1, LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0)

    n_acquisitions = 3
    
    # if you pass in `times_to_test` then we'll defer to that but otherwise
    # its set here
    if not times_to_test:
        # descending, but should be at worst case 10 seconds
        # define a coarse grid over powers of ten
        powers_of_ten = 10 ** np.arange(5, 10)[::-1]
        times_to_test = powers_of_ten
    failed_arr = np.zeros_like(times_to_test, dtype=bool)
    
    # outer loop searches over durations
    for ind, duration in enumerate(times_to_test):
        images = np.empty((n_acquisitions, 300, 300))
        send("cls")  # clear DO memory
        program_prawndo(duration=duration, reps=n_acquisitions)
        ready = False
        ready = check_ready()
        if not ready:
            logger.info("WARNING: Something went wrong trying to program PrawnDO")
            logger.info(send("abt"))
            logger.info(send("cls"))
            program_prawndo(duration=duration, reps=n_acquisitions)

            ready = check_ready()

        # call start from a different thread than the while loop
        t = threading.Thread(target=start_prawn, daemon=True)
        t.start()

        try:
            times = []
            res = -1
            # fudge how long the acquisition should never be longer than?
            # inner loop over each acquisition
            for i in range(n_acquisitions):
                deadline = (
                    5 + time.perf_counter() + duration / (100 * MHZ) * 6
                )  # 6x margin?? TODO: Make this a DO check
                while res < 1:
                    if time.perf_counter() > deadline:
                        # failed_arr[ind] = True
                        raise TimeoutError(
                            f"Acquire() timeout with duration {duration}"
                        )
                    logger.info(f"waiting, got {res}")
                    t_before = time.perf_counter_ns()
                    res = cam.heSys.Acquire()
                logger.info(f"Triggered, got {res}")
                images[i], t_after = image_from_buffer(cam)
                # t_after = time.perf_counter_ns()
                times.append((t_after - t_before))  # should be T_acquisition
                res = -1

            t.join()
            logger.info(f"Measured readout {times=}")
        except Exception as e:
            logger.info(
                f"{e} on combo {(n_frames, n_demod_cycles, tqp)} with duration {duration / 100e6} s"
            )
            # Skip the rest, no reason to test more if we timed out
            failed_arr[ind:] = True
            break

    # logger.info(f'{len(images) == n_acquisitions=}')
    supposed_frm_dur = n_demod_cycles / get_demod_freq(tqp)
    supposed_acq_dur = n_frames * n_demod_cycles / get_demod_freq(tqp)
    logger.info("=" * 79)
    logger.info("============= RUN FINISHED =============")
    logger.info("=" * 79)
    logger.info(f"{supposed_frm_dur*10**6=:.4f} microseconds")
    logger.info(f"{supposed_acq_dur*10**6=:.4f} microseconds")
    logger.info(f"supposed_fps={1 / supposed_frm_dur} fps")
    logger.info(f"{n_frames=}")
    logger.info(f"{n_demod_cycles=}")
    logger.info(f"demod freq = {get_demod_freq(tqp)}")
    logger.info(
        f"{cam.heSys.Acquire()=}, check: frames*300*300*2*2={n_frames * 300 * 300 * 2 * 2} bytes"
    )

    # finally ? close.
    logger.info(f"{cam.heSys.Close()=}")
    # view_images(images,n=n_acquisitions)

    # save after loop to aggregate
    if SAVE:
        np.savez(
            file=os.path.join("results", f"f{n_frames}_d{n_demod_cycles}_t{tqp}.npz"),
            images=images,
            times=times,
            duration=duration,
            failed_arr=failed_arr,
            times_to_test=times_to_test,
            settings=np.array((n_frames, n_demod_cycles, tqp)),
        )


def check_ready():
    '''Check PrawnDO status with `sts` cmd

    Returns:
        bool
    '''

    resp = send("sts")
    match = re.match(r"run-status:(\d) clock-status:(\d)(\r\n)?", resp)
    status = int(match.group(1)), int(match.group(2))
    ready = status[0] == 0 or 5
    if ready:
        logger.info("PrawnDO Ready!")
    else:
        logger.info("PrawnDO not ready")

    return ready


if __name__ == "__main__":
    """
    TODO:
    [x] Measure USB comm speed:
        - set 100 frames, get acquisition time
        - Got ~ the speed of the usb readout for acq time 
    [] Debug 10s fails, but 1s success
    [x] Make Gradient descent loop
    """

    import threading
    from itertools import product
    import re
    SAVE = True

    frame_vals = [1, 100, 500]
    NavM2_vals = [1, 50, 200]
    tqp_vals = [4095, 2000, 1]
    combos = list(product(frame_vals, NavM2_vals, tqp_vals))
    # for idx, (f, d, t) in enumerate(combos):
    # resp = send('sts')
    # match = re.match(r"run-status:(\d) clock-status:(\d)(\r\n)?", resp)
    # status = int(match.group(1)), int(match.group(2))
    # ready = status[0] == 0
    # if ready:
    #     logger.info('PrawnDO Ready!')
    #     logger.info(f'Testing combo {idx+1}/{len(combos)}: {(f, d, t)}')
    #     # automated_checking(f, d, t)
    # else:
    #     logger.info('PrawnDO not ready')

    # resp = send("sts")
    # match = re.match(r"run-status:(\d) clock-status:(\d)(\r\n)?", resp)
    # status = int(match.group(1)), int(match.group(2))

    # for f_vals in [1, 100, 500]:
    for idx, (f, d, t) in enumerate(combos):
        ready = check_ready()
        if ready:
            logger.info("PrawnDO Ready!")
            logger.info(f"Testing combo {idx + 1}/{len(combos)}: {(f, d, t)}")
            automated_checking_binary_search(SensNFrames=f, SensNavM2=d, SensTqp=t)
        else:
            print("PrawnDO not ready")
