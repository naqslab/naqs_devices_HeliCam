import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

import ctypes as ct
import numpy as np
from enum import Enum
from enum import IntEnum
prgPath = os.environ["PROGRAMFILES"]
from KeysightScope import KeysightScope

sys.path.insert(0, prgPath + r"\Heliotis\heliCam\Python\wrapper")
from libHeLIC import LibHeLIC  # noqa: E402
MHZ = 1000000

def get_demod_freq(SensTqp: int):
    """
    Helper to calculate demodulation frequency from SensTqp param
    
    :param SensTqp: Sensor time quarter period, must be in [1, 4095]

    """
    freq_demod = (70 * MHZ / 8) * (1 / (SensTqp + 30))
    
    return freq_demod # Hz

def get_tqp(demod_freq):
    
    tqp = 70 * MHZ / (8 * demod_freq) - 30
    
    return int(tqp)

class CamMode(IntEnum):
    RAW_IQ = 0
    AMPLITUDE = 1
    SMOOTH_AMPLITUDE = 2
    INTENSITY = 3 # Manual calls this "intensity", what they mean is HDR
    SIMPLE_MAX = 4
    # Have to find Addrs 5,6 
    MIN_ENERGY = 7
    
class AcquisitionMode(Enum):
    CONTINUOUS = "continuous"
    SINGLE = "single"
    N_FRAMES = "n_frames"

def test1():
    """
    Test registers and mappings
    """
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    print("GetVersion SrCam", LibHeLIC.GetVersion())
    
    rd = heSys.GetRegDesc()
    sd = heSys.GetSysDesc()

    print("\n" + 20 * "-", "Registers:", rd.contents.numReg, 20 * "-")
    for idx in range(rd.contents.numReg):
        r = rd.contents.regs[idx]
        print(r.num, r.id, " = ",  r.cmt)
    print("\n" + 20 * "-", "Mappings:", rd.contents.numMap, 20 * "-")
    for idx in range(rd.contents.numMap):
        m = rd.contents.maps[idx]
        print(
            "\n",
            m.id
            , "\t"
            , m.group
            , " - lvl:cammode "
            , str(m.level)
            , ":"
            , str(m.level)
            , "\tdef:min:max "
            , str(m.defValue)
            , ":"
            , str(m.minValue)
            , ":"
            , str(m.maxValue)
            , "\n\t\t"
            , m.cmt
            , "\n"
            , "============================ ="
        )

    
    reg_as_dict = heSys.MapByName(rd, heSys).as_dict()
    print(reg_as_dict)
    heSys.Close()


def test2():
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    """
    Surface Acquisitioni A16Z16
    """

    frames=150
    settings = (
    ('SensTqp',540),
    ('SensNavM2',2),
    ('SensNFrames',frames),
    ('BSEnable',1),
    ('DdsGain',2),
    ('TrigFreeExtN',1),
    ('TrigExtSrcSel',0),
    ('CamMode', CamMode.SIMPLE_MAX),
    ('AcqStop',0)
    );

    for k,v in settings:
        try:
            setattr(heSys.map,k,v)#heSys.map.k=v
        except RuntimeError as err:
            print(f'Could not set map property {k} to {v}')
            
    heSys.AllocCamData(
        idx=1,
        format=LibHeLIC.CamDataFmt['DF_A16Z16'],
        prop=0,
        extData=0,
        extDataSz=0
        )

    def on_idle():
        if on_idle.cnt < 0:
            print("on_idle.stop")
            return

        res = on_idle.heSys.Acquire()
        print("Acquire", on_idle.cnt, "returned", res)

        img = heSys.GetCamData(0, 0, 0)
        data = img.contents.data + 1024 * 2
        data = LibHeLIC.Ptr2Arr(data, (292, 280, 2), ct.c_ushort)

        if on_idle.cnt == 0:
            on_idle.f.suptitle('GetCamData')

            plt.subplot(1, 2, 1)
            on_idle.imHdl1 = plt.imshow(data[:, :, 1], interpolation='nearest')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            on_idle.imHdl2 = plt.imshow(data[:, :, 0], interpolation='nearest')
            plt.colorbar()
        else:
            on_idle.imHdl1.set_array(data[:, :, 1])
            on_idle.imHdl2.set_array(data[:, :, 0])

        on_idle.f.canvas.draw_idle()
        on_idle.cnt += 1

    def on_close(event):
        on_idle.cnt=-1
    f = plt.figure()
    on_idle.f = f
    on_idle.heSys = heSys
    on_idle.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=50)
    timer.add_callback(on_idle)
    timer.start()

    plt.show()
    heSys.Close()

def test3():
    """
    Volume Acquisition A16
    """
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    frames = 150
    settings = (
        ('SensTqp', 540),
        ('SensNavM2', 2),
        ('SensNFrames', frames),
        ('BSEnable', 1),
        ('DdsGain', 2),
        ('TrigFreeExtN', 1),
        ('TrigExtSrcSel', 0),
        ('CamMode', CamMode.AMPLITUDE),  # amplitude
        ('Comp11to8', 0),
        ('AcqStop', 0),
    )

    for k, v in settings:
        try:
            print("Setting: ", k, v)
            setattr(heSys.map, k, v)
        except RuntimeError:
            print(f'Could not set map property {k} to {v}')

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_A16'], 0, 0, 0)

    def on_timer():
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        cd = heSys.ProcessCamData(1, 0, 0)
        print("ProcessCamData", on_timer.cnt, "returned", cd.contents.data)

        img = heSys.GetCamData(1, 0, 0)
        data = img.contents.data
        data = LibHeLIC.Ptr2Arr(data, (frames, 292, 282), ct.c_ushort)

        if on_timer.cnt == 0:
            on_timer.f.suptitle('GetCamData')

            plt.subplot(1, 2, 1)
            # on_timer.imHdl1 = plt.imshow(data[:, :, 141], interpolation='nearest')
            on_timer.imHdl1 = plt.imshow(data[141, :, :], interpolation='nearest')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            on_timer.imHdl2 = plt.imshow(data[40, :, :], interpolation='nearest')
            plt.colorbar()
        else:
            on_timer.imHdl1.set_array(data[:, :, 141])
            on_timer.imHdl2.set_array(data[40, :, :])

        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=50)  # ms
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    
    heSys.Close()
    
def test4():
    heSys = LibHeLIC()
    heSys.Open(0, sys='c3cam_sl70')

    frames = 150
    settings = (
        ('SensTqp', 699),
        ('SensNavM2', 2),
        ('SensNFrames', frames),
        ('BSEnable', 1),
        ('DdsGain', 2),
        ('TrigFreeExtN', 1),
        ('TrigExtSrcSel', 0),
        ('CamMode', CamMode.AMPLITUDE),  # amplitude
        ('Comp11to8', 1),
        ('AcqStop', 0)
    )

    for k, v in settings:
        try:
            setattr(heSys.map, k, v)
        except RuntimeError:
            error('Could not set map property %s to %s', k, v)

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_A8'], 0, 0, 0)

    def on_timer():
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        cd = heSys.ProcessCamData(1, 0, 0)
        print("ProcessCamData", on_timer.cnt, "returned", cd.contents.data)

        img = heSys.GetCamData(1, 0, 0)
        data = img.contents.data
        data = LibHeLIC.Ptr2Arr(data, (frames, 292, 282), ct.c_uint8)

        if on_timer.cnt == 0:
            on_timer.f.suptitle('GetCamData')

            plt.subplot(1, 2, 1)
            on_timer.imHdl1 = plt.imshow(data[:, :, 141], interpolation='nearest')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            on_timer.imHdl2 = plt.imshow(data[40, :, :], interpolation='nearest')
            plt.colorbar()
        else:
            on_timer.imHdl1.set_array(data[:, :, 141])
            on_timer.imHdl2.set_array(data[40, :, :])

        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=50)  # ms
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    heSys.Close()
    
    
def test5():
    heSys = LibHeLIC()
    heSys.Open(0, sys='c3cam_sl70')

    frames = 200
    settings = (
        ('SensTqp', 540),
        ('SensNavM2', 2),
        ('SensNFrames', frames),
        ('BSEnable', 1),
        ('DdsGain', 2),
        ('TrigFreeExtN', 1),
        ('TrigExtSrcSel', 0),
        ('CamMode', CamMode.RAW_IQ),  # raw IQ
        ('AcqStop', 0),
    )

    for k, v in settings:
        try:
            print("Setting: ", k, v)
            setattr(heSys.map, k, v)
        except RuntimeError:
            print(f'Could not set map property {k} to {v}')

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_I16Q16'], 0, 0, 0)

    def on_timer():
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        cd = heSys.ProcessCamData(1, 0, 0)
        print("ProcessCamData", on_timer.cnt, "returned", cd.contents.data)

        meta = heSys.CamDataMeta()
        img = heSys.GetCamData(1, 0, ct.byref(meta))
        data = img.contents.data

        data = LibHeLIC.Ptr2Arr(
            data, (frames, 300, 300, 2), ct.c_ushort
        )
        print(tuple(meta.dimSz[meta.numDim-1::-1]))
        print(dir(img.contents))
        print("size = " + str(img.contents.size))
        print("format = " + str(img.contents.format))
        print("prop = " + str(img.contents.prop))
        print("data.shape = " + str(data.shape))
        # data = data[:,:,:,0]
        if on_timer.cnt == 0:
            on_timer.f.suptitle('GetCamData')

            plt.subplot(1, 2, 1)
            on_timer.imHdl1 = plt.imshow(
                data[:, 45, :, 0], interpolation='nearest'
            )
            plt.colorbar()

            plt.subplot(1, 2, 2)
            on_timer.imHdl2 = plt.imshow(
                data[25, :, :, 1], interpolation='nearest'
            )
            plt.colorbar()
        else:
            on_timer.imHdl1.set_array(data[:, 45, :, 0])
            on_timer.imHdl2.set_array(data[25, :, :, 1])

        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=50)  # ms
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    heSys.Close()

def test6():
    """
    Intensity Image - Low noise / clean image version
    This is an odd test, using DF_I16Q16 and sums the resulting I and Q data
    to get intensity instead of using DF_Hf, which intensity is mapped to in 
    the programming manual.
    """
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    frames = 20
    SensTqp = 4095
    print(f'{SensTqp=}')
    print(f'{get_demod_freq(SensTqp)=}')
    
    print(f'{get_tqp(demod_freq=2137)=}')
    settings = (
        
        ## Required:
        ('CamMode', CamMode.RAW_IQ),
        # ('CamMode', CamMode.INTENSITY),
        ('SensNFrames', frames), # Frames taken when triggered
        # ('SensExpTime', finddefault),
        ('BSEnable', 0), # Bias suppression enable (offset compensation)
        
        ## Optional:
        # ('SensTqp', 10000),
        ('SensTqp', SensTqp), # (0, 4095?) # time quarter period
        # ('SensTqp', 128), # (0, 4095?) # time quarter period
        ('SensNavM2', 10), # (1, 255) #  Num avg/demod cycles per frame = SensNavM2 * 2 + 2
        
        # ('SensNDarkFrames', finddefault), # must be >= 7
        ('DdsGain', 2), # (0-3)
        ('TrigFreeExtN', 1), # ext trig: 0, free run: 1
        ('TrigExtSrcSel', 0), # Sets trigger source to 0
        ('AcqStop', 0), # Sets acquisition to running
        ('EnSynFOut', 1),
    )

    for k, v in settings:
        try:
            print("Setting: ", k, v)
            setattr(heSys.map, k, v)
        except RuntimeError:
            print(f'Could not set map property {k} to {v}')

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_I16Q16'], 0, 0, 0)

    def on_timer():
        """This gets called every update of the plot window"""
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        cd = heSys.ProcessCamData(1, 0, 0)
        # print("ProcessCamData", on_timer.cnt, "returned", cd.contents.data)

        img = heSys.GetCamData(1, 0, 0)
        data = img.contents.data
        data = LibHeLIC.Ptr2Arr(
            data, (frames, 300, 300, 2), ct.c_int16
        )

        # integrate intensity over frames and I/Q -- oddly no sqrt
        intensity = (
            data[1:, :, :, :].sum(axis=0, dtype=np.int16)
                              .sum(axis=2, dtype=np.int16)
        )
        I = data[:, :, :, 0]
        Q = data[:, :, :, 1]
        # Rotate counterclockwise once after processing
        intensity = np.rot90(intensity)
        data_slice = np.rot90(data[:, :, 150, 0])

        # I suspect this ignores the first set of images to accumulate enough
        # I and Q data before integrating, but this is just a hunch.
        ignore = 5
        if on_timer.cnt < ignore:
            print("ignore image")

        elif on_timer.cnt == ignore:
            print("make fixpattern image")
            on_timer.fixPtrn = intensity.copy()

            # Avoid overflow / integer wraparound
            intensity_disp = np.uint8(np.clip(intensity - on_timer.fixPtrn + 128, 0, 255))

            on_timer.f.suptitle('GetCamData')

            plt.subplot(1, 3, 1)
            on_timer.imHdl1 = plt.imshow(
                I.sum(axis=0), interpolation='nearest'
            )
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            on_timer.imHdl2 = plt.imshow(
                Q.sum(axis=0), interpolation='nearest'
            )
            plt.colorbar()

            plt.subplot(1, 3, 3)
            on_timer.imHdl3 = plt.imshow(
                intensity_disp, vmin=0, vmax=255, cmap='gray'
            )
            plt.colorbar()

        else:
            # intensity_diff = intensity - on_timer.fixPtrn
            intensity_diff = intensity - on_timer.fixPtrn
            
            # Avoid overflow / integer wraparound
            intensity_disp = np.uint8(np.clip(intensity_diff + 128, 0, 255))
            # print(intensity_disp.mean(), intensity_disp.std())

            on_timer.imHdl1.set_array(I.sum(axis=0))
            on_timer.imHdl2.set_array(Q.sum(axis=0))
            on_timer.imHdl3.set_array(intensity_disp)
        
            
        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        """Sends a -1 signal to close the plot window"""
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    # timer = f.canvas.new_timer(interval=200)  # Slower updates are fine when prioritizing image quality (lowest noise)
    timer = f.canvas.new_timer(interval=1)  # interval in ms, will pyqtgraph be faster?
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    
    heSys.Close()
    
    
def test_hdr():
    """
    Intensity Image - Low noise / clean image version
    using DF_Hf, which intensity is mapped to in 
    the programming manual.
    
    This is HDR
    """
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    frames = 64
    settings = (
        
        ## Required:
        ('CamMode', CamMode.INTENSITY),
        ('SensNFrames', frames), # Frames taken when triggered
        # ('SensExpTime', finddefault),
        ('BSEnable', 0), # Bias suppression enable (offset compensation)
        
        ## Optional:
        # ('SensTqp', 10000),
        ('SensTqp', 4095), # (0, 4095?) # time quarter period
        # ('SensTqp', 128), # (0, 4095?) # time quarter period
        ('SensNavM2', 255), # (1, 255) #  Num avg/demod cycles per frame = SensNavM2 * 2 + 2
        
        ('SensNDarkFrames', 7),  # Minimum 7
        ('DdsGain', 2), # (0-3)
        ('TrigFreeExtN', 1), # ext trig: 0, free run: 1
        ('TrigExtSrcSel', 0), # Sets trigger source to 0
        ('AcqStop', 0), # Sets acquisition to running
    )

    for k, v in settings:
        try:
            print("Setting: ", k, v)
            setattr(heSys.map, k, v)
        except RuntimeError:
            print(f'Could not set map property {k} to {v}')

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_Hf'], 0, 0, 0)

    def on_timer():
        """This gets called every update of the plot window"""
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        cd = heSys.ProcessCamData(1, 0, 0)
        print("ProcessCamData", on_timer.cnt, "returned", cd.contents.data)

        img = heSys.GetCamData(1, 0, 0)
        data = img.contents.data
        # data = LibHeLIC.Ptr2Arr(
        #     data, (frames, 300, 300), ct.c_int16
        # )
        
        # Changed from c_int16 to c_float -- why does this work
        # It works because this is float32 HDR, see programming manual
        data = LibHeLIC.Ptr2Arr(
            data, (frames, 300, 300), ct.c_float  
        )

        ignore = 0
        if on_timer.cnt < ignore:
            print("ignore image")
        elif on_timer.cnt == ignore:
            on_timer.f.suptitle('HDR Image from GetCamData')
            plt.subplot(1, 1, 1)
            # Average across the frame dimension (axis 0) to get a 2D image
            img_2d = np.mean(data, axis=0)
            p_low, p_high = np.percentile(img_2d, [1, 99])  # Use 1st and 99th percentiles
            img_2d_stretched = np.clip((img_2d - p_low) / (p_high - p_low) * 255, 0, 255)
            on_timer.imHdl1 = plt.imshow(
            np.rot90(img_2d_stretched), vmin=0, vmax=255, cmap='gray'
            )
            plt.colorbar()
        else:
            # Average across frames for display
            img_2d = np.mean(data, axis=0)
            
            # Check actual range:
            print(f"Image range: [{img_2d.min():.2f}, {img_2d.max():.2f}]")
            print(f"Image mean: {img_2d.mean():.2f}, std: {img_2d.std():.2f}")

            # Try contrast stretching:
            p_low, p_high = np.percentile(img_2d, [1, 99])  # Use 1st and 99th percentiles
            img_2d_stretched = np.clip((img_2d - p_low) / (p_high - p_low) * 255, 0, 255)

            on_timer.imHdl1.set_array(np.rot90(img_2d_stretched))

        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        """Sends a -1 signal to close the plot window"""
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=100)  # interval in ms, will pyqtgraph be faster?
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    
    heSys.Close()
    
    
def test7():
    heSys = LibHeLIC()
    heSys.Open(0, sys='c3cam_sl70')

    frames = 150
    settings = (
        ('SensTqp', 712),
        ('SensNavM2', 2),
        ('SensNFrames', frames),
        ('BSEnable', 1),
        ('DdsGain', 2),
        ('TrigFreeExtN', 1),
        ('TrigExtSrcSel', 0),
        ('CamMode', 4),
        ('AcqStop', 0)
    )

    for k, v in settings:
        try:
            setattr(heSys.map, k, v)
        except RuntimeError:
            error('Could not set map property %s to %s', k, v)

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_A16Z16'],
                       LibHeLIC.CamDataProperty['DP_INTERP_CROSS'], 0, 0)

    def on_timer():
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)
        print("sizeof ushort:", ct.sizeof(ct.c_ushort))

        data = heSys.GetCamArr(1)

        cd = heSys.ProcessCamData(1, 0, 0)

        if on_timer.cnt == 0:
            on_timer.f.suptitle('GetCamData')
            plt.subplot(1, 2, 1)
            on_timer.imHdl1 = plt.imshow(data[:, :, 1], interpolation='nearest')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            on_timer.imHdl2 = plt.imshow(data[:, :, 0], interpolation='nearest')
            plt.colorbar()
        else:
            on_timer.imHdl1.set_array(data[:, :, 1])
            on_timer.imHdl2.set_array(data[:, :, 0])

        print("Z-Value", data[10, 10, 1] / 32)
        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=50)
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    heSys.Close()

def test8():
    heSys = LibHeLIC()
    heSys.Open(0, sys='c3cam_sl70')
    heSys.SetTimeout(0)

    frames = 150
    hwin = 10
    settings = (
        ('SensTqp', 4065),
        # ('SensTqp', 699),
        ('SensNavM2', 2),
        ('SensNFrames', frames),
        ('BSEnable', 1),
        ('DdsGain', 2),
        ('TrigFreeExtN', 1),
        ('TrigExtSrcSel', 0),
        ('ExSimpMaxHwin', hwin),
        ('CamMode', 5),
        ('AcqStop', 0)
    )

    for k, v in settings:
        try:
            setattr(heSys.map, k, v)
        except RuntimeError:
            error('Could not set map property %s to %s', k, v)

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_Z16A16P16'], 0, 0, 0)

    def on_timer():
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        data = heSys.GetCamArr(1)
        cd = heSys.ProcessCamData(1, 0, 0)

        if on_timer.cnt == 0:
            on_timer.f.suptitle('GetCamData')

            plt.subplot(2, 3, 1)
            on_timer.imHdl1 = plt.imshow(data[:, :, hwin + 1, 0], interpolation='nearest')
            plt.colorbar()
            plt.title('Z - values')

            plt.subplot(2, 3, 2)
            on_timer.imHdl2 = plt.imshow(data[:, :, hwin + 1, 1], interpolation='nearest')
            plt.colorbar()
            plt.title('A - values')

            plt.subplot(2, 3, 3)
            on_timer.imHdl3 = plt.imshow(data[:, :, hwin + 1, 2], interpolation='nearest')
            plt.colorbar()
            plt.title('Phi - values')

            plt.subplot(2, 3, 4)
            on_timer.imHdl4 = plt.plot(data[10, 10, :, 2] / 8192.0)
            plt.title('Pixel')

            plt.subplot(2, 3, 5)
            on_timer.imHdl5 = plt.plot(data[10, 10, :, 1])
            plt.title('Ascan')

            print("Zvalue:", data[10, 10, hwin + 1, 0] / 32)
        else:
            on_timer.imHdl1.set_array(data[:, :, hwin + 1, 0])
            on_timer.imHdl2.set_array(data[:, :, hwin + 1, 1])
            on_timer.imHdl3.set_array(data[:, :, hwin + 1, 2])

            plt.subplot(2, 3, 4)
            plt.cla()
            plt.plot(data[10, 10, :, 2] / 8192.0)

            plt.subplot(2, 3, 5)
            plt.cla()
            plt.plot(data[10, 10, :, 1])

            print("Zvalue:", data[10, 10, hwin + 1, 0] / 32)

        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        on_timer.cnt = -1

    f = plt.figure()
    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect('close_event', on_close)

    timer = f.canvas.new_timer(interval=50)
    timer.add_callback(on_timer)
    timer.start()

    plt.show()
    heSys.Close()
    
def acquire_IQ_image_single():
    """Wrapper for single image acquisition - menu compatible."""
    _get_image(mode=AcquisitionMode.SINGLE)


def acquire_IQ_n_images():
    """Wrapper for N frames acquisition - menu compatible."""
    n = input("Enter number of acquisitions (default 10): ").strip()
    n_acquisitions = int(n) if n else 10
    e = input("enter early stop (default 0): ").strip()
    early_stop = int(e) if e else n_acquisitions
    _get_image(mode=AcquisitionMode.N_FRAMES, n_acquisitions=n_acquisitions, early_stop=early_stop)


def acquire_IQ_image_continuous():
    """Wrapper for continuous acquisition - menu compatible."""
    _get_image(mode=AcquisitionMode.CONTINUOUS)


def _get_image(mode=AcquisitionMode.CONTINUOUS, n_acquisitions=10, early_stop=None):
    """
    Acquire intensity images from HeLIC camera with different modes.
    
    Parameters:
    -----------
    mode : AcquisitionMode
        - CONTINUOUS: Stream images continuously until window is closed
        - SINGLE: Acquire one image and display it
        - N_FRAMES: Acquire n_acquisitions images and then stop
    n_acquisitions : int
        Number of acquisitions to capture (only used for N_FRAMES mode)
    frames_per_acquisition : int
        Number of frames to average per acquisition (default 64)
    """
    
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    
    frames = 10
    # frames = 16
    # SensTqp = 4064 # 2137 hz, lowest?
    
    SensTqp = get_tqp(demod_freq=2137)
    # SensTqp = 4060
    # SensNavM2 = 255
    SensNavM2 = 1
    CalDur1Cyc = 1
    demod_freq = get_demod_freq(SensTqp=SensTqp)
    settings = (
        ## Required:
        # ('CamMode', CamMode.INTENSITY), # Poorly named mode, but what can we do.
        ('CamMode', CamMode.RAW_IQ),
        ('SensNFrames', frames),
        ('BSEnable', 0),
        
        # required for external acquisition trigger
        ('TrigFreeExtN', 0), # 0: external triggering, 1: internal
        ('ExtTqp', 1),
        ('ExtTqpPuls', 0),
        ('EnTrigOnPos', 0),
        ('SingleVolume', 0),
        
        # ('TrigFreeExtN', 1),
        ## Optional:
        # ('SingleVolume', 1),
        # ('SensTqp', SensTqp), # (1, 4095)
        ('SensNavM2', SensNavM2), # (1, 255) 
        ('CalDur1Cyc', CalDur1Cyc), # 1: offset compensation T_offset = 1 cycle, 0 uses SensCaldur
        ('DdsGain', 2), # (1, 3), default was 2, manual says best SNR is 2
        ('TrigExtSrcSel', 0), # selects triggering source but there's only one on the Helidriver
        ('AcqStop', 0), # 0: Acquisition running, 1: Acquisition stopped
        ('EnSynFOut', 0), # Provides the sync freq on OUT3 (and encoder??)
    )

    for k, v in settings:
        try:
            print("Setting: ", k, v)
            setattr(heSys.map, k, v)
        except RuntimeError:
            print(f'Could not set map property {k} to {v}')
    
    print(f"{demod_freq =:.4f}Hz")
    # max_framerate = 3800 # frames / sec
    
    # CalDur1Cyc = heSys.GetReg('CalDur1Cyc')
    # SensCaldur = heSys.GetReg('SensCaldur')
    # if CalDur1Cyc == 1:
    #     print('Offset compensation takes exactly 1 cycle')
    #     T_offset = 1 / 70e6
    # else:
    #     print('Offset compensation uses "SensCaldur" register')
    #     T_offset = (SensCaldur + 58) / 35e6 
        
    # num_demod_cycles = SensNavM2 * 2 + 2
    # time_between_frames = (num_demod_cycles / demod_freq) + T_offset
    # framerate = 1 / time_between_frames
    # if framerate > 3800:
    #     print(f'Requested {framerate} fps is higher than max allowed {max_framerate} fps')
    #     # EARLY RETURN!
    #     return   
    # print(f"Got: {CalDur1Cyc=}")
    # print(f"Got: {SensCaldur=}")
    # print(f"Got: {T_offset=}")
    # print(f"Got: {time_between_frames=}")
    # print(f"Got: {framerate=}")
    
    alloc = heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_I16Q16'], 0, 0, 0)
    print(f"Alloc: {alloc}")
    
    f = plt.figure(figsize=(8, 8))
    
    if mode == AcquisitionMode.SINGLE:
        acquire_single(heSys, frames, f)
        plt.show()
    
    elif mode == AcquisitionMode.N_FRAMES:
        acquire_n_images(heSys, frames, f, n_acquisitions, early_stop)
        plt.show()
    
    else:
        acquire_continuous(heSys, frames, f)
    
    
    heSys.Close()


def acquire_single(heSys, frames, fig):
    """Acquire and display a single image."""
    ignore = 0
    
    # Acquire images until we get past the ignore phase
    for i in range(ignore + 1):
        res = heSys.Acquire()
        print(f"Acquire {i} returned {res}")
        
        if i < ignore:
            print("Ignoring image")
            continue
        
        else:
            # Process the final image
            cd = heSys.ProcessCamData(1, 0, 0)
            img = heSys.GetCamData(1, 0, 0)
            data = img.contents.data
            data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)
            
            I = data[:, :, :, 0]
            Q = data[:, :, :, 1]
            
            # amplitude = data[1:,:,:,:].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
            amplitude = data.sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)

            # img_2d = np.mean(data, axis=0)
            # print(f"Image range: [{img_2d.min():.2f}, {img_2d.max():.2f}]")
            # print(f"Image mean: {img_2d.mean():.2f}, std: {img_2d.std():.2f}")
            
            # # Contrast stretching
            # p_low, p_high = np.percentile(img_2d, [1, 99])
            # img_2d_stretched = np.clip((img_2d - p_low) / (p_high - p_low) * 255, 0, 255)
            
            amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))
            
            fig, axs = plt.subplots(1, 3, sharey=True)
            fig.suptitle(f'Single Acquisition, integrated over {frames} frames')
            axs[0].imshow(I.sum(axis=0))
            axs[0].set_title('I')
            # axs[0].colorbar()
            axs[1].imshow(Q.sum(axis=0))
            axs[1].set_title('Q')
            axs[2].imshow(amplitude_disp)
            axs[2].set_title('Amplitude')
            # axs[1].colorbar()
            print("Single acquisition complete")


def acquire_n_images(heSys, frames, fig, n_acquisitions, early_stop):
    """Acquire N images and stop."""
    
    ignore = 0
    for i in range(ignore + n_acquisitions):
        res = heSys.Acquire()
        print(f"Acquire {i}: {res}")
        alloc = heSys.AllocCamData(i, LibHeLIC.CamDataFmt['DF_I16Q16'], 0, 0, 0)
        print(f"Alloc {i}: {alloc}")
        
        if i < ignore:
            print("Ignoring image")
            continue
        else:
            
            if i>early_stop:
                continue
            # Process 
            cd = heSys.ProcessCamData(i, 0, 0)
            img = heSys.GetCamData(i, 0, 0)
            data = img.contents.data
            data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)
            I = data[:, :, :, 0]
            Q = data[:, :, :, 1]
            amplitude = data[1:,:,:,:].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
            # amplitude_disp = np.uint8(np.clip(amplitude + 128, 0, 255))

            fig, axs = plt.subplots(1, 3, sharey=True)
            fig.suptitle(f'Acquisition {i}, integrated over {frames} frames')
            axs[0].imshow(np.rot90(I.sum(axis=0)), cmap='gray')
            axs[0].set_title('I')
            axs[1].imshow(np.rot90(Q.sum(axis=0)), cmap='gray')
            axs[1].set_title('Q')
            axs[2].imshow(np.rot90(amplitude), cmap='gray')
            axs[2].set_title('Amplitude')
            print(f"Acquisition {i} complete")
            # setattr(heSys.map, "AcqStop", 1)


def acquire_continuous(heSys, frames, fig):
    """Continuous streaming acquisition."""
    
    def on_timer():
        """This gets called every update of the plot window"""
        if on_timer.cnt < 0:
            print("Stopping continuous acquisition")
            return
        
        res = heSys.Acquire()
        print(f"Acquire {on_timer.cnt} returned {res}")
        cd = heSys.ProcessCamData(1, 0, 0)
        
        img = heSys.GetCamData(1, 0, 0)
        data = img.contents.data
        data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)
        amplitude = (
            data[1:, :, :, :].sum(axis=0, dtype=np.int16)
                              .sum(axis=2, dtype=np.int16)
        )

        # Rotate counterclockwise once after processing
        amplitude = np.rot90(amplitude)
        
        # This is old fix pattern code that we are nearly sure is both not necessary
        # and potentially obscuring the functionality.
        # It does, however, ensure all of the elements in your picture can be in view.
        
        ignore = 1
        if on_timer.cnt < ignore:
            print("Ignoring image")
        elif on_timer.cnt == ignore:
            print("make fixpattern image")
            on_timer.f.suptitle('Continuous Streaming')
            on_timer.fixPtrn = amplitude.copy()
            
            amplitude_disp = np.uint8(np.clip(amplitude - on_timer.fixPtrn + 128, 0, 255))
            
            plt.subplot(1, 1, 1)
            on_timer.imHdl1 = plt.imshow(
                amplitude_disp, vmin=0, vmax=255, cmap='gray'
                )
            plt.colorbar()
        else:
            amplitude_diff = amplitude - on_timer.fixPtrn
            
            amplitude_disp = np.uint8(np.clip(amplitude_diff + 128, 0, 255))
            
            on_timer.imHdl1.set_array(amplitude_disp)
        # ignore = 1
        # if on_timer.cnt < ignore:
        #     print("Ignoring image")
        # elif on_timer.cnt == ignore:
        #     # This block gets hit only once to draw the subplot elements only once
        #     on_timer.f.suptitle('Continuous Streaming')
        #     amplitude = np.uint8(np.clip(amplitude + 128, 0, 255))
        #     plt.subplot(1, 1, 1)
        #     on_timer.imHdl1 = plt.imshow(
        #         amplitude, vmin=0, vmax=255, cmap='gray'
        #         )
        #     plt.colorbar()
        # else:
        #     amplitude = np.uint8(np.clip(amplitude + 128, 0, 255))
        #     on_timer.imHdl1.set_array(amplitude)
        
        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1
    
    def on_close(event):
        """Sends a -1 signal to close the plot window"""
        on_timer.cnt = -1
    
    on_timer.f = fig
    on_timer.cnt = 0
    fig.canvas.mpl_connect('close_event', on_close)
    
    # timer = fig.canvas.new_timer(interval=1)
    timer = fig.canvas.new_timer(interval=200)
    timer.add_callback(on_timer)
    timer.start()
    plt.show()

    
def test_connected_cameras():
    print ('-'*20)
    heSys=LibHeLIC()
    [count,serials] = LibHeLIC.GetSerials()
    print ('No of installed heliCams: ' + str(count))
    for item in serials:
        print(item)
        
def testz():
  print ('-'*20)
  print ('Used Python version:')
  print (sys.version)
  print ('-'*20)
  print ('Search libHeLIC in:')
  for ImportPath in sys.path:
      print (ImportPath)
  print ('-'*20)
  print ('Numpy version:')
  print (np.version.version)
  print ('Matplotlib version:')
  print (mpl.__version__)
  print ('libHeLIC version:')
  heSys=LibHeLIC()
  print(str(LibHeLIC.GetVersion()[1][3]) +'.'+ str(LibHeLIC.GetVersion()[1][2])+'.'+ str(LibHeLIC.GetVersion()[1][1])+'.'+ str(LibHeLIC.GetVersion()[1][0]))

  
if __name__ == '__main__':
    def MenuSelection():
        entries = {
            '1': (test1, 'show registerDescr of sys=c3cam_sl70'),
            '2': (test2, 'surface acquisition A16Z16 - simple max (Unfixed)'),
            '3': (test3, 'volume acquisition A16 (Unfixed)'),
            '4': (test4, 'volume acquisition A8 (Unfixed)'),
            '5': (test5, 'raw data acquisition I16Q16 (Unfixed?)'),
            '6': (test6, '"intensity" image I16Q16'),
            'h': (test_hdr, 'hdr image DF_Hf'),
            # 'q': (test6b, 'intensity image hf'),
            '7': (test7, 'surface acquisition, cross remove A16Z16 (Unfixed)'),
            '8': (test8, 'surface acquisition, extended simple max Z16A16P16 (Unfixed)'),
            '9': (test_connected_cameras, 'scan connected heliCams and print the serial numbers'),
            'as': (acquire_IQ_image_single, 'acquire single image'),
            'an': (acquire_IQ_n_images, 'acquire N images'),
            'ac': (acquire_IQ_image_continuous, 'acquire continuous stream'),
            'z': (testz, 'print out versions from python, print out path'),
        }
        while True:
            print('\n' + '-' * 40)
            for k, (_, desc) in entries.items():
                print(f'{k}: {desc}')
            print('x: exit')
            choice = input('Select test: ').strip().lower()
            if choice == 'x':
                break
            if choice not in entries:
                print('Invalid selection')
                continue
            func, desc = entries[choice]
            print(f'\nRunning {func.__name__}(): {desc}\n')
            try:
                func()   # blocks until window is closed
            except Exception as err:
                print('ERROR:', err)
            print('\nTest finished, returning to menu.')
    MenuSelection()