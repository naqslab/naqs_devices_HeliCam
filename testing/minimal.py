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

# def test_rawIQ():
#     """
#     Amplitude Image using raw I and Q data.
#     DF_I16Q16
#     """
#     heSys = LibHeLIC()
#     heSys.Open(0, sys="c3cam_sl70")
#     heSys.Acquire()
#     frames = 16
#     SensTqp = 1900
#     print(f'{SensTqp=}')

#     settings = (

#         ('CamMode', 0),
#         ('SensNFrames', frames), # Frames taken when triggered
#         ('BSEnable', 0), # Bias suppression enable (offset compensation)

#         ('ExtTqp', 1),
#         ('SensDeltaExp', 0),
#         ('EnTrigOnPos', 0),
#         ('EnSynFOut', 1),
#         ('SensTqp', SensTqp), # (0, 4095) # time quarter period
#         ('SensNavM2', 10), # (1, 255) #  Num avg/demod cycles per frame = SensNavM2 * 2 + 2
#         ('DdsGain', 2), # (0-3)
#         ('TrigExtSrcSel', 0), # Sets trigger source to 0
#         ('TrigFreeExtN', 1), # ext trig: 0, free run: 1
#         ('InvEncCnt', 0),
#         ('AcqStop', 0), # Sets acquisition to running
#     )

#     for k, v in settings:
#         try:
#             print("Setting: ", k, v)
#             setattr(heSys.map, k, v)
#         except RuntimeError:
#             print(f'Could not set map property {k} to {v}')

#     heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_I16Q16'], 0, 0, 0)

#     def on_timer():
#         """This gets called every update of the plot window"""
#         if on_timer.cnt < 0:
#             print("on_timer.stop")
#             return

#         res = heSys.Acquire()
#         print("Acquire", on_timer.cnt, "returned", res)

#         cd = heSys.ProcessCamData(1, 0, 0)

#         img = heSys.GetCamData(1, 0, 0)
#         data = img.contents.data
#         data = LibHeLIC.Ptr2Arr(
#             data, (frames, 300, 300, 2), ct.c_int16
#         )

#         # integrate intensity over frames and I/Qt
#         intensity = (
#             data[1:, :, :, :].sum(axis=0, dtype=np.int16)
#                               .sum(axis=2, dtype=np.int16)
#         )
#         I = data[:, :, :, 0]
#         Q = data[:, :, :, 1]

#         intensity = np.rot90(intensity)

#         # amplitude = np.hypot(I, Q).astype(np.int64)
#         # amplitude = np.sum(amplitude, axis=0)
#         ignore = 7
#         if on_timer.cnt < ignore:
#             print("ignore image")

#         elif on_timer.cnt == ignore:
#             print("make fixpattern image")
#             on_timer.fixPtrn = intensity.copy()

#             intensity_disp = np.uint8(np.clip(intensity - on_timer.fixPtrn + 128, 0, 255))

#             on_timer.f.suptitle(f'SensTqp: {SensTqp}')

#             plt.subplot(1, 3, 1)
#             on_timer.imHdl1 = plt.imshow(
#                 I.sum(axis=0), interpolation='nearest'
#             )
#             plt.colorbar()

#             plt.subplot(1, 3, 2)
#             on_timer.imHdl2 = plt.imshow(
#                 Q.sum(axis=0), interpolation='nearest'
#             )
#             plt.colorbar()

#             plt.subplot(1, 3, 3)
#             on_timer.imHdl3 = plt.imshow(
#                 intensity, vmin=0, vmax=255, cmap='gray_r'
#             )
#             plt.colorbar()

#             # plt.subplot(1, 4, 4)
#             # on_timer.imHdl4 = plt.imshow(
#             #     amplitude, vmin=0, vmax=255, cmap='gray_r'
#             # )
#             # plt.colorbar()
#         else:
#             intensity_diff = intensity - on_timer.fixPtrn

#             intensity_disp = np.uint8(np.clip(intensity_diff + 128, 0, 255))
#             on_timer.imHdl1.set_array(np.rot90(I.sum(axis=0)))
#             on_timer.imHdl2.set_array(np.rot90(Q.sum(axis=0)))
#             on_timer.imHdl3.set_array(intensity_disp)
#             # on_timer.imHdl4.set_array(np.rot90(amplitude))


#         on_timer.f.canvas.draw_idle()
#         on_timer.cnt += 1

#     def on_close(event):
#         """Sends a -1 signal to close the plot window"""
#         on_timer.cnt = -1

#     f = plt.figure()
#     on_timer.f = f
#     on_timer.cnt = 0

#     f.canvas.mpl_connect('close_event', on_close)

#     timer = f.canvas.new_timer(interval=1)  # interval in ms
#     timer.add_callback(on_timer)
#     timer.start()

#     plt.show()

#     print(f'{SensTqp=}')
#     print(heSys.GetReg('SensTqp'))

#     heSys.Close()


def test_rawIQ():
    """
    Amplitude Image using raw I and Q data.
    DF_I16Q16
    """
    heSys = LibHeLIC()
    heSys.Open(0, sys="c3cam_sl70")
    heSys.Acquire()
    frames = 4
    SensTqp = 1900
    SensNavM2 = 10
    print(f"{SensTqp=}")

    settings = (
        ("CamMode", 0),
        ("SensNFrames", frames),  # Frames taken when triggered
        ("BSEnable", 0),  # Bias suppression enable (offset compensation)
        ("ExtTqp", 1),
        ("SensDeltaExp", 0),
        ("EnTrigOnPos", 0),
        ("EnSynFOut", 1),
        ("SensTqp", SensTqp),  # (0, 4095) # time quarter period
        ("SensNavM2",SensNavM2,), # (1, 255) N_demod cycles = SensNavM2 * 2 + 2
        ("DdsGain", 2),  # (0-3)
        ("TrigExtSrcSel", 0),  # Sets trigger source to 0
        ("TrigFreeExtN", 0),  # ext trig: 0, free run: 1
        ("InvEncCnt", 0),
        ("AcqStop", 0),  # Sets acquisition to running
    )

    for k, v in settings:
        try:
            print("Setting: ", k, v)
            setattr(heSys.map, k, v)
        except RuntimeError:
            print(f"Could not set map property {k} to {v}")

    heSys.AllocCamData(1, LibHeLIC.CamDataFmt["DF_I16Q16"], 0, 0, 0)

    def on_timer():
        """This gets called every update of the plot window"""
        if on_timer.cnt < 0:
            print("on_timer.stop")
            return

        res = heSys.Acquire()
        print("Acquire", on_timer.cnt, "returned", res)

        cd = heSys.ProcessCamData(1, 0, 0)

        img = heSys.GetCamData(1, 0, 0)
        data = img.contents.data
        data = LibHeLIC.Ptr2Arr(data, (frames, 300, 300, 2), ct.c_int16)

        # integrate intensity over frames and I/Q
        intensity = (
            data[1:, :, :, :].sum(axis=0, dtype=np.int16).sum(axis=2, dtype=np.int16)
        )
        I = data[:, :, :, 0]
        Q = data[:, :, :, 1]

        intensity = np.rot90(intensity)
        ignore = 0
        if on_timer.cnt < ignore:
            print("ignore image")

        elif on_timer.cnt == ignore:
            print("make fixpattern image")
            on_timer.fixPtrn = intensity.copy()

            intensity_disp = np.uint8(
                np.clip(intensity - on_timer.fixPtrn + 128, 0, 255)
            )

            on_timer.f.suptitle(f"SensTqp: {SensTqp}, frames: {frames}, SensNavM2: {SensNavM2}")

            for i in range(frames):
                my_slice = np.s_[i, 150:, 150:200]
                on_timer.ims[f"I{i}"] = axd[f"I{i}"].imshow(I[my_slice])
                on_timer.ims[f"Q{i}"] = axd[f"Q{i}"].imshow(Q[my_slice])

        else:
            intensity_diff = intensity - on_timer.fixPtrn

            intensity_disp = np.uint8(np.clip(intensity_diff + 128, 0, 255))
            # on_timer.imHdl1.set_array(I)
            # on_timer.imHdl2.set_array(Q)
            for i in range(frames):
                my_slice = np.s_[i, 150:, 150:200]
                on_timer.ims[f"I{i}"].set_data(I[my_slice])
                on_timer.ims[f"Q{i}"].set_data(Q[my_slice])

            # on_timer.imHdl3.set_array(intensity_disp)
        axd["I0"].set_ylabel("I", fontsize=12)
        axd["Q0"].set_ylabel("Q", fontsize=12)

        on_timer.f.canvas.draw_idle()
        on_timer.cnt += 1

    def on_close(event):
        """Sends a -1 signal to close the plot window"""
        on_timer.cnt = -1

    # f = plt.figure()
    # f, axes = plt.subplots(
    #     nrows=2, ncols=frames, figsize=(12, 9), sharex=True, sharey=True
    # )
    layout = [[f"I{i}" for i in range(frames)], 
              [f"Q{i}" for i in range(frames)]]
    
    f, axd = plt.subplot_mosaic(layout, figsize=(20, 12), sharex=True, sharey=True)  # type: ignore
    on_timer.f = f
    on_timer.axd = axd  # Store the dictionary of axes
    on_timer.ims = {}   # Dictionary to store image handles

    on_timer.f = f
    on_timer.cnt = 0

    f.canvas.mpl_connect("close_event", on_close)

    timer = f.canvas.new_timer(interval=1)  # interval in ms
    timer.add_callback(on_timer)
    timer.start()
    plt.tight_layout()
    plt.show()

    print(f"{SensTqp=}")
    print(heSys.GetReg("SensTqp"))

    heSys.Close()


# def test_hdr():
#     """
#     HDR mode test
#     DF_Hf, which intensity is mapped to in the programming manual.

#     """
#     heSys = LibHeLIC()
#     heSys.Open(0, sys="c3cam_sl70")
#     heSys.Acquire()
#     frames = 16 # lowish frame count for faster stream
#     settings = (

#         ## Required:
#         ('CamMode', CamMode.INTENSITY),
#         ('SensNFrames', frames), # Frames taken when triggered
#         # ('SensExpTime', finddefault),
#         ('BSEnable', 0), # Bias suppression enable (offset compensation)

#         ## Optional:
#         # ('SensTqp', 1), # Shouldn't need in HDR
#         # ('SensNavM2', 1), # (1, 255) #  Num avg/demod cycles per frame = SensNavM2 * 2 + 2

#         ('SensNDarkFrames', 7),  # Minimum 7
#         ('DdsGain', 2), # (0-3)
#         ('TrigFreeExtN', 1), # ext trig: 0, free run: 1
#         ('TrigExtSrcSel', 0), # Sets trigger source to 0
#         ('AcqStop', 0), # Sets acquisition to running
#     )

#     for k, v in settings:
#         try:
#             print("Setting: ", k, v)
#             setattr(heSys.map, k, v)
#         except RuntimeError:
#             print(f'Could not set map property {k} to {v}')

#     heSys.AllocCamData(1, LibHeLIC.CamDataFmt['DF_Hf'], 0, 0, 0)

#     def on_timer():
#         """This gets called every update of the plot window"""
#         if on_timer.cnt < 0:
#             print("on_timer.stop")
#             return

#         res = heSys.Acquire()
#         print("Acquire", on_timer.cnt, "returned", res)

#         cd = heSys.ProcessCamData(1, 0, 0)
#         print("ProcessCamData", on_timer.cnt, "returned", cd.contents.data)

#         img = heSys.GetCamData(1, 0, 0)
#         data = img.contents.data

#         # HDR requires float
#         data = LibHeLIC.Ptr2Arr(
#             data, (frames, 300, 300), ct.c_float
#         )

#         ignore = 0
#         if on_timer.cnt < ignore:
#             print("ignore image")
#         elif on_timer.cnt == ignore:
#             on_timer.f.suptitle('HDR Image from GetCamData')
#             plt.subplot(1, 1, 1)
#             # Average across the frame dimension (axis 0) to get a 2D image
#             img_2d = np.mean(data, axis=0)
#             p_low, p_high = np.percentile(img_2d, [1, 99])  # Use 1st and 99th percentiles
#             img_2d_stretched = np.clip((img_2d - p_low) / (p_high - p_low) * 255, 0, 255)
#             on_timer.imHdl1 = plt.imshow(
#             np.rot90(img_2d_stretched), vmin=0, vmax=255, cmap='gray'
#             )
#             plt.colorbar()
#         else:
#             # Average across frames for display
#             img_2d = np.mean(data, axis=0)

#             # Check actual range:
#             print(f"Image range: [{img_2d.min():.2f}, {img_2d.max():.2f}]")
#             print(f"Image mean: {img_2d.mean():.2f}, std: {img_2d.std():.2f}")

#             # Try contrast stretching:
#             p_low, p_high = np.percentile(img_2d, [1, 99])  # Use 1st and 99th percentiles
#             img_2d_stretched = np.clip((img_2d - p_low) / (p_high - p_low) * 255, 0, 255)

#             on_timer.imHdl1.set_array(np.rot90(img_2d_stretched))

#         on_timer.f.canvas.draw_idle()
#         on_timer.cnt += 1

#     def on_close(event):
#         """Sends a -1 signal to close the plot window"""
#         on_timer.cnt = -1

#     f = plt.figure()
#     on_timer.f = f
#     on_timer.cnt = 0

#     f.canvas.mpl_connect('close_event', on_close)

#     timer = f.canvas.new_timer(interval=100)  # interval in ms, will pyqtgraph be faster?
#     timer.add_callback(on_timer)
#     timer.start()

#     plt.show()

#     heSys.Close()


if __name__ == "__main__":
    test_rawIQ()
