"""Soft E-stop with a button."""

# Defer annotation evaluation so 3.9+/3.10+ hints (e.g. `tuple[bytes, int]`,
# `npt.NDArray | None`) don't fail to import on Python 3.8 (ROS Noetic / NUC).
from __future__ import annotations

import contextlib
import os
import time

import numpy as np
import numpy.typing as npt
import pyaudio
from threading import Lock
import argparse


@contextlib.contextmanager
def suppressed_alsa_stderr():
    """Silence the ALSA/PortAudio banner PyAudio emits on every initialization.

    Those warnings are written straight to fd 2 by C libraries, so redirecting
    `sys.stderr` does not catch them. Callers that poll for a hot-plugged device
    re-initialize PyAudio on a timer (PortAudio snapshots the device list at
    init, so re-probing is the only way to notice a newly attached button), and
    without this every probe would reprint the whole banner.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for every PyAudio device that can capture audio."""
    with suppressed_alsa_stderr():
        audio = pyaudio.PyAudio()
        try:
            devices = []
            for index in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(index)
                if info["maxInputChannels"] > 0:  # Only consider input devices
                    devices.append((index, str(info["name"])))
            return devices
        finally:
            audio.terminate()


def find_input_device(name_substring: str) -> tuple[int, str] | None:
    """(index, name) of the first input device whose name contains
    `name_substring` (case-insensitive), or None when none is connected.

    Prefer this over a hard-coded index: PyAudio indices shift with USB
    enumeration order, so an index that was right at one boot can silently point
    at a different device (or an output-only one) after a replug.
    """
    needle = name_substring.lower()
    for index, name in list_input_devices():
        if needle in name.lower():
            return index, name
    return None


class Button:
    """Physical button that connects over audio jack."""

    PYAUDIO_STREAM_TROUBLESHOOTING = (
        "The Pyaudio stream not opening error is often caused by another process using "
        "the microphone and/or audio device. To address this, terminate the code and "
        "try the following:\n"
        "  1. Close all applications (e.g., System Settings) that may be accessing "
        "audio devices.\n"
        "  2. If that still doesn't address it, run `sudo alsa force-reload`.\n"
        "     Wait a few (~5) secs after running this command to restart the node,\n"
        "     and note that you may have to run this command multiple times.\n"
        "Note that until this is addressed, the e-stop button will not be working."
    )

    def __init__(
        self,
        input_device_index,
        max_threshold: int = 10000,
        min_threshold: int = -10000,
    ) -> None:

        self.start_time = time.time()
        self.prev_data_arr: npt.NDArray | None = None
        self.detection_time: float | None = None
        # Detection thresholds depend on the audio adapter's gain. The lab's
        # NUC hardware spikes to ~+/-10000 on a press (the defaults). Other
        # adapters (e.g. a Mac USB dongle) are much quieter (~+/-200), so pass
        # lower thresholds for those rather than editing these defaults.
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

        self.audio = pyaudio.PyAudio()

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,  # The e-stop button is mono
                rate=48000,
                input=True,
                frames_per_buffer=4800,
                input_device_index=input_device_index,
                stream_callback=self.__audio_callback,
            )
        except OSError as exc:
            # Release the PortAudio instance before bailing out: callers that treat
            # an absent/busy button as a normal state (transfer_button_listener)
            # retry on a timer, and leaking one of these per attempt adds up.
            self.audio.terminate()
            raise RuntimeError(
                (
                    f"Error opening audio device {input_device_index}. "
                    f"{Button.PYAUDIO_STREAM_TROUBLESHOOTING}\n\n"
                    f"Exception: {exc}"
                ),
            )
        
        self.button_lock = Lock()
        self.button_value = False

        # Peak absolute audio amplitude seen since the last get_peak() call.
        # Useful for tuning max/min_threshold to a given adapter's gain.
        self.peak_lock = Lock()
        self.peak_value = 0

    def close(self) -> None:
        """Close the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def check(self) -> bool:
        """Check if the e-stop button has been pressed."""
        with self.button_lock:
            return self.button_value
        
    def reset(self) -> None:
        """Reset the e-stop button."""
        with self.button_lock:
            self.button_value = False

    def get_peak(self) -> int:
        """Return the peak absolute audio amplitude seen since the last call,
        then reset the tracker. 0 if no audio has been processed yet."""
        with self.peak_lock:
            peak = self.peak_value
            self.peak_value = 0
            return peak

    def __audio_callback(
        self, data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple[bytes, int]:
        del frame_count, time_info, status  # unused

        # Skip the first few seconds of data, to avoid initial noise
        if time.time() - self.start_time < 2:
            return (data, pyaudio.paContinue)

        data_arr = np.frombuffer(data, dtype=np.int16)

        # Track the peak absolute amplitude for threshold-tuning diagnostics.
        with self.peak_lock:
            self.peak_value = max(self.peak_value, int(np.abs(data_arr).max()))

        # Check if the e-stop button has been pressed
        if Button.rising_edge_detector(
            data_arr,
            self.prev_data_arr,
            self.max_threshold,
        ) or Button.falling_edge_detector(
            data_arr,
            self.prev_data_arr,
            self.min_threshold,
        ):
            if self.detection_time is None or time.time() - self.detection_time > 2:
                self.detection_time = time.time()
                with self.button_lock:
                    self.button_value = True

        # Return the data
        self.prev_data_arr = data_arr

        # print("In audio callback: ", data_arr)
        return (data, pyaudio.paContinue)

    @staticmethod
    def rising_edge_detector(
        curr_data_arr: npt.NDArray,
        prev_data_arr: npt.NDArray | None,
        threshold: int | float,
    ) -> bool:
        """Detects whether there is a rising edge in `curr_data_arr` that
        exceeds `threshold`. In other words, this function returns True if
        there is a point in `curr_data_arr` that is greater than `threshold`
        and the previous point is less than `threshold`.

        Although this method of detecting a rising edge is suceptible to noise
        (since it only requires two points to determine an edge), in practice
        the e-stop button's signal has little noise. If noise is an issue
        moving forward, we can add a filter to smoothen the signal, and then
        continue using this detector.

        Parameters
        ----------
        curr_data_arr: npt.NDArray
            The current data array
        prev_data_arr: Optional[npt.NDArray]
            The previous data array
        threshold: Union[int, float]
            The threshold that the data must cross to be considered a rising edge

        Returns
        -------
        is_rising_edge: bool
            True if a rising edge was detected, False otherwise
        """
        is_above_threshold = curr_data_arr > threshold
        if np.any(is_above_threshold):
            first_index_above_threshold = np.argmax(is_above_threshold)
            # Get the previous value
            if first_index_above_threshold == 0:
                if prev_data_arr is None:
                    # If the first datapoint is above the threshold, it's not a
                    # rising edge
                    return False
                prev_value = prev_data_arr[-1]
            else:
                prev_value = curr_data_arr[first_index_above_threshold - 1]
            # If the previous value is less than the threshold, it is a rising edge
            return prev_value < threshold
        # If no point is above the threshold, there is no rising edge
        return False

    @staticmethod
    def falling_edge_detector(
        curr_data_arr: npt.NDArray,
        prev_data_arr: npt.NDArray | None,
        threshold: int | float,
    ) -> bool:
        """Detects whether there is a falling edge in `curr_data_arr` that
        exceeds `threshold`. In other words, this function returns True if
        there is a point in `curr_data_arr` that is less than `threshold` and
        the previous point is greater than `threshold`.

        Parameters
        ----------
        curr_data_arr: npt.NDArray
            The current data array
        prev_data_arr: Optional[npt.NDArray]
            The previous data array
        threshold: Union[int, float]
            The threshold that the data must cross to be considered a falling edge

        Returns
        -------
        is_falling_edge: bool
            True if a falling edge was detected, False otherwise
        """
        # Flip all signs and call the rising edge detector
        return Button.rising_edge_detector(
            -curr_data_arr,
            None if prev_data_arr is None else -prev_data_arr,
            -threshold,
        )


if __name__ == "__main__":

    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="The index of the input device")
    parser.add_argument("--max_threshold", type=int, default=10000,
                        help="Rising-edge detection threshold (lower for quiet adapters, e.g. 200 on a Mac dongle)")
    parser.add_argument("--min_threshold", type=int, default=-10000,
                        help="Falling-edge detection threshold (e.g. -200 on a Mac dongle)")
    args = parser.parse_args()

    # -1 is user emergency stop button
    # 9 is experimentor emergency stop button
    # 7 is transfer button

    if args.id is None:
        for index, name in list_input_devices():
            print(f"Device {index}: {name}")
        raise ValueError("Please provide the input device index")

    button = Button(args.id, max_threshold=args.max_threshold, min_threshold=args.min_threshold)
    while True:
        if button.check():
            print("E-stop pressed!")
            break
        time.sleep(0.01)
    
    button.close()
