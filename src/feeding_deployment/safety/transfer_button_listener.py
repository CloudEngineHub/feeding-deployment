"""Compute-side transfer button -> /transfer_button.

A fallback for the iPad transfer button, which reaches the robot through the
webapp (App.vue detects the click, robot_executing relays it on /webapp_to_robot)
and fails intermittently and silently. This node reads a button plugged straight
into the compute box and publishes it on /transfer_button; WebInterface consumes
BOTH sources, so whichever fires first satisfies a transfer wait.

The button is optional. When it is not plugged in this node stays quiet and keeps
re-probing, so it never breaks the rest of launch_robot.sh and a mid-session
plug-in starts working without relaunching the stack.
"""

import argparse
import time

import rospy
from std_msgs.msg import Bool

from feeding_deployment.safety.button import (
    Button,
    find_input_device,
    list_input_devices,
    suppressed_alsa_stderr,
)

BUTTON_CHECK_FREQUENCY = 100
# How often to re-probe for the adapter while it is absent. PortAudio snapshots
# the device list when it initializes, so noticing a hot-plug means re-probing.
DEVICE_POLL_INTERVAL_S = 5.0
# Audio adapter the transfer button plugs into on the compute box, matched as a
# case-insensitive substring of the PyAudio device name.
DEFAULT_DEVICE_NAME = "Plugable"


class TransferButtonListener:
    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        button_id: int = None,
        max_threshold: int = 10000,
        min_threshold: int = -10000,
    ):
        self.device_name = device_name
        self.button_id = button_id
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.button_pub = rospy.Publisher("/transfer_button", Bool, queue_size=1)

    def _resolve(self):
        """(index, name) of the button adapter, or None when it is not attached.

        An explicit --button_id wins over the name so a second adapter can still
        be addressed directly; we look its name up anyway so the log line says
        what actually got opened.
        """
        if self.button_id is not None:
            for index, name in list_input_devices():
                if index == self.button_id:
                    return index, name
            return None
        return find_input_device(self.device_name)

    def acquire(self):
        """Block until the button is attached and its stream opens, then return
        the Button. Returns None if ROS shuts down first.

        A missing button is a normal state here, not an error, so it must not
        raise: this is a fallback path and the rest of the stack has to come up
        regardless. We log only on transitions, so an absent button costs one
        line and then silence however long it stays unplugged.
        """
        announced_missing = False
        while not rospy.is_shutdown():
            found = self._resolve()
            if found is not None:
                index, name = found
                try:
                    # Suppress the ALSA/PortAudio banner: this runs on a retry timer,
                    # and reprinting it every cycle while the card is busy would be
                    # exactly the noise this node is supposed to avoid.
                    with suppressed_alsa_stderr():
                        button = Button(
                            index,
                            max_threshold=self.max_threshold,
                            min_threshold=self.min_threshold,
                        )
                    print(f"[transfer_button] found '{name}' (index {index}) - listening")
                    return button
                except RuntimeError as exc:
                    # Device present but the stream would not open -- almost always
                    # another process holding the card. Retry rather than die; it
                    # often frees up. The exception carries Button's troubleshooting
                    # text, which is worth showing (once).
                    reason = (f"'{name}' (index {index}) found but its audio stream "
                              f"would not open.\n{exc}")
            else:
                target = (f"at index {self.button_id}" if self.button_id is not None
                          else f"matching '{self.device_name}'")
                reason = f"no input device {target}"
            if not announced_missing:
                print(f"[transfer_button] {reason}; transfer button not available. "
                      f"Re-checking every {DEVICE_POLL_INTERVAL_S:.0f}s (quiet from here; "
                      f"plug it in any time).")
                announced_missing = True
            time.sleep(DEVICE_POLL_INTERVAL_S)
        return None

    def run(self):
        button = self.acquire()
        if button is None:
            return
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                button_pressed = button.check()

                if button_pressed:
                    print("Transfer button pressed")
                    self.button_pub.publish(Bool(data=button_pressed))
                    button.reset()

                time.sleep(max(0, 1.0/BUTTON_CHECK_FREQUENCY - (time.time() - start_time)))
        finally:
            try:
                button.close()
            except Exception:
                pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device_name", default=DEFAULT_DEVICE_NAME,
                        help="substring of the audio device name the button is plugged into "
                             f"(default: {DEFAULT_DEVICE_NAME!r})")
    parser.add_argument("--button_id", type=int, default=None,
                        help="explicit PyAudio input device index, overriding --device_name. "
                             "Indices move around on replug; prefer --device_name.")
    parser.add_argument("--max_threshold", type=int, default=10000,
                        help="Rising-edge detection threshold (lower for quiet adapters, e.g. 200)")
    parser.add_argument("--min_threshold", type=int, default=-10000,
                        help="Falling-edge detection threshold (e.g. -200 for quiet adapters)")
    parser.add_argument("--list_devices", action="store_true",
                        help="print the available input devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        for index, name in list_input_devices():
            print(f"Device {index}: {name}")
        raise SystemExit(0)

    rospy.init_node("transfer_button_listener")
    transfer_button_listener = TransferButtonListener(
        device_name=args.device_name,
        button_id=args.button_id,
        max_threshold=args.max_threshold,
        min_threshold=args.min_threshold,
    )
    transfer_button_listener.run()
