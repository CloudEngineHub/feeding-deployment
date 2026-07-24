#!/usr/bin/env python3
"""
map_odom_clamp.py -- absorb spurious Cartographer map->odom jumps while the base
is parked, WITHOUT starving Cartographer.

Why not scan_gate's approach: scan_gate "froze" localization by dropping all
lidar scans while parked/occluded. But starving Cartographer of range data
(while odometry keeps flowing) WEDGES it -- its ordered sensor queue stalls, it
stops publishing a fresh map->odom entirely, the transform ages out of
consumers' TF buffers ("target_frame map does not exist"), and move_base's
costmaps die. It also did not recover from a long starvation without a restart.
(session_20260723_131754: gate CLOSED "parked 300s" at 14:48:41 -> cartographer
tf_bridge "extrapolation into the past" storm, requested-time frozen at the last
scan -> map frame gone -> shared_autonomy_manager RuntimeError on the next nav.)

This node clamps at the OUTPUT instead, so Cartographer always runs and always
publishes a fresh transform:

  Cartographer's map_frame is renamed to "map_carto" (it publishes map_carto->
  odom at ~100 Hz, unchanged). This node owns the map->map_carto edge (planar
  SE2), broadcast at pub_rate_hz with the stamp post-dated by
  publish_lookahead_s -- matching the lua's tf_publish_lookahead_sec on
  map_carto->odom, so the composed map->odom chain never lags a now()-stamped
  move_base/TEB plan lookup (otherwise TEB gets "extrapolation into the future"
  and can't transform the global plan). Every downstream consumer keeps using
  "map" and resolves the chain map -> map_carto -> odom -> base.

  - The correction starts at IDENTITY (map == map_carto), so the loaded pbstream
    and every stored map-frame goal stay numerically valid.
  - While NOT parked: the correction is held constant, so Cartographer's
    map_carto->odom corrections flow straight through to consumers -- normal
    localization, including the fine-parking catch-up during the 15 s
    goal-confirm settle (park_freeze_s is longer than that settle, so the clamp
    is not engaged then).
  - While PARKED (no /cmd_vel[_teleop] for park_freeze_s): any single step in
    map_carto->odom larger than {lin,ang}_jump is ABSORBED into the correction,
    so the net map->odom holds steady -- a parked robot's localization can no
    longer be yanked (the feeding-occlusion jump). Steps below the threshold
    still flow; set the threshold to 0 to pin fully.
  - On release (base commanded again) the accumulated correction WASHES OUT
    back to identity at washout_{lin,ang} rates, so localization re-converges
    to Cartographer's truth. This clamp DEFERS corrections during feeding, it
    does not discard them: a frozen correction would leave the map frame
    permanently shifted and every later goal would be executed at an offset
    spot (observed 2026-07-24 -- 42 absorptions over a 17 min table stay, then
    the robot parked well off the sink).

Parked detection mirrors scan_gate: disabled until the first command ever
arrives (so a dock-parked robot's initial lock flows through unclamped), and
disabled for startup_grace_s (default 300 s) after the first update -- so even
if you drive mid-convergence and then park, the clamp won't pin/absorb the
initial global lock (large corrections that can take a few minutes). After that
warmup, "parked" == park_freeze_s elapsed since the last /cmd_vel[_teleop].
Missing the grace is not fatal here (unlike scan_gate, Cartographer is never
starved; a mid-convergence pin self-heals on the next move), but it avoids a
confusing "map stuck wrong" during bring-up.

Failure behavior: with ~enabled false this publishes identity forever, so
map == map_carto == raw Cartographer (jumpy but correct) -- the safe fallback.
The node is deliberately tiny and runs respawn=true; if it dies the map frame
stops (same failure surface as any single map->odom publisher).
"""

import math
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import Twist, TransformStamped, Vector3
from std_msgs.msg import Bool


def wrap(theta):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


def se2_compose(a, b):
    """Compose two planar transforms a, b given as (x, y, yaw): result = a . b."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (a[0] + ca * b[0] - sa * b[1],
            a[1] + sa * b[0] + ca * b[1],
            wrap(a[2] + b[2]))


def se2_inverse(a):
    """Inverse of a planar transform (x, y, yaw)."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (-(ca * a[0] + sa * a[1]),
            -(-sa * a[0] + ca * a[1]),
            -a[2])


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ClampCore:
    """Pure clamp logic (no ROS) so it can be unit-tested offline.

    Feed the latest map_carto->odom via update(carto, t) and read the returned
    map->map_carto correction. carto/correction are (x, y, yaw) planar tuples;
    t is float seconds. Feed nonzero base commands via note_cmd(t).
    """

    def __init__(self, park_freeze_s=30.0, lin_jump_m=0.05, ang_jump_rad=0.03,
                 startup_grace_s=300.0, washout_lin_mps=0.15,
                 washout_ang_rps=0.15):
        self.park_freeze_s = park_freeze_s     # 0/None disables the parked clamp
        self.lin_jump_m = lin_jump_m           # step in map_carto->odom above
        self.ang_jump_rad = ang_jump_rad       # which (while parked) is absorbed
        self.startup_grace_s = startup_grace_s  # one-time warmup; 0/None disables
        self.washout_lin_mps = washout_lin_mps  # rate the correction decays back
        self.washout_ang_rps = washout_ang_rps  # to identity once NOT parked
        self.correction = (0.0, 0.0, 0.0)      # map->map_carto, starts identity
        self.last_carto = None                 # last map_carto->odom seen
        self.last_cmd_t = None                 # last /cmd_vel* message time
        self.start_t = None                    # first update() time
        self.last_t = None                     # previous update() time (for dt)
        self.last_absorbed = None              # (delta_m, delta_rad) or None

    def correction_magnitude(self):
        """(meters, radians) of the accumulated correction, for diagnostics."""
        return (math.hypot(self.correction[0], self.correction[1]),
                abs(self.correction[2]))

    def _washout(self, dt):
        """Decay the correction toward identity at the washout rates.

        THE reason this exists: absorbing a jump only DEFERS it. Cartographer
        keeps its own (thrashing) estimate, and once the base drives with a clear
        view it re-converges to the truth. If the correction stayed frozen at the
        absorbed value, that stale offset would remain composed on top forever --
        the map frame would sit shifted from reality and every subsequent goal
        would be executed at a physically offset spot (observed 2026-07-24: a
        17 min, 42-absorption storm at the table, then the robot parked well off
        the sink). So once we are no longer parked, bleed the correction back to
        identity: the clamp defers corrections during feeding, it never discards
        them. Slew-limited so the pose slides back smoothly instead of snapping.
        """
        if dt <= 0.0:
            return
        x, y, yaw = self.correction
        d = math.hypot(x, y)
        max_d = self.washout_lin_mps * dt
        if d <= max_d:
            x, y = 0.0, 0.0
        elif d > 0.0:
            scale = (d - max_d) / d
            x, y = x * scale, y * scale
        max_a = self.washout_ang_rps * dt
        if abs(yaw) <= max_a:
            yaw = 0.0
        else:
            yaw -= math.copysign(max_a, yaw)
        self.correction = (x, y, yaw)

    def note_cmd(self, t):
        """Record a base velocity command (any /cmd_vel* message)."""
        self.last_cmd_t = t

    def _parked(self, t):
        """True once the base has gone park_freeze_s with no command.

        Two one-time startup guards keep the clamp from pinning/absorbing
        Cartographer's INITIAL global lock (which can make large corrections for
        up to a few minutes against the loaded pbstream):
          - disabled until the first command ever arrives (a robot parked at its
            dock at boot converges with the clamp fully transparent), and
          - disabled for startup_grace_s after the first update, so even if you
            drive/teleop mid-convergence and then park, the clamp still holds off
            until the lock is comfortably done.
        Unlike scan_gate, missing these is not fatal (Cartographer is never
        starved; a mid-convergence pin self-heals on the next move) -- but the
        grace avoids a confusing "map stuck wrong" during bring-up.
        """
        if not self.park_freeze_s or self.last_cmd_t is None:
            return False
        if (self.startup_grace_s and self.start_t is not None
                and t - self.start_t < self.startup_grace_s):
            return False
        return t - self.last_cmd_t > self.park_freeze_s

    def update(self, carto, t):
        """Fold in the latest map_carto->odom; return the map->map_carto SE2.

        While PARKED, a step in map_carto->odom bigger than the jump thresholds
        is absorbed into the correction so the net map->odom does not move.
        While NOT parked, the correction washes out toward identity (see
        _washout) and Cartographer's changes flow straight through.
        """
        self.last_absorbed = None
        if self.start_t is None:
            self.start_t = t
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        if self.last_carto is None:
            self.last_carto = carto
            return self.correction
        # delta = the change Cartographer just applied to map_carto->odom
        # (independent of the accumulated correction).
        delta = se2_compose(se2_inverse(self.last_carto), carto)
        d_lin = math.hypot(delta[0], delta[1])
        d_ang = abs(delta[2])
        if self._parked(t):
            if d_lin > self.lin_jump_m or d_ang > self.ang_jump_rad:
                # Absorb: correction_new = correction . last_carto . carto^-1,
                # keeping map->odom = correction . carto constant across the step.
                self.correction = se2_compose(
                    se2_compose(self.correction, self.last_carto),
                    se2_inverse(carto))
                self.last_absorbed = (d_lin, d_ang)
        else:
            # Moving (or still in startup grace): give the deferred correction
            # back so localization re-converges to Cartographer's truth.
            self._washout(dt)
        self.last_carto = carto
        return self.correction


class MapOdomClampNode:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.map_carto_frame = rospy.get_param("~map_carto_frame", "map_carto")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.pub_rate_hz = rospy.get_param("~pub_rate_hz", 50.0)
        # Post-date the published map->map_carto stamp, exactly like
        # Cartographer's tf_publish_lookahead_sec does for map_carto->odom.
        # move_base/TEB looks up the map->odom CHAIN at a plan freshly stamped
        # now(); the chain is only as fresh as its stalest edge, so if this edge
        # sits at now() it lags Cartographer's post-dated edge by a few ms and
        # TEB gets "extrapolation into the future" (can't transform the global
        # plan -> stalls). Must be >= Cartographer's lookahead AND > our publish
        # period (1/pub_rate). Default 0.05 s matches the lua tf_publish_lookahead_sec.
        self.publish_lookahead_s = rospy.get_param("~publish_lookahead_s", 0.05)
        self.enabled = rospy.get_param("~enabled", True)
        self.core = ClampCore(
            park_freeze_s=rospy.get_param("~park_freeze_s", 30.0),
            lin_jump_m=rospy.get_param("~lin_jump_m", 0.05),
            ang_jump_rad=rospy.get_param("~ang_jump_rad", 0.03),
            startup_grace_s=rospy.get_param("~startup_grace_s", 300.0),
            washout_lin_mps=rospy.get_param("~washout_lin_mps", 0.15),
            washout_ang_rps=rospy.get_param("~washout_ang_rps", 0.15))
        self.stats_period_s = rospy.get_param("~stats_period_s", 30.0)
        self.lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.pub_absorb = rospy.Publisher("/map_odom_clamp/absorbing", Bool,
                                          queue_size=1, latch=True)
        self.pub_absorb.publish(Bool(False))
        # The accumulated map->map_carto correction (x, y, yaw). Publishing this
        # is what was missing when the 2026-07-24 off-target parking had to be
        # diagnosed from absorb WARNs alone -- always log the state you clamp.
        self.pub_corr = rospy.Publisher("/map_odom_clamp/correction", Vector3,
                                        queue_size=1, latch=True)
        self.pub_corr.publish(Vector3(0.0, 0.0, 0.0))

        rospy.Subscriber("/cmd_vel", Twist, self.cb_cmd, queue_size=10)
        rospy.Subscriber("/cmd_vel_teleop", Twist, self.cb_cmd, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / self.pub_rate_hz), self.cb_timer)
        rospy.Timer(rospy.Duration(self.stats_period_s), self.cb_stats)
        rospy.loginfo("map_odom_clamp: up (%s->%s, park_freeze=%.1fs "
                      "startup_grace=%.1fs lin_jump=%.3fm ang_jump=%.3frad "
                      "washout=%.2fm/s,%.2frad/s enabled=%s); correction starts "
                      "IDENTITY", self.map_frame, self.map_carto_frame,
                      self.core.park_freeze_s, self.core.startup_grace_s,
                      self.core.lin_jump_m, self.core.ang_jump_rad,
                      self.core.washout_lin_mps, self.core.washout_ang_rps,
                      self.enabled)

    def cb_stats(self, _evt):
        with self.lock:
            corr = self.core.correction
            d_lin, d_ang = self.core.correction_magnitude()
            parked = self.core._parked(rospy.Time.now().to_sec())
        self.pub_corr.publish(Vector3(corr[0], corr[1], corr[2]))
        rospy.loginfo("map_odom_clamp: %s | correction %.3f m / %.3f rad "
                      "(x=%.3f y=%.3f yaw=%.3f)",
                      "PARKED (absorbing)" if parked else "moving (washing out)",
                      d_lin, d_ang, corr[0], corr[1], corr[2])

    def cb_cmd(self, _msg):
        # Any command message means navigation is live -> not parked; see
        # scan_gate/map_odom_clamp docstrings on why jitter is not filtered.
        with self.lock:
            self.core.note_cmd(rospy.Time.now().to_sec())

    def _read_carto(self):
        """Latest map_carto->odom as (x, y, yaw), or None if unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_carto_frame, self.odom_frame, rospy.Time(0))
        except tf2_ros.TransformException:
            # Not available yet (Cartographer starting) or a transient gap.
            return None
        tr = tf.transform.translation
        rot = tf.transform.rotation
        return (tr.x, tr.y, yaw_from_quat(rot.x, rot.y, rot.z, rot.w))

    def cb_timer(self, _evt):
        now = rospy.Time.now()
        with self.lock:
            if self.enabled:
                carto = self._read_carto()
                if carto is not None:
                    self.core.update(carto, now.to_sec())
                    absorbed = self.core.last_absorbed
                else:
                    absorbed = None
                corr = self.core.correction
            else:
                corr = (0.0, 0.0, 0.0)
                absorbed = None
        if absorbed is not None:
            self.pub_absorb.publish(Bool(True))
            rospy.logwarn_throttle(
                2.0, "map_odom_clamp: absorbed parked map->odom jump "
                     "(%.2f m / %.3f rad) -- localization held steady",
                     absorbed[0], absorbed[1])

        t = TransformStamped()
        t.header.stamp = now + rospy.Duration(self.publish_lookahead_s)
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.map_carto_frame
        t.transform.translation.x = corr[0]
        t.transform.translation.y = corr[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(corr[2] / 2.0)
        t.transform.rotation.w = math.cos(corr[2] / 2.0)
        self.br.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("map_odom_clamp")
    MapOdomClampNode()
    rospy.spin()
