"""UI-independent iiwa14 differential-IK controller (kinematic mode).

MuJoCo is only the engine here: model, forward kinematics and Jacobians. No
physics is stepped yet, so q is integrated directly. A front end (iiwa_viser.py)
queues commands with `IKController.submit(name, *args)` from any thread and calls
`step()` from one loop. Nothing here depends on wall-clock time, so the
controller also runs headless (e.g. to benchmark the methods).

Adding an IK method: a solver in ik_solvers.py, an entry in METHODS, a branch in
`IKController._joint_velocity`. Adding a move: a path function and a MOVES entry.
"""

import queue
from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ik_solvers import (JointLimits, constrained_lsq_ik, damped_least_squares_ik,
                        joint_velocity_box, pseudoinverse_ik, velocity_scaling_ik)
from rotations import (mat_from_quat, quat_axis_angle, quat_conj, quat_from_axes,
                       quat_from_mat, quat_mul, rotvec_from_quat)

SCENE = Path(__file__).resolve().parent / "third_party/mujoco_menagerie/kuka_iiwa_14/scene.xml"
EE_SITE = "attachment_site"

DT = 0.01                         # control period [s]
KP_POS, KP_ROT = 5.0, 5.0         # target servo gains [1/s]
V_LIN_MAX, V_ANG_MAX = 0.5, 1.5   # max commanded EE speed [m/s], [rad/s]
K_ELBOW = 3.0                     # elbow-angle servo gain in the null space [1/s]
ELBOW_RATE_MAX = 1.0              # [rad/s]

# iiwa14 datasheet joint speeds; the accel limit is a choice for smooth motion.
QD_MAX = np.deg2rad([85, 85, 100, 75, 130, 135, 135])
QDD_MAX = np.full(7, 8.0)         # [rad/s^2]
JOINT_MARGIN = 0.05               # rad kept away from the hard joint limits
# Joint 4 = 0 is the stretched-arm singularity: the radial direction drops out of
# J, so "move straight back in" has no solution and a direction-preserving solver
# stalls there forever. Keep the elbow bent on whichever side it is (sign of J4).
ELBOW_MIN_BEND = 0.15             # rad

# Workspace: sphere around the shoulder (joint 2), radius = max EE reach.
WORKSPACE_MARGIN = 0.01           # m inside the true max reach
FLOOR_Z = 0.03                    # m, target never below this
BLINK_TIME = 0.6                  # s the workspace limit stays flagged after a violation

LIMIT_WARN_DEG = 3.0              # joint "at limit" (the QPs stop about here)
LIMIT_NEAR_DEG = 15.0             # joint "near limit"
MOVING_QD = 0.01                  # rad/s: any joint faster counts as moving
REACHED_MM, REACHED_DEG = 2.0, 1.0    # "At target" tolerance
HISTORY_TIME = 8.0                # s of samples kept for the plots

# Home (startup): tool down, wrist bent. Menagerie's "home" keyframe has J6 = 0,
# which aligns the axes of J5 and J7 (wrist singularity): its null space is then a
# useless wrist twist instead of the elbow swing.
HOME_Q = np.array([0.0, 0.45, 0.0, -1.95, 0.0, 0.7, 0.0])
# KUKA transport (shipping) position A1..A7 = 0, 25, 0, 90, 0, 0, 0 deg
# (Spez LBR iiwa V8, 03.05.2019, sec. 7.1.1, p. 75). KUKA axis Ai = model joint i.
TRANSPORT_Q = np.deg2rad([0.0, 25.0, 0.0, 90.0, 0.0, 0.0, 0.0])
POSES = {"home": HOME_Q, "transport": TRANSPORT_Q, "zero": np.zeros(7)}   # zero: singular

METHODS = [                       # (key, display name)
    ("pinv", "Pseudoinverse"),
    ("dls", "Damped least squares"),
    ("qp-lsq", "Constrained least squares"),
    ("qp-scaled", "Velocity scaling QP"),
]
LIMITED = {"qp-lsq", "qp-scaled"}     # methods that respect the joint limits


# -- moves: scripted target paths, all started from HOME_Q ------------------------
# path(ctl, p0, quat0, u, ease) -> target (pos, quat); u = 0..1 over MOVE_TIME,
# ease = the same with a smooth start and stop.
MOVE_TIME = 6.0                   # s; afterwards the target holds its last pose


def _stretch(ctl, p0, quat0, u, ease):
    """Horizontally away from the shoulder, past the reach sphere, and back. The
    workspace clamp holds the target on the sphere: arm fully stretched (singular)."""
    d = p0 - ctl.ws_center
    d[2] = 0.0
    d /= np.linalg.norm(d)
    return p0 + 0.7 * np.sin(np.pi * u) ** 2 * d, quat0


def _swing(ctl, p0, quat0, u, ease):
    """230 deg around the base (world Z): runs into the joint 1 limit."""
    a = np.deg2rad(230.0) * ease
    c, s = np.cos(a), np.sin(a)
    pos = np.r_[c * p0[0] - s * p0[1], s * p0[0] + c * p0[1], p0[2]]
    return pos, quat_mul(quat_axis_angle([0, 0, 1], a), quat0)


def _spin(ctl, p0, quat0, u, ease):
    """330 deg about the tool's own Z axis: runs into the joint 7 limit."""
    return p0, quat_mul(quat0, quat_axis_angle([0, 0, 1], np.deg2rad(330.0) * ease))


MOVES = {"stretch": ("Stretch out", _stretch),
         "swing": ("Swing around", _swing),
         "spin": ("Spin tool", _spin)}


@dataclass
class RunningMove:
    key: str
    p0: np.ndarray
    quat0: np.ndarray
    t: float = 0.0


# -- kinematics ------------------------------------------------------------------
class Kinematics:
    """Forward kinematics + EE Jacobian on a private MjData. Each call overwrites
    self.d, so read body positions from it right after a call for that q."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(EE_SITE).id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def __call__(self, q):
        """-> EE position, EE quaternion, 6x7 world-frame Jacobian [linear; angular]."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
        return (self.d.site_xpos[self.site].copy(), quat_from_mat(self.d.site_xmat[self.site]),
                np.vstack([self.jacp, self.jacr]))


def pose_error(pos, quat, pos_d, quat_d):
    """6-D error [p_d - p; rotation vector of R_d R^T], both in the world frame."""
    q_err = quat_mul(quat_d, quat_conj(quat))
    if q_err[0] < 0:              # q and -q are the same rotation: take the short way
        q_err = -q_err
    return np.r_[pos_d - pos, rotvec_from_quat(q_err)]


def servo_twist(err):
    """Pose error -> commanded EE twist. Linear and angular parts are clipped in
    magnitude only, so the direction toward the target is kept."""
    v, w = KP_POS * err[:3], KP_ROT * err[3:]
    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv > V_LIN_MAX:
        v *= V_LIN_MAX / nv
    if nw > V_ANG_MAX:
        w *= V_ANG_MAX / nw
    return np.r_[v, w]


def retwist_about_z(quat_target, quat_ee):
    """Orientation with the target's Z axis but the EE's spin about it."""
    z = mat_from_quat(quat_target)[:, 2]
    Re = mat_from_quat(quat_ee)
    x = Re[:, 0] - (Re[:, 0] @ z) * z
    if np.linalg.norm(x) < 1e-6:          # EE x parallel to target z: use EE y
        x = Re[:, 1] - (Re[:, 1] @ z) * z
    x /= np.linalg.norm(x)
    return quat_from_axes(x, np.cross(z, x), z)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# Plot sample per step: alpha (speed scale), sigma_min(J), qd / qd_max (7,).
Sample = namedtuple("Sample", "t alpha sigma_min qd_frac")


@dataclass
class Status:
    state: str = "At target"      # "Moving" / "At target" / "Can't reach target"
    pos_err_mm: float = 0.0
    rot_err_deg: float = 0.0
    speed: float = 1.0            # alpha of the velocity scaling QP (1 = full speed)
    sigma_min: float = 0.0
    elbow_deg: float = 0.0        # actual elbow (arm) angle
    warnings: list = field(default_factory=list)   # e.g. "J4 at limit", solver notes


class IKController:
    def __init__(self, method="qp-scaled", flexible=True):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.kin = Kinematics(self.m)
        r = self.m.jnt_range
        self.q_min_hard, self.q_max_hard = r[:, 0].copy(), r[:, 1].copy()
        self.lim = JointLimits(r[:, 0] + JOINT_MARGIN, r[:, 1] - JOINT_MARGIN, QD_MAX, QDD_MAX)

        # Workspace sphere: centre at the shoulder, radius = EE distance with the
        # arm as straight as the elbow limit allows.
        q_stretch = np.zeros(7)
        q_stretch[3] = -ELBOW_MIN_BEND
        pos_stretch, _, _ = self.kin(q_stretch)
        self.ws_center = self.kin.d.xpos[self.m.body("link2").id].copy()
        self.ws_radius = np.linalg.norm(pos_stretch - self.ws_center) - WORKSPACE_MARGIN
        self.body_shoulder, self.body_elbow, self.body_wrist = (
            self.m.body(n).id for n in ("link2", "link4", "link6"))

        self.method_key = method
        self.flexible = flexible          # soft rows: priority part exact, the rest gives way
        self.z_flexible = False           # tool may spin freely about its own Z axis
        self.priority = "position"        # last drag: move -> position, rotate -> orientation
        self.t = 0.0                      # simulated time [s]
        self.blink_until = -1.0
        self.move = None                  # RunningMove while a move plays
        self.trail = deque(maxlen=400)
        self.history = deque(maxlen=int(HISTORY_TIME / DT))
        self.commands = queue.SimpleQueue()
        self.reset()

    @property
    def workspace_violation(self):
        return self.t < self.blink_until

    @property
    def move_progress(self):
        """(display name, fraction done) of the playing move, or None."""
        if self.move is None:
            return None
        return MOVES[self.move.key][0], min(self.move.t / MOVE_TIME, 1.0)

    def joint_margins_deg(self, q=None):
        """Distance of each joint to its nearest hard limit [deg]; negative = beyond."""
        q = self.q if q is None else q
        return np.rad2deg(np.minimum(q - self.q_min_hard, self.q_max_hard - q))

    def limit_levels(self):
        """Per joint: 0 = fine, 1 = near its limit, 2 = at or beyond it."""
        m = self.joint_margins_deg()
        return np.where(m < LIMIT_WARN_DEG, 2, np.where(m < LIMIT_NEAR_DEG, 1, 0))

    # -- commands: thread-safe entry point for any UI ----------------------------
    def submit(self, name, *args):
        """Queue a call of method `name`; applied at the start of the next step()."""
        self.commands.put((name, args))

    def _apply_commands(self):
        while True:
            try:
                name, args = self.commands.get_nowait()
            except queue.Empty:
                return
            getattr(self, name)(*args)

    # -- state changes (from UI threads: through submit) --------------------------
    def reset(self, pose="home"):
        """Jump to a pose from POSES; clears the trail and stops a playing move."""
        self.set_joints(POSES[pose])
        self.trail.clear()
        self.alpha, self.note = 1.0, ""
        self.status = Status()

    def set_joints(self, q):
        """Jump to a joint configuration (FK mode); the target follows the EE."""
        self.move = None
        self.q = np.clip(np.asarray(q, dtype=float), self.q_min_hard, self.q_max_hard)
        self.qd = np.zeros(7)
        self._set_elbow_side()
        self.target_pos, self.target_quat, _ = self.kin(self.q)
        self._init_elbow()

    def set_method(self, key):
        self.method_key = key
        self.qd[:] = 0
        self._set_elbow_side()        # the unlimited methods may have flipped the elbow
        self.trail.clear()

    def set_flexible(self, on):
        """Reach flexibility; mutually exclusive with set_z_flexible."""
        self.flexible = bool(on)
        if self.flexible:
            self.z_flexible = False

    def set_z_flexible(self, on):
        self.z_flexible = bool(on)
        if self.z_flexible:
            self.flexible = False

    def set_target(self, pos, quat, mode=None):
        """New target from the UI (stops a playing move). mode "position" /
        "orientation" = what the user is dragging, which sets the task priority."""
        self.move = None
        self.target_pos = np.asarray(pos, dtype=float).copy()
        q = np.asarray(quat, dtype=float)
        self.target_quat = q / np.linalg.norm(q)
        if mode in ("position", "orientation"):
            self.priority = mode

    def set_elbow(self, deg):
        """Elbow-angle setpoint; the velocity scaling QP swings the elbow there in
        the null space while the EE stays put."""
        self.elbow_target = np.deg2rad(float(deg))

    def start_move(self, key):
        """Reset to Home and play the scripted target path MOVES[key]."""
        self.reset("home")
        self.priority = "position"
        self.move = RunningMove(key, self.target_pos.copy(), self.target_quat.copy())

    def _set_elbow_side(self):
        """Joint-4 soft limit: |J4| >= ELBOW_MIN_BEND on the side the elbow is bent
        to now (J4 = 0 counts as the negative side)."""
        lo_hard, hi_hard = self.q_min_hard[3] + JOINT_MARGIN, self.q_max_hard[3] - JOINT_MARGIN
        if self.q[3] > 0:
            self.lim.q_min[3], self.lim.q_max[3] = ELBOW_MIN_BEND, hi_hard
        else:
            self.lim.q_min[3], self.lim.q_max[3] = lo_hard, -ELBOW_MIN_BEND

    # -- elbow (arm) angle: the iiwa's 1-D self-motion ---------------------------
    def elbow_geometry(self, q):
        """Shoulder, elbow, wrist positions, unit shoulder->wrist axis u and the
        zero-angle reference ref (unit, perpendicular to u, as close to world up
        as possible)."""
        self.kin(q)
        x = self.kin.d.xpos
        s, e, w = (x[b].copy() for b in (self.body_shoulder, self.body_elbow, self.body_wrist))
        u = (w - s) / np.linalg.norm(w - s)
        ref = np.r_[0.0, 0.0, 1.0] - u[2] * u
        if np.linalg.norm(ref) < 1e-3:            # shoulder-wrist line vertical
            ref = np.r_[1.0, 0.0, 0.0] - u[0] * u
        return s, e, w, u, ref / np.linalg.norm(ref)

    def elbow_angle(self, q):
        """Rotation of the elbow about the shoulder->wrist line [rad], 0 = elbow up,
        sign by the right-hand rule. Changing it moves only the elbow."""
        s, e, w, u, ref = self.elbow_geometry(q)
        pe = e - s - ((e - s) @ u) * u
        return float(np.arctan2(np.cross(ref, pe) @ u, ref @ pe))

    def _init_elbow(self):
        self._elbow_raw = self.elbow_angle(self.q)
        self.elbow = self._elbow_raw          # unwrapped (continuous) elbow angle
        self.elbow_target = self.elbow

    def _track_elbow(self):
        """Unwrap: passing +-180 deg must not flip the sign of the elbow error."""
        raw = self.elbow_angle(self.q)
        self.elbow += wrap(raw - self._elbow_raw)
        self._elbow_raw = raw

    def elbow_null_velocity(self, J):
        """Joint velocity along the null space of J that drives the elbow angle to
        its setpoint (the elbow-angle rate per unit null motion is found by finite
        differences)."""
        n = np.linalg.svd(J)[2][-1]               # unit null vector (7-DOF, 6-D task)
        eps = 1e-4
        dpsi_dn = wrap(self.elbow_angle(self.q + eps * n) - self.elbow_angle(self.q)) / eps
        if abs(dpsi_dn) < 1e-3:
            return np.zeros(7)
        # Error against the unwrapped angle: travel through 0 like the slider does
        # (the wrapped "short way" often runs into joint limits).
        rate = np.clip(K_ELBOW * (self.elbow_target - self.elbow), -ELBOW_RATE_MAX, ELBOW_RATE_MAX)
        return (rate / dpsi_dn) * n

    # -- control step --------------------------------------------------------------
    def _shape_target(self, ee_pos, ee_quat):
        """Apply the task priority and the workspace limit to the target.

        * Flexible: the non-priority part of the target copies the EE, so its
          desired velocity is "hold still". The solver lets it give way only when
          the priority part needs it, and the target shows the pose actually
          reached (e.g. the tilted wrist).
        * The position is clamped into the reach sphere and above the floor; a
          clamp flags the workspace limit.
        """
        if self.flexible and self.priority == "position":
            self.target_quat = ee_quat.copy()
        elif self.flexible:
            self.target_pos = ee_pos.copy()
        if self.z_flexible:
            self.target_quat = retwist_about_z(self.target_quat, ee_quat)

        p = self.target_pos.copy()
        r = p - self.ws_center
        dist = np.linalg.norm(r)
        clamped = dist > self.ws_radius or p[2] < FLOOR_Z
        if dist > self.ws_radius:
            p = self.ws_center + r * (self.ws_radius / dist)
        p[2] = max(p[2], FLOOR_Z)
        if clamped:
            self.target_pos = p
            self.blink_until = self.t + BLINK_TIME

    def _soft_rows(self):
        """Task rows (tool frame: vx vy vz wx wy wz) that may give way."""
        soft = set()
        if self.flexible:
            soft |= {3, 4, 5} if self.priority == "position" else {0, 1, 2}
        if self.z_flexible:
            soft.add(5)                           # spin about the tool's own Z
        return sorted(soft) or None

    def _joint_velocity(self, J, V, quat):
        """Velocity-level methods: -> (qd, alpha, note)."""
        if self.method_key == "pinv":
            res = pseudoinverse_ik(J, V)
            return res.qd, res.alpha, res.note

        # A hand-set pose (e.g. zero, J4 = 0) may be outside the *soft* range:
        # stretch the soft box to include q, so the robot holds there and can only
        # move back inward. That keeps qd = 0 feasible without snapping the robot
        # away from where it was put.
        lim = JointLimits(np.minimum(self.lim.q_min, self.q), np.maximum(self.lim.q_max, self.q),
                          self.lim.qd_max, self.lim.qdd_max)
        if self.method_key == "qp-scaled":
            qd_elbow = self.elbow_null_velocity(J)
            # Angular rows in the tool frame, so soft row 5 = spin about the tool's Z.
            T = np.eye(6)
            T[3:, 3:] = mat_from_quat(quat).T
            Jt, Vt = T @ J, T @ V

        for use_accel in (True, False):           # retry without the accel limit if infeasible
            lo, hi = joint_velocity_box(self.q, self.qd, DT, lim, use_accel)
            if self.method_key == "qp-lsq":
                res = constrained_lsq_ik(J, V, lo, hi)
            else:
                res = velocity_scaling_ik(Jt, Vt, lo, hi, qd_elbow, soft_rows=self._soft_rows())
            if res.ok:
                break
        if not res.ok and np.min(self.joint_margins_deg()) < LIMIT_WARN_DEG:
            # A joint on its limit blocks the commanded direction: the optimum is
            # "stop" (qd = 0, alpha = 0), which OSQP only creeps toward in this
            # degenerate case. The returned stop is right, and the status already
            # names the joint, so don't report a solver failure.
            res.note = ""
        return res.qd, res.alpha, res.note

    def step(self):
        self._apply_commands()
        self.t += DT
        if self.move is not None:
            mv = self.move
            mv.t += DT
            u = min(mv.t / MOVE_TIME, 1.0)
            self.target_pos, self.target_quat = MOVES[mv.key][1](
                self, mv.p0.copy(), mv.quat0.copy(), u, 0.5 - 0.5 * np.cos(np.pi * u))
            if mv.t >= MOVE_TIME:
                self.move = None

        if self.method_key in LIMITED:
            # The unlimited methods can leave q outside the joint range; a real
            # robot can't be there.
            self.q = np.clip(self.q, self.q_min_hard, self.q_max_hard)
        pos, quat, J = self.kin(self.q)
        self._shape_target(pos, quat)
        err = pose_error(pos, quat, self.target_pos, self.target_quat)

        if self.method_key == "dls":              # position level: solves the whole pose
            q_new, ok = damped_least_squares_ik(self.kin, self.q, self.target_pos.copy(),
                                                self.target_quat.copy(), pose_error)
            self.qd, self.alpha, self.note = (q_new - self.q) / DT, 1.0, "" if ok else "not converged"
            self.q = q_new
        else:
            self.qd, self.alpha, self.note = self._joint_velocity(J, servo_twist(err), quat)
            self.q = self.q + self.qd * DT

        if not np.all(np.isfinite(self.q)) or np.max(np.abs(self.q)) > 100:
            self.reset()
            self.status.warnings = ["joints blew up, reset"]
            return

        self._track_elbow()
        if self.method_key != "qp-scaled":
            self.elbow_target = self.elbow        # only the QP steers it; no jump on switch
        self.trail.append(pos)
        sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        self.history.append(Sample(self.t, float(self.alpha), sigma_min, self.qd / self.lim.qd_max))
        self.status = self._status(err, sigma_min)

    def _status(self, err, sigma_min):
        pos_mm, rot_deg = 1000 * float(np.linalg.norm(err[:3])), float(np.rad2deg(np.linalg.norm(err[3:])))
        moving = float(np.max(np.abs(self.qd))) > MOVING_QD
        if self.workspace_violation or (not moving and (pos_mm > REACHED_MM or rot_deg > REACHED_DEG)):
            state = "Can't reach target"          # outside reach, or stuck short of it
        else:
            state = "Moving" if moving else "At target"
        warnings = [f"J{i + 1} beyond limit" if m < 0 else f"J{i + 1} at limit"
                    for i, m in enumerate(self.joint_margins_deg()) if m < LIMIT_WARN_DEG]
        if self.note:
            warnings.append(self.note)
        return Status(state, pos_mm, rot_deg, float(self.alpha), sigma_min,
                      float(np.rad2deg(self.elbow)), warnings)
