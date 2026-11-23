"""Control mode: the iiwa as a dynamic system driven by joint torques.

IK mode (iiwa_core) integrates joint velocities directly. Here MuJoCo steps the
real rigid-body dynamics, M(q) q'' + C(q, q') q' + g(q) = tau, and a joint-torque
controller has to make the arm follow. The chain is

    target -> IK method (iiwa_core) -> reference q_r, q_r', q_r'' -> controller -> tau -> physics

so the IK method still shapes the motion, and the controller decides how well the
real arm tracks it. The task impedance controller is the exception: it pulls the
tool toward the target directly, without IK.

Two models: the *plant* (what is simulated) and the *nominal* model the controller
uses for g(q), M(q) and C. The payload is added to the plant only, so a controller
that relies on the model shows the error an unmodelled load causes.

Adding a controller: a function tau = f(model, q, qd, ref, gains) and a
CONTROLLERS entry.
"""

import queue
from collections import deque, namedtuple
from dataclasses import dataclass, field

import mujoco
import numpy as np

from iiwa_core import (DT, EE_SITE, HISTORY_TIME, HOME_Q, LIMIT_WARN_DEG, MOVING_QD, QD_MAX,
                       REACHED_DEG, REACHED_MM, SCENE, Kinematics, pose_error)

# KUKA LBR iiwa 14 R820 datasheet: max joint torques [N m].
TAU_MAX = np.array([320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0])
PHYS_DT = 0.001                   # physics step = controller period (1 kHz, like KUKA FRI)
SUBSTEPS = round(DT / PHYS_DT)    # physics steps per app step
GRAVITY = 9.81
OMEGA_DEFAULT = 15.0              # closed-loop bandwidth [rad/s]
PAYLOAD_MAX = 14.0                # iiwa 14 rated payload [kg]
STEEL = 7800.0                    # payload drawn and simulated as a steel ball [kg/m^3]
PUSH_FORCE = 60.0                 # N, sideways on the tool
PUSH_TIME = 0.25                  # s
TASK_DAMPING = 1e-3               # regularises the tool's apparent inertia near singularities
NULL_SCALE = 0.3                  # null-space posture gains relative to the joint gains

CONTROLLERS = [                   # (key, display name)
    ("pd", "Joint PD"),
    ("pd-gravity", "PD + gravity"),
    ("computed-torque", "Computed torque"),
    ("impedance", "Task impedance"),
]


def build_model():
    """The scene with torque motors and a payload body (mass 0) at the flange.

    Menagerie's actuators are position servos (tau = 2000 (u - q) - 200 q'), which
    hide the torque. Gain 1 and no bias make ctrl the joint torque itself."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    site = spec.site(EE_SITE)
    body = site.parent.add_body(name="payload", pos=site.pos)
    body.explicitinertial = True
    body.mass, body.inertia = 0.0, [0.0, 0.0, 0.0]
    m = spec.compile()
    m.opt.timestep = PHYS_DT
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_biasprm[:] = 0.0
    m.actuator_ctrllimited[:] = 0     # saturation is done (and reported) by ControlSim
    return m


class ModelTerms:
    """Dynamics terms of a model at (q, q'), on a private MjData: M (7x7),
    bias = C q' + g, gravity g, and the tool pose and Jacobian."""

    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.M = np.zeros((model.nv, model.nv))
        self.kin = Kinematics(model)

    def update(self, q, qd):
        m, d = self.m, self.d
        d.qpos[:], d.qvel[:] = q, 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)               # composite rigid body: the mass matrix
        mujoco.mj_fullM(m, d, self.M)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)   # q' = 0: gravity only
        self.g = d.qfrc_bias.copy()
        d.qvel[:] = qd
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)   # C q' + g
        self.bias = d.qfrc_bias.copy()
        self.pos, self.quat, self.J = self.kin(q)


Reference = namedtuple("Reference", "q qd qdd pos quat")


@dataclass
class Gains:
    """All controller gains from one bandwidth w (critically damped):
    joint PD Kp = w^2 M_ii, Kd = 2 w M_ii with M_ii the diagonal of M at Home, so
    every joint responds about equally fast; computed torque uses w^2 and 2 w
    directly (M is in the law), the impedance the same on the tool error."""
    kp: np.ndarray
    kd: np.ndarray
    w: float

    @classmethod
    def from_omega(cls, w, m_diag):
        return cls(w * w * m_diag, 2 * w * m_diag, w)


# -- controllers: tau = f(model terms at (q, qd), q, qd, reference, gains) ---------------
def pd(mt, q, qd, r, k):
    """Springs and dampers on the joint errors. Gravity pulls the arm down until
    the spring force balances it: a steady sag."""
    return k.kp * (r.q - q) + k.kd * (r.qd - qd)


def pd_gravity(mt, q, qd, r, k):
    """PD plus the model's gravity torque: no sag at rest (if the model is right),
    but the inertia and Coriolis forces of a fast move still make it lag."""
    return pd(mt, q, qd, r, k) + mt.g


def computed_torque(mt, q, qd, r, k):
    """Inverse dynamics: choose the joint acceleration (reference acceleration plus
    PD on the error), then the torque that produces it. The error then obeys
    e'' + 2w e' + w^2 e = 0 on every joint, whatever the pose or speed."""
    a = r.qdd + k.w * k.w * (r.q - q) + 2 * k.w * (r.qd - qd)
    return mt.M @ a + mt.bias


def impedance(mt, q, qd, r, k):
    """Virtual spring and damper between tool and target, mapped to joint torques
    by J^T (no IK). Operational-space form: the spring and damper act through the
    tool's apparent inertia Lambda = (J M^-1 J^T)^-1, so the tool error obeys
    e'' + 2w e' + w^2 e = 0 in every direction (a fixed spring would be far too
    stiff for the light wrist). Near a singularity Lambda is regularised by
    TASK_DAMPING. The 7th-axis motion left over is held near the IK posture through
    the dynamically consistent null-space projector, so it doesn't move the tool."""
    J = mt.J
    err = pose_error(mt.pos, mt.quat, r.pos, r.quat)
    Minv_JT = np.linalg.solve(mt.M, J.T)
    lam = np.linalg.inv(J @ Minv_JT + TASK_DAMPING * np.eye(6))
    wrench = lam @ (k.w * k.w * err - 2 * k.w * (J @ qd))
    posture = NULL_SCALE * (k.kp * (r.q - q) - k.kd * qd)
    null = np.eye(7) - J.T @ (lam @ Minv_JT.T)
    return J.T @ wrench + null @ posture + mt.g


CONTROL_LAWS = {"pd": pd, "pd-gravity": pd_gravity, "computed-torque": computed_torque,
                "impedance": impedance}
assert CONTROL_LAWS.keys() == {k for k, _ in CONTROLLERS}

# Plot sample per step: torque / TAU_MAX (7,), tool errors.
ControlSample = namedtuple("ControlSample", "t tau_frac pos_err_mm rot_err_deg")


@dataclass
class ControlStatus:
    state: str = "At target"      # also "Off target": settled, but with an error
    pos_err_mm: float = 0.0
    rot_err_deg: float = 0.0
    track_err_deg: float = 0.0    # max joint error to the IK reference
    peak_joint: int = 0
    peak_frac: float = 0.0        # |tau| / TAU_MAX of the most loaded joint
    sigma_min: float = 0.0
    warnings: list = field(default_factory=list)


class ControlSim:
    """Plant + torque controller tracking an IKController (the reference
    generator). Same command pattern: submit() from any thread, step() from the
    loop, right after ik.step()."""

    def __init__(self, ik):
        self.ik = ik
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.nominal = ModelTerms(build_model())  # the controller's model: no payload
        self.nominal.update(HOME_Q, np.zeros(7))
        self.m_diag = np.diag(self.nominal.M).copy()
        self.kin = Kinematics(self.m)
        self.payload_id = self.m.body("payload").id
        self.flange_id = self.m.body("link7").id

        self.active = False
        self.controller = "pd-gravity"
        self.gains = Gains.from_omega(OMEGA_DEFAULT, self.m_diag)
        self.gravity = True
        self.torque_limits = True
        self.payload = 0.0
        self.push_until, self.push_force = -1.0, np.zeros(3)
        self.t = 0.0
        self.tau = np.zeros(7)
        self.saturated = np.zeros(7, bool)
        self.trail = deque(maxlen=400)
        self.history = deque(maxlen=int(HISTORY_TIME / DT))
        self.status = ControlStatus()
        self.commands = queue.SimpleQueue()
        self.sync()

    @property
    def q(self):
        return self.d.qpos.copy()

    @property
    def pushing(self):
        return self.t < self.push_until

    @property
    def payload_radius(self):
        return (3 * self.payload / (4 * np.pi * STEEL)) ** (1 / 3)

    # -- commands -------------------------------------------------------------------
    def submit(self, name, *args):
        self.commands.put((name, args))

    def _apply_commands(self):
        while True:
            try:
                name, args = self.commands.get_nowait()
            except queue.Empty:
                return
            getattr(self, name)(*args)

    def sync(self):
        """Put the plant at the reference, at rest (poses, moves, entering the mode)."""
        self.d.qpos[:], self.d.qvel[:] = self.ik.q, 0.0
        self.d.xfrc_applied[:] = 0.0
        self.push_until = -1.0
        mujoco.mj_forward(self.m, self.d)
        self.ref_prev, self.qd_ref = self.ik.q.copy(), np.zeros(7)
        self.trail.clear()

    def set_active(self, on):
        if on and not self.active:
            self.sync()
        elif not on and self.active:
            self.ik.set_joints(self.q)    # IK mode continues from where the arm is
        self.active = on

    def set_controller(self, key):
        if self.controller == "impedance" and key != "impedance":
            # The IK reference ran on its own meanwhile: restart it at the arm.
            self.ik.set_joints(self.q)
        self.controller = key

    def set_omega(self, w):
        self.gains = Gains.from_omega(float(w), self.m_diag)

    def set_gravity(self, on):
        """Gravity of the world: plant and controller model alike."""
        self.gravity = on
        for m in (self.m, self.nominal.m):
            m.opt.gravity[:] = (0.0, 0.0, -GRAVITY if on else 0.0)

    def set_torque_limits(self, on):
        self.torque_limits = on

    def set_payload(self, kg):
        """Steel ball at the flange, in the plant only (the controller doesn't know)."""
        self.payload = float(kg)
        r = self.payload_radius
        self.m.body_mass[self.payload_id] = self.payload
        self.m.body_ipos[self.payload_id] = (0.0, 0.0, r)
        self.m.body_inertia[self.payload_id] = 0.4 * self.payload * r * r

    def push(self):
        """Sideways shove on the tool: horizontal, perpendicular to the arm's reach."""
        pos = self.d.xpos[self.flange_id]
        side = np.cross([0.0, 0.0, 1.0], [pos[0], pos[1], 0.0])
        n = np.linalg.norm(side)
        self.push_force = PUSH_FORCE * (side / n if n > 1e-6 else np.r_[0.0, 1.0, 0.0])
        self.push_until = self.t + PUSH_TIME

    # -- step -------------------------------------------------------------------------
    def step(self):
        self._apply_commands()
        q1 = self.ik.q.copy()
        if not self.active:
            self.ref_prev = q1
            return
        q0 = self.ref_prev
        # A reference jump (joint slider) is a step input: no velocity feedforward.
        jump = np.any(np.abs(q1 - q0) > 2 * QD_MAX * DT)
        qd_r = np.zeros(7) if jump else (q1 - q0) / DT
        qdd_r = np.zeros(7) if jump else (qd_r - self.qd_ref) / DT
        self.ref_prev, self.qd_ref = q1, qd_r

        law = CONTROL_LAWS[self.controller]
        self.saturated[:] = False
        for k in range(SUBSTEPS):
            # Reference interpolated over the 10 ms app step, so the 1 kHz loop sees
            # a smooth ramp instead of a 100 Hz staircase.
            q_r = q1 if jump else q0 + (k + 1) / SUBSTEPS * (q1 - q0)
            ref = Reference(q_r, qd_r, qdd_r, self.ik.target_pos, self.ik.target_quat)
            q, qd = self.d.qpos.copy(), self.d.qvel.copy()
            self.nominal.update(q, qd)
            tau = law(self.nominal, q, qd, ref, self.gains)
            if self.torque_limits:
                self.saturated |= np.abs(tau) > TAU_MAX
                tau = np.clip(tau, -TAU_MAX, TAU_MAX)
            self.d.ctrl[:] = tau
            self.d.xfrc_applied[self.flange_id, :3] = self.push_force if self.pushing else 0.0
            mujoco.mj_step(self.m, self.d)
            self.t += PHYS_DT
        self.tau = tau

        if not np.all(np.isfinite(self.d.qpos)):
            self.sync()
            self.status.warnings = ["simulation blew up, reset"]
            return
        self._record(q1)

    def _record(self, q_ref):
        pos, quat, J = self.kin(self.q)
        err = pose_error(pos, quat, self.ik.target_pos, self.ik.target_quat)
        pos_mm, rot_deg = 1000 * float(np.linalg.norm(err[:3])), float(np.rad2deg(np.linalg.norm(err[3:])))
        frac = self.tau / TAU_MAX
        self.trail.append(pos)
        self.history.append(ControlSample(self.t, frac.copy(), pos_mm, rot_deg))

        if self.ik.status.state == "Can't reach target":
            state = "Can't reach target"
        elif float(np.max(np.abs(self.d.qvel))) > MOVING_QD:
            state = "Moving"
        elif pos_mm > REACHED_MM or rot_deg > REACHED_DEG:
            state = "Off target"
        else:
            state = "At target"
        warnings = [f"J{i + 1} torque limit" for i in np.flatnonzero(self.saturated)]
        warnings += [f"J{i + 1} at limit" for i, m in enumerate(self.ik.joint_margins_deg(self.q))
                     if m < LIMIT_WARN_DEG]
        if self.pushing:
            warnings.append("pushed")
        j = int(np.argmax(np.abs(frac)))
        self.status = ControlStatus(
            state, pos_mm, rot_deg, float(np.rad2deg(np.max(np.abs(q_ref - self.q)))),
            j, float(abs(frac[j])), float(np.linalg.svd(J, compute_uv=False)[-1]), warnings)
