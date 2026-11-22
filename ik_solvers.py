"""Differential IK solvers for the KUKA iiwa14, from naive to robust.

The progression mirrors the video "Jacobian inverse kinematics. How to do it
right", rebuilt for a 7-DOF iiwa in MuJoCo. Every solver answers the same
question:

    Given the end-effector (EE) spatial velocity we want, V = [v; w] (6,),
    which joint velocity qd (7,) should we command this tick?

The kinematic relation is V = J(q) @ qd, with J the 6x7 geometric Jacobian
(world frame, linear rows first, angular rows second).

  pseudoinverse_ik          qd = J^+ V
  damped_least_squares_ik   q += J^T (J J^T + lambda^2 I)^-1 e, iterated
  constrained_lsq_ik        min |J qd - V|^2  s.t. joint limits
  velocity_scaling_ik       max alpha  s.t. J qd = alpha V, joint limits
"""

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse


@dataclass
class JointLimits:
    q_min: np.ndarray
    q_max: np.ndarray
    qd_max: np.ndarray
    qdd_max: np.ndarray


@dataclass
class IKResult:
    qd: np.ndarray
    alpha: float = 1.0
    ok: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# Pseudoinverse
# ---------------------------------------------------------------------------
def pseudoinverse_ik(J: np.ndarray, V: np.ndarray) -> IKResult:
    """qd = J^+ V.

    The most direct inversion of V = J qd. With 7 joints and a 6-D task, J^+
    gives the minimum-norm qd that reproduces V exactly.

    Why it fails: J^+ = sum_i (1/sigma_i) v_i u_i^T. Near a singularity
    (arm stretched out, or joints 2/4/6 at zero aligning axes), the smallest
    singular value sigma_min -> 0, so 1/sigma_min explodes. A tiny V component
    along that direction becomes a huge joint velocity, and the arm "jumps
    off" its configuration. Nothing bounds velocity or joint range either.
    """
    return IKResult(qd=np.linalg.pinv(J) @ V)


# ---------------------------------------------------------------------------
# Damped least squares (Levenberg-Marquardt), iterated to convergence
# ---------------------------------------------------------------------------
def damped_least_squares_ik(fk_jac, q0: np.ndarray, pos_d: np.ndarray, quat_d: np.ndarray,
                            pose_error, damping: float = 0.1, iters: int = 200,
                            tol: float = 1e-3) -> tuple[np.ndarray, bool]:
    """Position-level IK: repeat q += J^T (J J^T + lambda^2 I)^-1 e until |e| < tol.

    DLS minimises |J dq - e|^2 + lambda^2 |dq|^2. In SVD terms each gain
    1/sigma_i becomes sigma_i / (sigma_i^2 + lambda^2), which stays bounded
    (max 1/(2 lambda)) instead of exploding. That fixes the pseudoinverse blow-up.

    Why it is still jumpy and inexact:
      * Damping trades accuracy for stability. Near singularities the solution
        is deliberately wrong (it under-shoots), so the EE lags or misses the
        target.
      * It solves the whole pose IK every frame with no limit on how far q
        moves in one call. When the target is far or unreachable, the
        iteration can land on a quite different configuration (or wander
        for all its iterations), so q teleports between frames.
      * No velocity, acceleration or joint-range limits.
    (In the video's version, `success = True` sits after `break`, so it never
    reports success. Here we return the flag properly.)
    """
    q = q0.copy()
    for _ in range(iters):
        pos, quat, J = fk_jac(q)
        e = pose_error(pos, quat, pos_d, quat_d)
        if np.linalg.norm(e) < tol:
            return q, True
        q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
    return q, False


# ---------------------------------------------------------------------------
# helpers shared by the optimisation-based methods
# ---------------------------------------------------------------------------
def joint_velocity_box(q, qd_prev, dt, lim: JointLimits, use_accel: bool = True):
    """Turn velocity, position and acceleration limits into one box lo <= qd <= hi.

      |qd| <= qd_max
      q_min <= q + qd dt <= q_max          ->  (q_min - q)/dt <= qd <= (q_max - q)/dt
      |qd - qd_prev| / dt <= qdd_max       ->  qd_prev -/+ qdd_max dt

    If the acceleration box conflicts with the others (e.g. moving fast toward
    a joint limit), drop acceleration for that joint. Position safety wins.
    use_accel=False gives the fallback box used when the accel-limited QP is
    infeasible (the box can exclude qd = 0, so "stop" may not be allowed).
    Callers must keep q inside [q_min, q_max]; then qd = 0 is always in the
    fallback box and the QPs below are always feasible.
    """
    lo_vp = np.maximum(-lim.qd_max, (lim.q_min - q) / dt)
    hi_vp = np.minimum(lim.qd_max, (lim.q_max - q) / dt)
    a = lim.qdd_max * dt if use_accel else np.inf
    lo = np.maximum(lo_vp, qd_prev - a)
    hi = np.minimum(hi_vp, qd_prev + a)
    bad = lo > hi
    lo[bad], hi[bad] = lo_vp[bad], hi_vp[bad]
    return lo, hi


def _solve_qp(P, c, A, l, u):
    prob = osqp.OSQP()
    prob.setup(sparse.csc_matrix(P), c, sparse.csc_matrix(A), l, u,
               verbose=False, eps_abs=1e-7, eps_rel=1e-7, polishing=False,
               max_iter=10000)
    res = prob.solve()
    ok = res.info.status in ("solved", "solved inaccurate") and np.all(np.isfinite(res.x))
    return res.x, ok


# ---------------------------------------------------------------------------
# Constrained least squares
# ---------------------------------------------------------------------------
def constrained_lsq_ik(J, V, lo, hi, reg: float = 1e-6) -> IKResult:
    """min |J qd - V|^2  s.t.  lo <= qd <= hi  (velocity, position, accel limits).

    Hard limits mean the robot never jumps or violates its range.

    Why it deviates: when a limit is active, V is not achievable, and least
    squares returns the qd whose J qd is *closest* to V. "Closest" in a
    6-D sum of squares does not keep the direction. The EE moves along a
    different line than commanded, and translation and rotation errors get
    mixed. A welding, cutting or insertion task or a collision-checked path
    assumes the tool stays on the line, so this is unacceptable on real robots.
    """
    n = J.shape[1]
    P = 2 * J.T @ J + reg * np.eye(n)
    c = -2 * J.T @ V
    x, ok = _solve_qp(P, c, np.eye(n), lo, hi)
    if not ok:
        return IKResult(qd=np.zeros(n), ok=False, note="QP failed -> stop")
    return IKResult(qd=x[:n])


# ---------------------------------------------------------------------------
# Direction-preserving velocity scaling + null-space posture
# ---------------------------------------------------------------------------
def velocity_scaling_ik(J, V, lo, hi, qd_null_des, null_weight: float = 1e-2,
                        soft_rows=None, soft_weight: float = 0.5,
                        reg: float = 1e-6) -> IKResult:
    """max alpha  s.t.  J qd = alpha V,  0 <= alpha <= 1,  lo <= qd <= hi.

    Instead of approximating V, scale it: the EE moves *exactly* along the
    commanded direction, only slower (alpha < 1) when a limit would be hit.
    Near a singularity, or when V points where the arm cannot go, alpha
    drops toward 0 and the robot stops safely instead of jumping
    (pseudoinverse) or drifting off course (constrained least squares). This
    is the formulation Drake's DifferentialInverseKinematics uses.

    7-DOF extra: J qd = alpha V leaves one free direction (the 1-D null space,
    the elbow self-motion). Pure "max alpha" leaves it undetermined, and the
    solver would pick arbitrary elbow motion. A small secondary cost fixes it:

        min  -alpha + w |N (qd - qd_null_des)|^2,   N = I - J^+ J

    N projects onto motions that do not move the EE, so this term only shapes
    the elbow (posture regulation, user elbow swing) and never fights the task.

    Task priority (soft_rows): rows listed there leave the hard equality and
    become a cost  soft_weight |J_s qd - V_s|^2. Example: soft_rows = [3, 4, 5]
    keeps position exact (direction-preserving, scaled by alpha) and lets
    orientation give way only when holding it would slow the position down.
    That is how "move the position, change the angle if needed" works.

    Decision vector x = [qd (n), alpha].
    """
    n = J.shape[1]
    N = np.eye(n) - np.linalg.pinv(J) @ J  # symmetric, idempotent: N^T N = N
    soft = np.zeros(6, dtype=bool)
    if soft_rows is not None:
        soft[list(soft_rows)] = True
    Jh, Vh, Js, Vs = J[~soft], V[~soft], J[soft], V[soft]

    P = np.zeros((n + 1, n + 1))
    P[:n, :n] = 2 * null_weight * N + 2 * soft_weight * Js.T @ Js
    P += reg * np.eye(n + 1)
    c = np.zeros(n + 1)
    c[:n] = -2 * null_weight * N @ qd_null_des - 2 * soft_weight * Js.T @ Vs
    c[n] = -1.0

    # equality  J_h qd - alpha V_h = 0   and box on [qd, alpha]
    A = np.vstack([np.hstack([Jh, -Vh[:, None]]), np.eye(n + 1)])
    l = np.concatenate([np.zeros(len(Vh)), lo, [0.0]])
    u = np.concatenate([np.zeros(len(Vh)), hi, [1.0]])

    x, ok = _solve_qp(P, c, A, l, u)
    if not ok:
        return IKResult(qd=np.zeros(n), alpha=0.0, ok=False, note="QP failed -> stop")
    return IKResult(qd=x[:n], alpha=float(np.clip(x[n], 0.0, 1.0)))
