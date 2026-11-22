"""Quaternion helpers (MuJoCo convention: [w, x, y, z]) wrapping the mju_* routines,
which write into preallocated output arrays."""

import mujoco
import numpy as np


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).ravel())
    return q


def mat_from_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def quat_from_axes(x, y, z):
    """Rotation whose columns are the unit vectors x, y, z."""
    return quat_from_mat(np.column_stack([x, y, z]))


def quat_mul(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def quat_conj(q):
    out = np.zeros(4)
    mujoco.mju_negQuat(out, q)
    return out


def quat_axis_angle(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), angle)
    return q


def rotvec_from_quat(q):
    """Rotation vector (axis * angle) of a unit quaternion."""
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def same_rotation(a, b, tol=1e-9):
    """True when unit quaternions a and b describe the same rotation (q and -q do)."""
    return 1.0 - abs(float(np.dot(a, b))) <= tol
