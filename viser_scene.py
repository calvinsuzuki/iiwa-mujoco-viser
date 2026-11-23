"""3-D scene pieces for the Viser UI: studio look, robot meshes and the teaching
overlays (manipulability ellipsoid, null-space circle, joint-limit rings)."""

import mujoco
import numpy as np
import trimesh

from rotations import quat_from_axes, quat_from_mat
from viser_style import BG_CENTRE, BG_EDGE

VISUAL_GROUP = 2                  # menagerie: visual meshes are in geom group 2
ELLIPSOID_SCALE = 0.20            # m of drawn radius per (m/s)/(rad/s) of singular value
NULL_COLOR = (170, 120, 255)      # null-space overlay (violet)
NO_SHADOW = dict(cast_shadow=False, receive_shadow=False)


def set_if_changed(handle, attr, value):
    """Assign a scalar property only when it changed: every assignment is sent to
    all clients."""
    if getattr(handle, attr) != value:
        setattr(handle, attr, value)


def segments(points):
    """Polyline (n, 3) -> line segments (n-1, 2, 3) for add_line_segments."""
    pts = np.asarray(points, dtype=np.float32)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def unit_circle(n):
    """n+1 points of the unit circle in the XY plane, closed (last = first)."""
    th = np.linspace(0, 2 * np.pi, n + 1)
    return np.c_[np.cos(th), np.sin(th), np.zeros_like(th)]


def look_at_origin_quat(pos):
    """Orientation that points a light's -Z axis from pos toward the origin."""
    z = np.asarray(pos, float) / np.linalg.norm(pos)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    return quat_from_axes(x, np.cross(z, x), z)


def build_studio(scene):
    """Dark studio: radial gradient backdrop, floor grid fading into fog, shadows."""
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.clip(np.hypot((xx - w / 2) / (w / 2), (yy - h * 0.42) / (h / 2)) / 1.3, 0, 1)[..., None] ** 1.4
    centre, edge = np.array(BG_CENTRE, float), np.array(BG_EDGE, float)
    scene.set_background_image((centre * (1 - t) + edge * t).astype(np.uint8), format="png")
    scene.configure_fog(near=3.0, far=9.0, color=BG_EDGE)
    scene.add_grid("/floor", width=20, height=20, infinite_grid=True,
                   cell_size=0.1, cell_color=(58, 64, 72), cell_thickness=0.6,
                   section_size=0.5, section_color=(84, 92, 104), section_thickness=1.0,
                   fade_distance=7.0, fade_strength=1.5, fade_from="origin",
                   plane_color=BG_EDGE, plane_opacity=0.0, shadow_opacity=0.35)
    scene.configure_default_lights(enabled=False)
    scene.add_light_hemisphere("/lights/sky", sky_color=(220, 228, 240),
                               ground_color=(40, 42, 48), intensity=1.6)
    for name, pos, intensity, color, shadow in (("key", (-0.5, 0.5, 2.6), 2.2, (255, 255, 255), True),
                                                ("rim", (1.5, -1.0, 1.0), 0.8, (200, 215, 255), False)):
        scene.add_light_directional(f"/lights/{name}", intensity=intensity, color=color,
                                    cast_shadow=shadow, position=pos, wxyz=look_at_origin_quat(pos))


class RobotMeshes:
    """The MuJoCo visual meshes as scene nodes, posed by update(q). self.d holds
    the forward kinematics of the last update (joint anchors and axes too)."""

    def __init__(self, scene, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.handles = []
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != VISUAL_GROUP:
                continue
            mid = model.geom_dataid[g]
            va, fa, fn = model.mesh_vertadr[mid], model.mesh_faceadr[mid], model.mesh_facenum[mid]
            matid = model.geom_matid[g]
            rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[g]
            # One vertex per face corner with MuJoCo's per-corner normals (split at
            # hard edges). Shared vertices would make three.js average the normals
            # across sharp CAD edges: blotchy, faceted shading.
            faces = model.mesh_face[fa:fa + fn]
            verts = model.mesh_vert[va:va + model.mesh_vertnum[mid]][faces].reshape(-1, 3)
            normals = model.mesh_normal[model.mesh_normaladr[mid] + model.mesh_facenormal[fa:fa + fn]]
            tm = trimesh.Trimesh(verts, np.arange(len(verts)).reshape(-1, 3),
                                 vertex_normals=normals.reshape(-1, 3), process=False)
            tm.visual = trimesh.visual.ColorVisuals(
                tm, vertex_colors=np.tile((255 * rgba).astype(np.uint8), (len(verts), 1)))
            # Casts shadows but doesn't receive them: self-shadowing from the shadow
            # map gives "shadow acne" (a stripe/square pattern on the surfaces).
            self.handles.append((g, scene.add_mesh_trimesh(f"/robot/geom_{g}", tm, receive_shadow=False)))

    def update(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        for g, h in self.handles:
            h.position = self.d.geom_xpos[g]
            h.wxyz = quat_from_mat(self.d.geom_xmat[g])


def isotropy_color(ratio):
    """Ellipsoid colour from sigma_min / sigma_max: red (flat) -> amber -> green
    (round); a ratio of 0.5 or more already counts as healthy."""
    t = float(np.clip(ratio / 0.5, 0.0, 1.0))
    red, amber, green = np.array([235, 60, 50]), np.array([245, 170, 40]), np.array([60, 200, 110])
    c = red + (amber - red) * (t / 0.5) if t < 0.5 else amber + (green - amber) * ((t - 0.5) / 0.5)
    return tuple(int(v) for v in c)


class Overlays:
    """Teaching overlays.

    * Manipulability ellipsoid at the tool: the image of the unit joint-velocity
      ball under the translational Jacobian Jv. Its semi-axes are the left singular
      vectors of Jv scaled by the singular values: long where the tool can move
      fast, short where it barely can. It flattens near a singularity.
    * Null space: the iiwa's self-motion keeps shoulder, wrist and tool fixed and
      swings the elbow on a circle around the shoulder-wrist line. Drawn: that
      line, the circle (with a faint disc), a spoke to the elbow and, for the
      velocity scaling QP, the elbow-angle setpoint on the circle.
    """

    def __init__(self, scene):
        self.ellipsoid = scene.add_icosphere("/overlay/ellipsoid", radius=1.0, subdivisions=4,
                                             color=(60, 200, 110), opacity=0.35, **NO_SHADOW)
        # Semi-axes in world coordinates: a child of the icosphere would not
        # inherit its per-axis scale.
        self.ellipsoid_axes = scene.add_line_segments(
            "/overlay/ellipsoid_axes", np.zeros((3, 2, 3)), (235, 240, 245),
            thickness=1.5, thickness_units="screen")

        # Circle frame: origin at the circle centre, x = zero-angle reference, z =
        # shoulder->wrist axis. Its children are unit size, scaled by the radius.
        self.null_frame = scene.add_frame("/overlay/null", show_axes=False)
        ring = unit_circle(96)
        self.circle = scene.add_line_segments("/overlay/null/circle", segments(ring), NULL_COLOR,
                                              thickness=2.5, thickness_units="screen")
        n = len(ring) - 1
        self.disc = scene.add_mesh_simple(
            "/overlay/null/disc", np.vstack([[0, 0, 0], ring[:-1]]),
            np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]),
            color=NULL_COLOR, opacity=0.10, side="double", **NO_SHADOW)
        line = dict(thickness_units="screen")
        self.axis_line = scene.add_line_segments("/overlay/null_axis", np.zeros((1, 2, 3)),
                                                 NULL_COLOR, thickness=1.5, **line)
        self.spoke = scene.add_line_segments("/overlay/null_spoke", np.zeros((1, 2, 3)),
                                             NULL_COLOR, thickness=2.5, **line)
        self.ends = [scene.add_icosphere(f"/overlay/null_{k}", radius=0.012, color=NULL_COLOR,
                                         **NO_SHADOW) for k in ("shoulder", "wrist")]
        self.setpoint = scene.add_icosphere("/overlay/null_setpoint", radius=0.016,
                                            color=(255, 255, 255), opacity=0.8, **NO_SHADOW)
        self.null_nodes = [self.null_frame, self.axis_line, self.spoke, *self.ends, self.setpoint]

    def set_visible(self, ellipsoid, null_space):
        self.ellipsoid.visible = self.ellipsoid_axes.visible = ellipsoid
        for h in self.null_nodes:
            h.visible = null_space

    def update(self, ctl, q, pos, J):
        """ctl: the IKController (elbow geometry, setpoint); q: the pose drawn."""
        if self.ellipsoid.visible:
            U, S, _ = np.linalg.svd(J[:3])
            if np.linalg.det(U) < 0:              # make U a rotation (right-handed)
                U[:, 2] *= -1
            self.ellipsoid.position = pos
            self.ellipsoid.wxyz = quat_from_axes(*U.T)
            self.ellipsoid.scale = tuple(np.maximum(ELLIPSOID_SCALE * S, 1e-3))
            self.ellipsoid.color = isotropy_color(S[2] / S[0])
            half = (U * (ELLIPSOID_SCALE * S)).T  # rows: semi-axis vectors
            self.ellipsoid_axes.points = np.stack([pos - half, pos + half], axis=1).astype(np.float32)

        if self.null_frame.visible:
            s, e, w, u, ref = ctl.elbow_geometry(q)
            c = s + ((e - s) @ u) * u             # circle centre on the shoulder-wrist line
            r = max(float(np.linalg.norm(e - c)), 1e-4)
            b = np.cross(u, ref)
            self.null_frame.position, self.null_frame.wxyz = c, quat_from_axes(ref, b, u)
            self.circle.scale = self.disc.scale = (r, r, 1.0)
            self.axis_line.points = segments([s, w])
            self.spoke.points = segments([c, e])
            self.ends[0].position, self.ends[1].position = s, w
            show_sp = ctl.method_key == "qp-scaled" and abs(ctl.elbow_target - ctl.elbow) > np.deg2rad(2)
            set_if_changed(self.setpoint, "visible", show_sp)
            if show_sp:
                psi = ctl.elbow_target
                self.setpoint.position = c + r * (np.cos(psi) * ref + np.sin(psi) * b)


class JointLimitRings:
    """Ring around each joint axis near a hard limit: amber when near, red at or
    beyond it (IKController.limit_levels). Two fixed-colour rings per joint,
    switched by visibility (recolouring line segments is unreliable)."""

    RADIUS = 0.13                 # m, wider than the links so the ring stays visible

    def __init__(self, scene):
        self.unit = unit_circle(48)[:, :2]

        def ring(name, rgb):
            return scene.add_line_segments(name, np.zeros((48, 2, 3)), rgb, thickness=4.0,
                                           thickness_units="screen", visible=False)
        self.rings = [(ring(f"/overlay/limit_warn_{j}", (245, 165, 36)),
                       ring(f"/overlay/limit_hit_{j}", (229, 72, 77))) for j in range(7)]

    def update(self, levels, d):
        """levels: per-joint limit level; d: MjData posed at the current q."""
        for j, ((near, hit), level) in enumerate(zip(self.rings, levels)):
            set_if_changed(near, "visible", bool(level == 1))
            set_if_changed(hit, "visible", bool(level == 2))
            if level == 0:
                continue
            a = d.xaxis[j] / np.linalg.norm(d.xaxis[j])
            u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            pts = d.xanchor[j] + self.RADIUS * (self.unit[:, :1] * u + self.unit[:, 1:] * np.cross(a, u))
            (hit if level == 2 else near).points = segments(pts)
