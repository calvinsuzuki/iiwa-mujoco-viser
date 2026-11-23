"""Browser UI for the iiwa14 IK playground (Viser + headless MuJoCo).

Two modes: IK (kinematic, iiwa_core) and Control (torque controllers on the
simulated dynamics, iiwa_dynamics), which tracks the IK motion as its reference.

Run:
  .venv/bin/python iiwa_viser.py [--port 8080] [--local] [--open]
then open the printed URL: this machine's Tailscale name (any device on the
tailnet), or http://localhost:8080 with --local or when Tailscale is down.

Scene:  drag the gizmo arrows to move the tool (position exact, orientation may
        give way) or its rings to rotate it (the opposite). Red sphere =
        workspace limit. Ellipsoid = manipulability, violet circle = null space,
        rings = joints near a limit.
Panel:  mode, controller and physics (Control mode), IK method (hover a button
        for its formulas), poses, scripted moves, elbow swing, overlay toggles,
        joint sliders; status and live plots bottom-left.

Modules: iiwa_core (IK controller), ik_solvers (the IK methods), iiwa_dynamics
(Control mode), viser_scene (3-D scene), viser_style (colours, CSS), rotations
(quaternion helpers).
"""

import argparse
import json
import subprocess
import time
import webbrowser

import numpy as np
import viser

from iiwa_core import DT, METHODS, MOVES, POSES, IKController
from iiwa_dynamics import CONTROLLERS, OMEGA_DEFAULT, PAYLOAD_MAX, ControlSim
from rotations import mat_from_quat, same_rotation
from viser_scene import (JointLimitRings, Overlays, RobotMeshes, build_studio, segments,
                         set_if_changed)
from viser_style import (AMBER, GREEN, JOINT_COLORS, KUKA_ORANGE, ORANGE, PANEL_CSS, RED,
                         controller_folder_html, ik_folder_html, joint_limit_css, marker)

RENDER_EVERY = 2                  # scene update every N control steps (50 Hz)
GUI_EVERY = 10                    # panel update every N control steps (10 Hz)
SLIDER_HOLD = 0.4                 # s the server leaves a slider alone after the user touched it
CLOSE_SINGULAR = 0.05             # sigma_min(J) below this: "close to singularity"
PANEL_WIDTH = 360                 # status + plots column
PLOT_ASPECT = 2.9                 # width / height; the "Expand plot" view keeps the ratio
TRAIL_COLOR = {"pinv": (230, 60, 60), "dls": (240, 170, 30),          # IK methods
               "qp-lsq": (80, 130, 255), "qp-scaled": (50, 200, 90),
               "pd": (230, 60, 60), "pd-gravity": (240, 170, 30),     # controllers
               "computed-torque": (80, 130, 255), "impedance": (50, 200, 90)}
STATE_COLOR = {"Moving": ORANGE, "At target": GREEN, "Off target": AMBER, "Can't reach target": RED}
PUSH_ARROW = 0.35                 # m, drawn length of the push arrow
POSE_HINTS = {"home": ("Home", "Tool down, wrist bent."),
              "transport": ("Transport", "KUKA shipping pose (0, 25, 0, 90, 0, 0, 0 deg)."),
              "zero": ("Zero", "All joints 0: arm straight up (singular).")}
assert POSE_HINTS.keys() == POSES.keys()


class App:
    def __init__(self, port, host="127.0.0.1", url_host="localhost"):
        self.url = f"http://{url_host}:{port}"
        self.ctl = IKController("qp-scaled", flexible=False)
        self.dyn = ControlSim(self.ctl)   # Control mode; idle until activated
        self.server = viser.ViserServer(host=host, port=port)
        self.dragging = False             # gizmo held: don't overwrite it from the target
        self.last_gizmo = None
        self.slider_touched = self.elbow_touched = 0.0
        self.limit_levels = None
        self.build_scene()
        self.build_panel()

    # -- scene -----------------------------------------------------------------------
    def build_scene(self):
        sc = self.server.scene
        sc.set_up_direction("+z")
        cam = self.server.initial_camera
        cam.position, cam.look_at, cam.up = (1.05, -0.90, 0.85), (-0.05, 0.0, 0.52), (0.0, 0.0, 1.0)
        build_studio(sc)
        self.robot = RobotMeshes(sc, self.ctl.m)
        self.ee_frame = sc.add_frame("/ee", axes_length=0.07, axes_radius=0.004)
        self.overlays = Overlays(sc)
        self.limit_rings = JointLimitRings(sc)
        self.trail = None
        self.workspace = sc.add_icosphere(
            "/workspace", radius=self.ctl.ws_radius, color=(255, 40, 40), opacity=0.15,
            subdivisions=4, position=self.ctl.ws_center, visible=False,
            cast_shadow=False, receive_shadow=False)
        # Free spin axis of "Restrain only Z-axis" (the target's Z), shown when on.
        self.free_axis = sc.add_line_segments(
            "/target/free_axis", np.array([[[0, 0, -0.18], [0, 0, 0.30]]]), (0, 220, 255),
            thickness=4.0, thickness_units="screen", visible=False)
        self.gizmo = sc.add_transform_controls(
            "/target", scale=0.22, line_width=3.0, depth_test=False,
            position=self.ctl.target_pos, wxyz=self.ctl.target_quat)
        self.gizmo.on_update(self.on_gizmo)
        # Control mode: payload ball at the flange, arrow while the tool is pushed.
        self.payload_ball = sc.add_icosphere("/payload", radius=1.0, subdivisions=3,
                                             color=(150, 156, 164), visible=False)
        self.push_arrow = sc.add_line_segments("/push", np.zeros((1, 2, 3)), (229, 72, 77),
                                               thickness=6.0, thickness_units="screen", visible=False)

    async def on_gizmo(self, event):
        """Gizmo drag -> target. Whether this drag moves or rotates the gizmo sets
        the task priority (position or orientation exact)."""
        pos, quat = np.array(event.target.position), np.array(event.target.wxyz)
        if event.phase == "start":
            self.dragging = True
            self.last_gizmo = (pos, quat)
            return
        mode = None
        if self.last_gizmo is not None:
            if np.linalg.norm(pos - self.last_gizmo[0]) > 1e-6:
                mode = "position"
            elif not same_rotation(quat, self.last_gizmo[1]):
                mode = "orientation"
        self.last_gizmo = (pos, quat)
        self.ctl.submit("set_target", pos, quat, mode)
        if event.phase == "end":
            self.dragging = False

    # -- panel -----------------------------------------------------------------------
    def build_panel(self):
        gui, ctl = self.server.gui, self.ctl
        gui.configure_theme(control_width="medium", dark_mode=True, show_logo=False,
                            show_share_button=False, brand_color=KUKA_ORANGE)
        gui.set_panel_label("Controls")
        gui.main_panel.float(x=-15, y=50)     # room for the status pill above
        gui.add_html(PANEL_CSS)

        dyn = self.dyn

        def toggle(label, value, command, hint, target=ctl):
            cb = gui.add_checkbox(label, value, hint=hint)
            # client is None for our own sync from the controller state (update_panel).
            cb.on_update(lambda e: e.client is not None and target.submit(command, cb.value))
            return cb

        with gui.add_folder("Mode"):
            self.mode_buttons = {}
            for on, label, hint in ((False, "IK", "Kinematic: the IK method moves the joints directly."),
                                    (True, "Control", "Dynamics: joint torques from a controller move the "
                                                      "simulated arm, which tracks the IK motion.")):
                b = gui.add_button(label, color="gray", hint=hint)
                b.on_click(lambda _e, on=on: dyn.submit("set_active", on))
                self.mode_buttons[on] = b
            gui.add_html(marker("iiwa-btn2"))

        self.control_folders = [gui.add_folder("Controller"), gui.add_folder("Physics")]
        with self.control_folders[0]:
            self.controller_buttons = {}
            for key, label in CONTROLLERS:
                b = gui.add_button(label, color="gray")
                b.on_click(lambda _e, k=key: dyn.submit("set_controller", k))
                self.controller_buttons[key] = b
            self.omega = gui.add_slider(
                "Bandwidth ω [rad/s]", min=4, max=40, step=1, initial_value=OMEGA_DEFAULT,
                hint="How fast errors are corrected (gains ∝ ω² and 2ω). Higher: stiffer and "
                     "faster, but more torque. Range 4 to 40 rad/s.")
            self.omega.on_update(lambda e: e.client is not None and dyn.submit("set_omega", self.omega.value))
            gui.add_html(controller_folder_html(CONTROLLERS))
        with self.control_folders[1]:
            self.gravity = toggle("Gravity", dyn.gravity, "set_gravity",
                                  "Gravity on the arm (and in the controller's model).", dyn)
            self.torque_limits = toggle("Torque limits", dyn.torque_limits, "set_torque_limits",
                                        "Clip the torques to the iiwa 14 datasheet maxima: "
                                        "320, 320, 176, 176, 110, 40, 40 N·m.", dyn)
            self.payload = gui.add_slider(
                "Payload [kg]", min=0, max=PAYLOAD_MAX, step=0.5, initial_value=0,
                hint="Steel ball at the flange. The controller's model doesn't know it. "
                     f"Range 0 to {PAYLOAD_MAX:.0f} kg (iiwa 14 rated payload).")
            self.payload.on_update(lambda e: e.client is not None and dyn.submit("set_payload", self.payload.value))
            gui.add_button("Push the tool", color="gray",
                           hint="60 N sideways on the tool for 0.25 s.").on_click(lambda _e: dyn.submit("push"))
            gui.add_html(marker("iiwa-pair") + marker("iiwa-notick"))

        with gui.add_folder("IK method"):
            self.method_buttons = {}
            for key, label in METHODS:
                b = gui.add_button(label, color="gray")
                b.on_click(lambda _e, k=key: ctl.submit("set_method", k))
                self.method_buttons[key] = b
            self.flex = toggle("Reach flexibility", ctl.flexible, "set_flexible",
                               "Move: position exact, orientation gives way. Rotate: the opposite. "
                               "Off: full pose exact, robot stops instead. Turns off Restrain only Z-axis.")
            self.zflex = toggle("Restrain only Z-axis", ctl.z_flexible, "set_z_flexible",
                                "The tool may spin freely about the target's Z axis (cyan line). Position "
                                "and the Z direction stay exact. Turns off reach flexibility.")
            gui.add_html(ik_folder_html(METHODS))

        with gui.add_folder("Poses"):
            for key, (label, hint) in POSE_HINTS.items():
                gui.add_button(label, color="gray", hint=hint).on_click(
                    lambda _e, k=key: (ctl.submit("reset", k), dyn.submit("sync")))
            gui.add_html(marker("iiwa-btn3"))

        with gui.add_folder("Moves"):
            self.move_buttons = {}
            for key, (label, _) in MOVES.items():
                b = gui.add_button(label, color="gray")
                b.on_click(lambda _e, k=key: (ctl.submit("start_move", k), dyn.submit("sync")))
                self.move_buttons[key] = b
            self.move_note = gui.add_html("")
            gui.add_html(marker("iiwa-btn3"))

        self.motion = gui.add_folder("Motion")    # shown for the velocity scaling QP only
        with self.motion:
            self.elbow = gui.add_slider(
                "Elbow angle [deg]", min=-180, max=180, step=1,
                initial_value=round(float(np.rad2deg(ctl.elbow_target))),
                hint="Null space: swings the elbow around the shoulder-wrist line while the "
                     "tool stays put. Held while you move the target. Range -180° to 180°.")

            @self.elbow.on_update
            def _(e):
                if e.client is not None:
                    self.elbow_touched = time.perf_counter()
                    ctl.submit("set_elbow", self.elbow.value)
            gui.add_html(marker("iiwa-motion"))

        with gui.add_folder("Show"):
            self.show_ellipsoid = gui.add_checkbox(
                "Manipulability", True,
                hint="Tool velocities reachable with unit joint speed (translational Jacobian). "
                     "Round and green = can move well in every direction; flat and red = near a "
                     "singularity.")
            self.show_null = gui.add_checkbox(
                "Null space", True,
                hint="Self-motion: the elbow can swing on this circle around the shoulder-wrist "
                     "line while the tool stays put. White dot = elbow angle setpoint.")
            gui.add_html(marker("iiwa-pair"))
            for cb in (self.show_ellipsoid, self.show_null):
                cb.on_update(lambda _e: self.overlays.set_visible(self.show_ellipsoid.value,
                                                                  self.show_null.value))

        status = gui.add_panel()                   # bottom-left column
        with status.add_tab("Status", viser.Icon.INFO_CIRCLE):
            self.status_md = gui.add_markdown("")
            self.build_plots()
        status.float(x=15, y=-15, width=PANEL_WIDTH)

        with gui.add_folder("Joints [deg]"):
            # Marker, and also where the per-joint limit tint CSS goes (update_panel).
            self.joints_html = gui.add_html(marker("iiwa-joints"))
            self.joint_sliders = []
            for i in range(7):
                lo, hi = np.rad2deg(ctl.q_min_hard[i]), np.rad2deg(ctl.q_max_hard[i])
                s = gui.add_slider(f"J{i + 1}", min=round(lo, 1), max=round(hi, 1), step=0.1,
                                   initial_value=round(float(np.rad2deg(ctl.q[i])), 1),
                                   hint=f"Joint {i + 1} range: {lo:.0f}° to {hi:.0f}° (centre tick = 0°).")
                s.on_update(self.on_joint_slider)
                self.joint_sliders.append(s)

    def build_plots(self):
        """Two plot folders under the status table, last few seconds. Sized by
        aspect ratio, so "Expand plot" gives a taller plot rather than a strip."""
        gui = self.server.gui
        axis = dict(stroke="#9aa4b0", grid={"stroke": "rgba(255,255,255,0.07)", "width": 1},
                    ticks={"stroke": "rgba(255,255,255,0.15)", "width": 1})
        common = dict(cursor={"drag": {"x": False, "y": False}, "points": {"size": 6}},
                      aspect=PLOT_ASPECT)
        x = np.zeros(2)
        self.plot_folders = {False: [gui.add_folder("Speed"), gui.add_folder("Joint speeds")],
                             True: [gui.add_folder("Torques"), gui.add_folder("Tool error")]}
        with self.plot_folders[False][0]:
            self.plot_speed = gui.add_uplot(
                data=(x, x, x), title="Tool speed α and σmin(J), last 8 s",
                series=({"label": "t [s]"},
                        {"label": "speed α [%]", "stroke": ORANGE, "width": 2, "scale": "pct"},
                        {"label": "σmin(J)", "stroke": "#7cc4ff", "width": 2, "scale": "sig"}),
                scales={"x": {"time": False}, "pct": {"auto": False, "range": [0, 105]},
                        "sig": {"auto": False, "range": [0, 0.35]}},
                axes=({**axis, "size": 26}, {**axis, "scale": "pct", "size": 44, "space": 16},
                      {**axis, "scale": "sig", "side": 1, "size": 42}),
                padding=(8, 4, 0, 0), **common)
            gui.add_html(f'<div class="iiwa-note"><b style="color:{ORANGE}">speed α %</b> (left) · '
                         '<b style="color:#7cc4ff">σ<sub>min</sub></b> (right) · time [s]</div>')
        with self.plot_folders[False][1]:
            self.plot_qd = gui.add_uplot(
                data=(x,) * 8, title="Joint speed, % of limit, last 8 s",
                series=({"label": "t [s]"}, *({"label": f"J{i + 1}", "stroke": c, "width": 1.5}
                                              for i, c in enumerate(JOINT_COLORS))),
                scales={"x": {"time": False}, "y": {"auto": False, "range": [-125, 125]}},
                axes=({**axis, "size": 26}, {**axis, "size": 44, "space": 16}),
                padding=(8, 12, 0, 0), **common)
            key = " ".join(f'<b style="color:{c}">J{i + 1}</b>' for i, c in enumerate(JOINT_COLORS))
            gui.add_html(f'<div class="iiwa-note">% of limit: {key}</div>')
        with self.plot_folders[True][0]:
            self.plot_tau = gui.add_uplot(
                data=(x,) * 8, title="Joint torque, % of the datasheet maximum, last 8 s",
                series=({"label": "t [s]"}, *({"label": f"J{i + 1}", "stroke": c, "width": 1.5}
                                              for i, c in enumerate(JOINT_COLORS))),
                scales={"x": {"time": False}, "y": {"auto": False, "range": [-125, 125]}},
                axes=({**axis, "size": 26}, {**axis, "size": 44, "space": 16}),
                padding=(8, 12, 0, 0), **common)
            self.tau_note = gui.add_html("")
        with self.plot_folders[True][1]:
            self.plot_err = gui.add_uplot(
                data=(x, x, x), title="Tool error to the target, last 8 s",
                series=({"label": "t [s]"},
                        {"label": "position [mm]", "stroke": ORANGE, "width": 2, "scale": "mm"},
                        {"label": "orientation [°]", "stroke": "#7cc4ff", "width": 2, "scale": "deg"}),
                scales={"x": {"time": False}, "mm": {"auto": True}, "deg": {"auto": True}},
                axes=({**axis, "size": 26}, {**axis, "scale": "mm", "size": 44, "space": 16},
                      {**axis, "scale": "deg", "side": 1, "size": 42}),
                padding=(8, 4, 0, 0), **common)
            gui.add_html(f'<div class="iiwa-note"><b style="color:{ORANGE}">position [mm]</b> (left) · '
                         '<b style="color:#7cc4ff">orientation [°]</b> (right) · time [s]</div>')

    def on_joint_slider(self, event):
        if event.client is None:                   # our own sync from the robot state
            return
        self.slider_touched = time.perf_counter()
        self.ctl.submit("set_joints", np.deg2rad([s.value for s in self.joint_sliders]))

    # -- per-tick updates --------------------------------------------------------------
    def update_scene(self, tick):
        ctl, dyn = self.ctl, self.dyn
        control = dyn.active
        q = dyn.q if control else ctl.q
        with self.server.atomic():
            self.robot.update(q)
            pos, quat, J = ctl.kin(q)
            self.ee_frame.position, self.ee_frame.wxyz = pos, quat
            self.overlays.update(ctl, q, pos, J)
            self.limit_rings.update(ctl.limit_levels(q), self.robot.d)
            r = dyn.payload_radius
            set_if_changed(self.payload_ball, "visible", bool(control and r > 0))
            if control and r > 0:
                self.payload_ball.position = pos + r * mat_from_quat(quat)[:, 2]
                self.payload_ball.scale = (r, r, r)
            set_if_changed(self.push_arrow, "visible", bool(control and dyn.pushing))
            if control and dyn.pushing:
                tip = dyn.d.xpos[dyn.flange_id]
                f = dyn.push_force / np.linalg.norm(dyn.push_force)
                self.push_arrow.points = segments([tip - PUSH_ARROW * f, tip])
            # Show the shaped target (clamped into reach, flexible part copied).
            if not self.dragging and (np.linalg.norm(np.array(self.gizmo.position) - ctl.target_pos) > 1e-5
                                      or not same_rotation(np.array(self.gizmo.wxyz), ctl.target_quat)):
                self.gizmo.position, self.gizmo.wxyz = ctl.target_pos, ctl.target_quat
            set_if_changed(self.free_axis, "visible", ctl.z_flexible)
            set_if_changed(self.workspace, "visible",
                           ctl.workspace_violation and (time.perf_counter() * 6) % 2 < 1)
            trail = dyn.trail if control else ctl.trail
            if len(trail) <= 1:
                if self.trail is not None:
                    self.trail.remove()
                    self.trail = None
            elif tick % GUI_EVERY == 0:            # re-adding a node replaces it
                self.trail = self.server.scene.add_line_segments(
                    "/trail", segments(trail), TRAIL_COLOR[dyn.controller if control else ctl.method_key],
                    thickness=3.0, thickness_units="screen")

    def update_panel(self):
        ctl, dyn = self.ctl, self.dyn
        control = dyn.active
        for on, b in self.mode_buttons.items():
            set_if_changed(b, "color", "orange" if on == control else "gray")
        for f in self.control_folders:
            set_if_changed(f, "visible", control)
        for on, folders in self.plot_folders.items():
            for f in folders:
                set_if_changed(f, "visible", on == control)
        for key, b in self.controller_buttons.items():
            set_if_changed(b, "color", "orange" if key == dyn.controller else "gray")
        set_if_changed(self.gravity, "value", dyn.gravity)
        set_if_changed(self.torque_limits, "value", dyn.torque_limits)
        for key, b in self.method_buttons.items():
            set_if_changed(b, "color", "orange" if key == ctl.method_key else "gray")
        set_if_changed(self.motion, "visible", ctl.method_key == "qp-scaled")
        set_if_changed(self.flex, "value", ctl.flexible)
        set_if_changed(self.zflex, "value", ctl.z_flexible)

        # Sliders follow the robot, except right after the user moved them.
        now = time.perf_counter()
        if now - self.elbow_touched > SLIDER_HOLD:
            set_if_changed(self.elbow, "value",
                           int(round(float(np.clip(np.rad2deg(ctl.elbow_target), -180, 180)))))
        if now - self.slider_touched > SLIDER_HOLD:
            for s, qi in zip(self.joint_sliders, np.rad2deg(dyn.q if control else ctl.q)):
                set_if_changed(s, "value", round(float(np.clip(qi, s.min, s.max)), 1))
        levels = tuple(ctl.limit_levels(dyn.q if control else None))
        if levels != self.limit_levels:
            self.limit_levels = levels
            self.joints_html.content = marker("iiwa-joints") + joint_limit_css(levels)

        move = ctl.move_progress
        for key, b in self.move_buttons.items():
            set_if_changed(b, "color", "orange" if move and ctl.move.key == key else "gray")
        set_if_changed(self.move_note, "content",
                       f'<div class="iiwa-note">Playing <b>{move[0]}</b> … {100 * move[1]:.0f} %. '
                       'Drag the target or pick a pose to stop.</div>' if move else "")
        if control:
            self.update_control_status()
            self.update_control_plots()
        else:
            self.update_status()
            self.update_plots()

    @staticmethod
    def state_html(st):
        state = f'<b style="color:{STATE_COLOR[st.state]}">{st.state}</b>'
        if st.warnings:            # nowrap: the parenthesis moves to the next line whole
            ws = ", ".join(f'<span style="color:{RED if "limit" in w or "blew" in w else AMBER}">{w}</span>'
                           for w in st.warnings)
            state += f' <span style="white-space: nowrap">({ws})</span>'
        return state

    @staticmethod
    def sigma_html(sigma_min):
        near = (f' <span style="color:{AMBER}">(close to singularity)</span>'
                if sigma_min < CLOSE_SINGULAR else "")
        return f"{sigma_min:.3f}{near}"

    def update_status(self):
        st = self.ctl.status
        self.status_md.content = (
            "| | |\n|---|---|\n"
            f"| State | {self.state_html(st)} |\n"
            f"| Position error | {st.pos_err_mm:.1f} mm |\n"
            f"| Orientation error | {st.rot_err_deg:.2f}° |\n"
            f"| Elbow angle | {st.elbow_deg:.1f}° |\n"
            f"| Speed | {100 * st.speed:.0f} % |\n"
            f"| σ<sub>min</sub>(J) | {self.sigma_html(st.sigma_min)} |")

    def update_control_status(self):
        st = self.dyn.status
        peak = 100 * st.peak_frac
        color = RED if peak >= 99.5 else AMBER if peak > 80 else "inherit"
        self.status_md.content = (
            "| | |\n|---|---|\n"
            f"| State | {self.state_html(st)} |\n"
            f"| Position error | {st.pos_err_mm:.1f} mm |\n"
            f"| Orientation error | {st.rot_err_deg:.2f}° |\n"
            f"| Joint error to IK | {st.track_err_deg:.2f}° |\n"
            f'| Peak torque | J{st.peak_joint + 1} <span style="color:{color}">{peak:.0f} %</span> |\n'
            f"| σ<sub>min</sub>(J) | {self.sigma_html(st.sigma_min)} |")

    def update_plots(self):
        rows = list(self.ctl.history)[::2]         # 50 Hz is plenty to draw
        if len(rows) < 2:
            return
        t = np.array([r.t for r in rows])
        qd = np.clip(100 * np.array([r.qd_frac for r in rows]), -130, 130)
        self.plot_speed.data = (t - t[-1], 100 * np.array([r.alpha for r in rows]),
                                np.array([r.sigma_min for r in rows]))
        self.plot_qd.data = (t - t[-1], *qd.T)

    def update_control_plots(self):
        dyn = self.dyn
        rows = list(dyn.history)[::2]
        # Live torques [N m] under the plot, doubling as its colour key.
        self.tau_note.content = '<div class="iiwa-note">N·m: ' + " ".join(
            f'<b style="color:{c}">J{i + 1}</b>&nbsp;{round(tau) + 0:+d}'
            for i, (c, tau) in enumerate(zip(JOINT_COLORS, dyn.tau))) + "</div>"
        if len(rows) < 2:
            return
        t = np.array([r.t for r in rows])
        tau = np.clip(100 * np.array([r.tau_frac for r in rows]), -130, 130)
        self.plot_tau.data = (t - t[-1], *tau.T)
        self.plot_err.data = (t - t[-1], np.array([r.pos_err_mm for r in rows]),
                              np.array([r.rot_err_deg for r in rows]))

    def run(self):
        print(f"Open {self.url}  (Ctrl+C to quit)", flush=True)
        tick = 0
        try:
            while True:
                t0 = time.perf_counter()
                self.ctl.step()
                self.dyn.step()
                if tick % RENDER_EVERY == 0:
                    self.update_scene(tick)
                if tick % GUI_EVERY == 0:
                    self.update_panel()
                tick += 1
                time.sleep(max(0.0, DT - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def tailscale_address():
    """(IPv4, MagicDNS name) of this machine on the tailnet, or None when Tailscale
    is not installed or not connected. Binding to that IP keeps the app off the LAN:
    only devices on the tailnet can open it."""
    try:
        out = subprocess.run(["tailscale", "status", "--json", "--self", "--peers=false"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        me = json.loads(out)["Self"]
        ip = next(a for a in me.get("TailscaleIPs", []) if "." in a)
        return ip, me.get("DNSName", "").rstrip(".") or ip
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration):
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--local", action="store_true",
                    help="serve on localhost only instead of the Tailscale network")
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    args = ap.parse_args()
    ts = None if args.local else tailscale_address()
    if not ts and not args.local:
        print("Tailscale not running, serving on localhost only.")
    app = App(args.port, *(ts or ()))
    if args.open:
        webbrowser.open(app.url)
    app.run()
