import sys
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QFormLayout,
                             QCheckBox, QSpinBox, QPushButton, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #232629; color: #e8e8e8; }
QGroupBox {
    border: 1px solid #444; border-radius: 6px; margin-top: 10px;
    font-weight: 600; padding-top: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #7fc7ff; }
QLabel { color: #e8e8e8; }
QSlider::groove:horizontal { height: 6px; background: #3a3d40; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #7fc7ff; width: 14px; margin: -5px 0; border-radius: 7px;
}
QSpinBox, QComboBox {
    background: #2c2f33; border: 1px solid #444; border-radius: 4px;
    padding: 2px 4px; color: #e8e8e8;
}
QPushButton {
    background: #3a3d40; border: 1px solid #555; border-radius: 5px; padding: 6px;
}
QPushButton:hover { background: #48505a; }
QCheckBox { color: #e8e8e8; }
QScrollArea { border: none; }
"""


class LegIK:
    """Inverse kinematics solver for a single 3-DOF (coxa/femur/tibia) leg.
    All coordinates here are in the LEG's own local frame: origin at the
    coxa (hip) mount, +x/+z horizontal, y = height relative to the hip."""

    def __init__(self, coxa=50.0, femur=55.0, tibia=30.0, ground_clearance=50.0):
        self.Coxa = coxa
        self.Femur = femur
        self.Tibia = tibia
        self.G_C = ground_clearance

    def solve(self, x, y, z):
        """Returns (theta_1, theta_2, theta_3) in degrees, or (None, None, None)
        if the target is outside the leg's reachable workspace."""
        try:
            D = math.sqrt(x ** 2 + z ** 2)
            theta_1 = math.degrees(math.atan2(x, z))

            d = D - self.Coxa
            y_offset = self.G_C - y
            R = math.sqrt(d ** 2 + y_offset ** 2)

            if R > (self.Femur + self.Tibia) or R < abs(self.Femur - self.Tibia):
                return None, None, None
            if R < 1e-9:
                return None, None, None

            alpha_1 = math.degrees(math.acos(np.clip(y_offset / R, -1.0, 1.0)))
            cos_alpha_2 = (self.Femur ** 2 + R ** 2 - self.Tibia ** 2) / (2 * self.Femur * R)
            alpha_2 = math.degrees(math.acos(np.clip(cos_alpha_2, -1.0, 1.0)))
            theta_2 = alpha_1 + alpha_2

            cos_theta_3 = (self.Femur ** 2 + self.Tibia ** 2 - R ** 2) / (2 * self.Femur * self.Tibia)
            theta_3 = math.degrees(math.acos(np.clip(cos_theta_3, -1.0, 1.0)))

            return theta_1, theta_2, theta_3
        except (ValueError, ZeroDivisionError):
            return None, None, None

    def forward(self, t1, t2, t3):
        """Rebuilds the 4 joint positions (hip, coxa-end, knee, foot) in the
        leg's local frame purely from the joint angles, for FK validation."""
        t1_rad, t2_rad, t3_rad = math.radians(t1), math.radians(t2), math.radians(t3)
        u = np.array([math.sin(t1_rad), 0, math.cos(t1_rad)])
        v = np.array([0, -1, 0])

        p0 = np.array([0.0, 0.0, 0.0])
        p1 = p0 + self.Coxa * u
        p2 = p1 + self.Femur * (math.sin(t2_rad) * u + math.cos(t2_rad) * v)
        tibia_angle = t2_rad - (math.pi - t3_rad)
        p3 = p2 + self.Tibia * (math.sin(tibia_angle) * u + math.cos(tibia_angle) * v)
        return [p0, p1, p2, p3]


# ---------------------------------------------------------------------
# Robot-level geometry: 4 legs mounted around a rectangular body
# ---------------------------------------------------------------------
LEG_NAMES = ['FR', 'FL', 'BR', 'BL']  # Front-Right, Front-Left, Back-Right, Back-Left
LEG_COLORS = {
    'FR': '#7fc7ff',
    'FL': '#6bff95',
    'BR': '#ffb37f',
    'BL': '#ff7fd4',
}
LEG_MOUNT_YAW_DEG = {  # rotation applied to each leg's local frame to place it in world space
    'FR': 45.0,
    'FL': -45.0,
    'BR': 135.0,
    'BL': -135.0,
}


def leg_mount_point(name, body_length, body_width):
    """World-frame (x, z) of a leg's hip, given body dimensions. Body is
    centered on the origin; +x = forward, +z = right."""
    hl, hw = body_length / 2.0, body_width / 2.0
    if name == 'FR':
        return hl, hw
    if name == 'FL':
        return hl, -hw
    if name == 'BR':
        return -hl, hw
    if name == 'BL':
        return -hl, -hw
    raise ValueError(f"Unknown leg name: {name}")


def rotate_xz(x, z, yaw_deg):
    """Rotates a point (x, z) by yaw_deg about the vertical (Y) axis."""
    r = math.radians(yaw_deg)
    c, s = math.cos(r), math.sin(r)
    return x * c - z * s, x * s + z * c


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spider Robot IK Visualizer (4 Legs + Body)")
        self.resize(1300, 780)
        self.setStyleSheet(DARK_STYLESHEET)

        self.default_xyz = (70, 0, 40)
        self.body_length = 160
        self.body_width = 100

        # One independent IK solver + current local target per leg
        self.leg_solvers = {name: LegIK() for name in LEG_NAMES}
        self.leg_targets = {name: list(self.default_xyz) for name in LEG_NAMES}
        self.current_leg = 'FR'

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # ---------------- Left panel (scrollable controls) ----------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        scroll.setWidget(control_widget)
        layout.addWidget(scroll, 1)

        # Body dimensions
        body_group = QGroupBox("Body")
        body_form = QFormLayout()
        self.spin_length = QSpinBox()
        self.spin_length.setRange(80, 280)
        self.spin_length.setValue(self.body_length)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(60, 220)
        self.spin_width.setValue(self.body_width)
        body_form.addRow("Length (X, fwd/back):", self.spin_length)
        body_form.addRow("Width (Z, left/right):", self.spin_width)
        body_group.setLayout(body_form)
        control_layout.addWidget(body_group)

        # Leg selector
        select_group = QGroupBox("Leg Being Edited")
        select_layout = QVBoxLayout()
        self.combo_leg = QComboBox()
        self.combo_leg.addItems(LEG_NAMES)
        select_layout.addWidget(self.combo_leg)
        select_group.setLayout(select_layout)
        control_layout.addWidget(select_group)

        # Sliders + spin boxes for the selected leg's local X, Y, Z target
        coord_group = QGroupBox("Selected Leg's Local Foot Target (X, Y, Z)")
        coord_form = QFormLayout()
        self.slider_x, self.spin_x = self.create_slider_row(20, 150, self.default_xyz[0])
        self.slider_y, self.spin_y = self.create_slider_row(-30, 50, self.default_xyz[1])
        self.slider_z, self.spin_z = self.create_slider_row(-100, 100, self.default_xyz[2])
        coord_form.addRow(self.wrap_row("X (mm):", self.slider_x, self.spin_x))
        coord_form.addRow(self.wrap_row("Y (mm):", self.slider_y, self.spin_y))
        coord_form.addRow(self.wrap_row("Z (mm):", self.slider_z, self.spin_z))
        coord_group.setLayout(coord_form)
        control_layout.addWidget(coord_group)

        btn_row = QHBoxLayout()
        reset_leg_btn = QPushButton("Reset This Leg")
        reset_leg_btn.clicked.connect(self.reset_current_leg)
        reset_all_btn = QPushButton("Reset All Legs")
        reset_all_btn.clicked.connect(self.reset_all_legs)
        btn_row.addWidget(reset_leg_btn)
        btn_row.addWidget(reset_all_btn)
        control_layout.addLayout(btn_row)

        # Status overview for all 4 legs at a glance
        status_group = QGroupBox("All-Legs Status")
        status_layout = QVBoxLayout()
        self.lbl_status_overview = QLabel("")
        self.lbl_status_overview.setTextFormat(Qt.RichText)
        status_layout.addWidget(self.lbl_status_overview)
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)

        # Angle output for the selected leg
        angle_group = QGroupBox("Selected Leg's Joint Angles")
        angle_form = QFormLayout()
        self.lbl_theta1 = QLabel("Theta 1 (Coxa): -")
        self.lbl_theta2 = QLabel("Theta 2 (Femur): -")
        self.lbl_theta3 = QLabel("Theta 3 (Tibia): -")
        angle_form.addRow(self.lbl_theta1)
        angle_form.addRow(self.lbl_theta2)
        angle_form.addRow(self.lbl_theta3)
        angle_group.setLayout(angle_form)
        control_layout.addWidget(angle_group)

        # FK Validation for the selected leg
        val_group = QGroupBox("Selected Leg's FK Validation")
        val_layout = QVBoxLayout()
        self.check_validation = QCheckBox("Enable strict math validation check")
        self.check_validation.setChecked(True)
        self.lbl_validation = QLabel("Validation Output Here")
        self.lbl_validation.setWordWrap(True)
        self.lbl_validation.setStyleSheet("font-family: monospace;")
        val_layout.addWidget(self.check_validation)
        val_layout.addWidget(self.lbl_validation)
        val_group.setLayout(val_layout)
        control_layout.addWidget(val_group)

        self.check_workspace = QCheckBox("Show reachable workspace shell (selected leg)")
        self.check_workspace.setChecked(True)
        control_layout.addWidget(self.check_workspace)

        control_layout.addStretch()

        # ---------------- Right panel (3D plot) ----------------
        self.figure = Figure(facecolor="#232629")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, 2)

        # Signals
        self.combo_leg.currentTextChanged.connect(self.on_leg_selected)
        self.slider_x.valueChanged.connect(self.on_target_changed)
        self.slider_y.valueChanged.connect(self.on_target_changed)
        self.slider_z.valueChanged.connect(self.on_target_changed)
        self.spin_length.valueChanged.connect(self.update_plot)
        self.spin_width.valueChanged.connect(self.update_plot)
        self.check_validation.stateChanged.connect(self.update_plot)
        self.check_workspace.stateChanged.connect(self.update_plot)

        self.update_plot()

    # ---------------------------------------------------------------
    # UI helpers
    # ---------------------------------------------------------------
    def create_slider_row(self, min_val, max_val, default_val):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)

        spin = QSpinBox()
        spin.setMinimum(min_val)
        spin.setMaximum(max_val)
        spin.setValue(default_val)
        spin.setFixedWidth(70)

        slider.valueChanged.connect(lambda v: spin.blockSignals(True) or spin.setValue(v) or spin.blockSignals(False))
        spin.valueChanged.connect(lambda v: slider.blockSignals(True) or slider.setValue(v) or slider.blockSignals(False))
        return slider, spin

    def wrap_row(self, label_text, slider, spin):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(label_text))
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return row

    def on_leg_selected(self, name):
        """Switch which leg the sliders control, loading its stored target
        without re-triggering on_target_changed for the old leg."""
        self.current_leg = name
        x, y, z = self.leg_targets[name]
        for widget, val in ((self.slider_x, x), (self.slider_y, y), (self.slider_z, z)):
            widget.blockSignals(True)
            widget.setValue(val)
            widget.blockSignals(False)
        for widget, val in ((self.spin_x, x), (self.spin_y, y), (self.spin_z, z)):
            widget.blockSignals(True)
            widget.setValue(val)
            widget.blockSignals(False)
        self.update_plot()

    def on_target_changed(self):
        self.leg_targets[self.current_leg] = [
            self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        ]
        self.update_plot()

    def reset_current_leg(self):
        self.leg_targets[self.current_leg] = list(self.default_xyz)
        self.on_leg_selected(self.current_leg)

    def reset_all_legs(self):
        for name in LEG_NAMES:
            self.leg_targets[name] = list(self.default_xyz)
        self.on_leg_selected(self.current_leg)

    # ---------------------------------------------------------------
    # Per-leg computation
    # ---------------------------------------------------------------
    def compute_leg(self, name, body_length, body_width):
        """Solves IK/FK for one leg and returns everything needed to render
        and validate it, with points already transformed into world space."""
        solver = self.leg_solvers[name]
        x, y, z = self.leg_targets[name]
        yaw = LEG_MOUNT_YAW_DEG[name]
        mount_x, mount_z = leg_mount_point(name, body_length, body_width)

        t1, t2, t3 = solver.solve(x, y, z)
        result = {
            'name': name, 'target_local': (x, y, z), 'mount_world': (mount_x, mount_z),
            'yaw': yaw, 'angles': (t1, t2, t3), 'reachable': t1 is not None,
            'points_world': None, 'points_local': None,
        }
        if t1 is None:
            return result

        points_local = solver.forward(t1, t2, t3)
        points_world = []
        for p in points_local:
            wx_local, wz_local = rotate_xz(p[0], p[2], yaw)
            points_world.append(np.array([mount_x + wx_local, p[1], mount_z + wz_local]))
        result['points_local'] = points_local
        result['points_world'] = points_world
        return result

    # ---------------------------------------------------------------
    # Drawing helpers
    # ---------------------------------------------------------------
    def draw_body(self, length, width, thickness=14):
        hl, hw = length / 2.0, width / 2.0
        rx = [hl, hl, -hl, -hl]
        rz = [hw, -hw, -hw, hw]
        bottom = [(x, 0.0, z) for x, z in zip(rx, rz)]
        top = [(x, thickness, z) for x, z in zip(rx, rz)]

        def to_mpl(pt):
            x, y, z = pt
            return (x, z, y)  # mpl (X, Y, Z) = robot (X, Z, Y)

        bottom_m = [to_mpl(p) for p in bottom]
        top_m = [to_mpl(p) for p in top]
        faces = [bottom_m, top_m]
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom_m[i], bottom_m[j], top_m[j], top_m[i]])

        body_poly = Poly3DCollection(faces, facecolor='#5a6472', edgecolor='#9aa5b1',
                                      alpha=0.55, linewidths=1)
        self.ax.add_collection3d(body_poly)

    def draw_transparent_ground(self, half_extent=220):
        xx, zz = np.meshgrid(np.linspace(-half_extent, half_extent, 2),
                              np.linspace(-half_extent, half_extent, 2))
        yy = np.full_like(xx, -self.leg_solvers[self.current_leg].G_C)
        self.ax.plot_surface(xx, zz, yy, alpha=0.08, color='blue')

    def draw_workspace_shell(self, hip_world_xyz, solver):
        r_min = abs(solver.Femur - solver.Tibia)
        r_max = solver.Femur + solver.Tibia
        u = np.linspace(0, 2 * np.pi, 18)
        v = np.linspace(0, np.pi, 10)
        cx, cy, cz = hip_world_xyz
        for radius, color, alpha in ((r_max, '#7fc7ff', 0.06), (r_min, '#ffb37f', 0.09)):
            xs = cx + radius * np.outer(np.cos(u), np.sin(v))
            zs = cz + radius * np.outer(np.sin(u), np.sin(v))
            ys = cy + radius * np.outer(np.ones_like(u), np.cos(v))
            self.ax.plot_wireframe(xs, zs, ys, color=color, alpha=alpha, linewidth=0.5)

    def update_plot(self):
        body_length = self.spin_length.value()
        body_width = self.spin_width.value()

        leg_results = {name: self.compute_leg(name, body_length, body_width) for name in LEG_NAMES}
        selected = leg_results[self.current_leg]

        # -- Angle readout for the selected leg --
        t1, t2, t3 = selected['angles']
        if selected['reachable']:
            self.lbl_theta1.setText(f"Theta 1 (Coxa): {t1:.2f}\u00b0")
            self.lbl_theta2.setText(f"Theta 2 (Femur): {t2:.2f}\u00b0")
            self.lbl_theta3.setText(f"Theta 3 (Tibia): {t3:.2f}\u00b0")
        else:
            self.lbl_theta1.setText("Theta 1 (Coxa): UNREACHABLE")
            self.lbl_theta2.setText("Theta 2 (Femur): UNREACHABLE")
            self.lbl_theta3.setText("Theta 3 (Tibia): UNREACHABLE")

        # -- Status overview across all 4 legs --
        parts = []
        for name in LEG_NAMES:
            ok = leg_results[name]['reachable']
            color = '#6bff95' if ok else '#ff6b6b'
            text = 'OK' if ok else 'UNREACHABLE'
            weight = 'bold' if name == self.current_leg else 'normal'
            parts.append(f"<span style='color:{color}; font-weight:{weight};'>{name}: {text}</span>")
        self.lbl_status_overview.setText("&nbsp;&nbsp;".join(parts))

        # -- FK validation for the selected leg (in its own local frame) --
        if selected['reachable'] and self.check_validation.isChecked():
            solver = self.leg_solvers[self.current_leg]
            p0, p1, p2, p3 = selected['points_local']
            x, y, z = selected['target_local']
            target_local = np.array([x, -(solver.G_C - y), z])

            l_coxa = np.linalg.norm(p1 - p0)
            l_femur = np.linalg.norm(p2 - p1)
            l_tibia = np.linalg.norm(p3 - p2)
            err_coxa = abs(l_coxa - solver.Coxa)
            err_femur = abs(l_femur - solver.Femur)
            err_tibia = abs(l_tibia - solver.Tibia)
            dist_error = np.linalg.norm(p3 - target_local)
            is_valid = (dist_error < 0.1) and (err_coxa < 0.1) and (err_femur < 0.1) and (err_tibia < 0.1)

            status_text = "STATUS: " + ("VALID \u2713" if is_valid else "INVALID \u2717") + "\n"
            status_text += f"Target  XYZ: ({target_local[0]:.1f}, {target_local[1]:.1f}, {target_local[2]:.1f})\n"
            status_text += f"Final   XYZ: ({p3[0]:.1f}, {p3[1]:.1f}, {p3[2]:.1f}) | Err: {dist_error:.3f}\n"
            status_text += f"Coxa  Stick: {l_coxa:.1f} (Conf: {solver.Coxa}) | Err: {err_coxa:.3f}\n"
            status_text += f"Femur Stick: {l_femur:.1f} (Conf: {solver.Femur}) | Err: {err_femur:.3f}\n"
            status_text += f"Tibia Stick: {l_tibia:.1f} (Conf: {solver.Tibia}) | Err: {err_tibia:.3f}"
            self.lbl_validation.setText(status_text)
            self.lbl_validation.setStyleSheet(
                "font-family: monospace; color: #6bff95;" if is_valid else "font-family: monospace; color: #ff6b6b;")
            self.lbl_validation.show()
        elif not self.check_validation.isChecked():
            self.lbl_validation.hide()
        else:
            self.lbl_validation.setText("Selected leg's target is out of reach \u2014 nothing to validate.")
            self.lbl_validation.setStyleSheet("font-family: monospace; color: #ff6b6b;")
            self.lbl_validation.show()

        # -- 3D rendering --
        self.ax.clear()
        self.ax.set_facecolor("#232629")
        self.draw_transparent_ground()
        self.draw_body(body_length, body_width)

        if self.check_workspace.isChecked():
            mount_x, mount_z = selected['mount_world']
            self.draw_workspace_shell((mount_x, 0.0, mount_z), self.leg_solvers[self.current_leg])

        for name in LEG_NAMES:
            res = leg_results[name]
            color = LEG_COLORS[name]
            is_selected = (name == self.current_leg)
            mount_x, mount_z = res['mount_world']

            # Hip marker (always drawn, even if unreachable)
            self.ax.plot([mount_x], [mount_z], [0], 'o', color=color,
                         markersize=9 if is_selected else 6,
                         alpha=1.0 if is_selected else 0.6)
            self.ax.text(mount_x, mount_z, 8, name, color=color, fontsize=9)

            if not res['reachable']:
                continue

            pw = res['points_world']
            xs = [p[0] for p in pw]
            ys = [p[2] for p in pw]  # robot Z -> mpl Y
            zs = [p[1] for p in pw]  # robot Y -> mpl Z

            self.ax.plot(xs, ys, zs, '-o', color=color,
                         linewidth=4 if is_selected else 2.2,
                         markersize=7 if is_selected else 4,
                         alpha=1.0 if is_selected else 0.75)
            self.ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'o', color='#ff3b3b',
                         markersize=9 if is_selected else 5,
                         alpha=1.0 if is_selected else 0.7)

        self.finalize_axes()
        self.canvas.draw()

    def finalize_axes(self):
        ext_xz = max(220, self.body_length_extent())
        x_min, x_max = -ext_xz, ext_xz
        y_min, y_max = -ext_xz, ext_xz
        z_min, z_max = -160, 70

        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])
        self.ax.set_zlim([z_min, z_max])
        self.ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

        self.ax.set_xlabel('Robot X (fwd/back)', color='#cccccc')
        self.ax.set_ylabel('Robot Z (left/right)', color='#cccccc')
        self.ax.set_zlabel('Robot Y (height)', color='#cccccc')
        self.ax.tick_params(colors='#999999')
        self.ax.view_init(elev=24, azim=50)

    def body_length_extent(self):
        # Rough bound so the view stays framed as body size / leg reach change
        max_reach = 150 + 55 + 30  # slider x max + femur + tibia
        return max(self.spin_length.value(), self.spin_width.value()) / 2.0 + max_reach


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())