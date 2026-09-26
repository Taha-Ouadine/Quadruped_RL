import sys
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QFormLayout,
                             QCheckBox, QSpinBox, QPushButton)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
QSpinBox {
    background: #2c2f33; border: 1px solid #444; border-radius: 4px;
    padding: 2px 4px; color: #e8e8e8;
}
QPushButton {
    background: #3a3d40; border: 1px solid #555; border-radius: 5px; padding: 6px;
}
QPushButton:hover { background: #48505a; }
QCheckBox { color: #e8e8e8; }
"""


class QuadrupedLegIK:
    """Inverse kinematics solver for a single 3-DOF (coxa/femur/tibia) leg."""

    def __init__(self):
        # Robot physical parameters based on the Arduino code
        self.Coxa = 50.00
        self.Femur = 55.00
        self.Tibia = 30.00
        self.G_C = 50.00

    def solve(self, x, y, z):
        """Calculates joint angles based on the Inverse Kinematics equations.

        Returns (theta_1, theta_2, theta_3) in degrees, or (None, None, None)
        if the target point is outside the leg's reachable workspace.
        """
        try:
            D = math.sqrt(x ** 2 + z ** 2)

            # Theta 1 (Coxa / yaw angle). atan2 handles every quadrant,
            # including z == 0, without any special-casing.
            theta_1 = math.degrees(math.atan2(x, z))

            d = D - self.Coxa
            y_offset = self.G_C - y
            R = math.sqrt(d ** 2 + y_offset ** 2)

            # Check for unreachable coordinates (outstretched or folded past limits)
            if R > (self.Femur + self.Tibia) or R < abs(self.Femur - self.Tibia):
                return None, None, None

            # Guard against division by zero when the target sits exactly on the joint
            if R < 1e-9:
                return None, None, None

            # Alpha 1 & 2 (clip guards against tiny floating-point overshoot past +/-1)
            alpha_1 = math.degrees(math.acos(np.clip(y_offset / R, -1.0, 1.0)))
            cos_alpha_2 = (self.Femur ** 2 + R ** 2 - self.Tibia ** 2) / (2 * self.Femur * R)
            alpha_2 = math.degrees(math.acos(np.clip(cos_alpha_2, -1.0, 1.0)))

            # Theta 2 (Femur Angle)
            theta_2 = alpha_1 + alpha_2

            # Theta 3 (Tibia Angle)
            cos_theta_3 = (self.Femur ** 2 + self.Tibia ** 2 - R ** 2) / (2 * self.Femur * self.Tibia)
            theta_3 = math.degrees(math.acos(np.clip(cos_theta_3, -1.0, 1.0)))

            return theta_1, theta_2, theta_3

        except (ValueError, ZeroDivisionError):
            # Math domain error / degenerate geometry (unreachable point)
            return None, None, None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quadruped IK Visualizer + FK Validation")
        self.resize(1150, 720)
        self.setStyleSheet(DARK_STYLESHEET)

        self.ik_solver = QuadrupedLegIK()
        self.default_xyz = (70, 0, 40)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Panel (Controls)
        control_layout = QVBoxLayout()
        layout.addLayout(control_layout, 1)

        # Sliders + spin boxes for X, Y, Z
        coord_group = QGroupBox("End-Effector Coordinates (X, Y, Z)")
        coord_form = QFormLayout()

        self.slider_x, self.spin_x = self.create_slider_row(20, 150, self.default_xyz[0])
        self.slider_y, self.spin_y = self.create_slider_row(-30, 50, self.default_xyz[1])
        self.slider_z, self.spin_z = self.create_slider_row(-100, 100, self.default_xyz[2])

        coord_form.addRow(self.wrap_row("X (mm):", self.slider_x, self.spin_x))
        coord_form.addRow(self.wrap_row("Y (mm):", self.slider_y, self.spin_y))
        coord_form.addRow(self.wrap_row("Z (mm):", self.slider_z, self.spin_z))
        coord_group.setLayout(coord_form)
        control_layout.addWidget(coord_group)

        reset_btn = QPushButton("Reset to Home")
        reset_btn.clicked.connect(self.reset_to_home)
        control_layout.addWidget(reset_btn)

        # Outputs for Joint Angles
        angle_group = QGroupBox("Calculated Joint Angles")
        angle_form = QFormLayout()

        self.lbl_theta1 = QLabel("Theta 1 (Coxa): -")
        self.lbl_theta2 = QLabel("Theta 2 (Femur): -")
        self.lbl_theta3 = QLabel("Theta 3 (Tibia): -")

        angle_form.addRow(self.lbl_theta1)
        angle_form.addRow(self.lbl_theta2)
        angle_form.addRow(self.lbl_theta3)
        angle_group.setLayout(angle_form)
        control_layout.addWidget(angle_group)

        # FK Validation Check
        val_group = QGroupBox("Forward Kinematics Validation")
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

        # Workspace shell toggle
        self.check_workspace = QCheckBox("Show reachable workspace shell")
        self.check_workspace.setChecked(True)
        control_layout.addWidget(self.check_workspace)

        control_layout.addStretch()

        # Right Panel (3D Plot)
        self.figure = Figure(facecolor="#232629")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, 2)

        # Connect signals
        self.slider_x.valueChanged.connect(self.update_plot)
        self.slider_y.valueChanged.connect(self.update_plot)
        self.slider_z.valueChanged.connect(self.update_plot)
        self.check_validation.stateChanged.connect(self.update_plot)
        self.check_workspace.stateChanged.connect(self.update_plot)

        # Initial draw
        self.update_plot()

    # ---------------------------------------------------------------
    # UI helpers
    # ---------------------------------------------------------------
    def create_slider_row(self, min_val, max_val, default_val):
        """Creates a slider and a spin box kept in sync with each other."""
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

    def reset_to_home(self):
        x, y, z = self.default_xyz
        self.slider_x.setValue(x)
        self.slider_y.setValue(y)
        self.slider_z.setValue(z)

    # ---------------------------------------------------------------
    # Drawing helpers
    # ---------------------------------------------------------------
    def draw_transparent_planes(self):
        """Draws transparent planes for XY, YZ, and XZ to help visualize the 3D space."""
        x_min, x_max = -100, 150
        y_min, y_max = -150, 50   # Robot Y (Height)
        z_min, z_max = -100, 150  # Robot Z

        # Note: Matplotlib 3D axes are mapped as:
        # ax X = Robot X, ax Y = Robot Z, ax Z = Robot Y (Height)

        xx_xy, yy_xy = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2))
        zz_xy = np.zeros_like(xx_xy)
        self.ax.plot_surface(xx_xy, zz_xy, yy_xy, alpha=0.12, color='blue')

        yy_yz, zz_yz = np.meshgrid(np.linspace(y_min, y_max, 2), np.linspace(z_min, z_max, 2))
        xx_yz = np.zeros_like(yy_yz)
        self.ax.plot_surface(xx_yz, zz_yz, yy_yz, alpha=0.12, color='red')

        xx_xz, zz_xz = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(z_min, z_max, 2))
        yy_xz = np.zeros_like(xx_xz)
        self.ax.plot_surface(xx_xz, zz_xz, yy_xz, alpha=0.12, color='green')

    def draw_workspace_shell(self, center_robot):
        """Draws translucent min/max reach spheres around the coxa joint (p1),
        showing the annular shell the foot can actually reach."""
        r_min = abs(self.ik_solver.Femur - self.ik_solver.Tibia)
        r_max = self.ik_solver.Femur + self.ik_solver.Tibia

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 12)

        cx, cy, cz = center_robot  # robot-frame coords (x, y, z)
        for radius, color, alpha in ((r_max, '#7fc7ff', 0.08), (r_min, '#ffb37f', 0.10)):
            xs = cx + radius * np.outer(np.cos(u), np.sin(v))
            zs = cz + radius * np.outer(np.sin(u), np.sin(v))   # robot Z -> mpl Y
            ys = cy + radius * np.outer(np.ones_like(u), np.cos(v))  # robot Y -> mpl Z
            self.ax.plot_wireframe(xs, zs, ys, color=color, alpha=alpha, linewidth=0.5)

    def update_plot(self):
        x = self.slider_x.value()
        y = self.slider_y.value()
        z = self.slider_z.value()

        t1, t2, t3 = self.ik_solver.solve(x, y, z)

        self.ax.clear()
        self.ax.set_facecolor("#232629")
        self.draw_transparent_planes()

        if t1 is None:
            self.lbl_theta1.setText("Theta 1 (Coxa): UNREACHABLE")
            self.lbl_theta2.setText("Theta 2 (Femur): UNREACHABLE")
            self.lbl_theta3.setText("Theta 3 (Tibia): UNREACHABLE")
            self.lbl_validation.setText("Target is physically out of reach.")
            self.lbl_validation.setStyleSheet("font-family: monospace; color: #ff6b6b;")
            self.finalize_axes()
            self.canvas.draw()
            return

        self.lbl_theta1.setText(f"Theta 1 (Coxa): {t1:.2f}°")
        self.lbl_theta2.setText(f"Theta 2 (Femur): {t2:.2f}°")
        self.lbl_theta3.setText(f"Theta 3 (Tibia): {t3:.2f}°")

        # -------------------------------------------------------------
        # FORWARD KINEMATICS (Building exact positions strictly from angles)
        # -------------------------------------------------------------
        t1_rad = math.radians(t1)
        t2_rad = math.radians(t2)
        t3_rad = math.radians(t3)

        u = np.array([math.sin(t1_rad), 0, math.cos(t1_rad)])  # Horizontal, pointing outward
        v = np.array([0, -1, 0])                                # Vertical, pointing downward

        p0 = np.array([0.0, 0.0, 0.0])
        p1 = p0 + self.ik_solver.Coxa * u
        p2 = p1 + self.ik_solver.Femur * (math.sin(t2_rad) * u + math.cos(t2_rad) * v)

        # Theta 3 is the interior angle, so the tibia bends backward relative to the femur.
        tibia_angle = t2_rad - (math.pi - t3_rad)
        p3 = p2 + self.ik_solver.Tibia * (math.sin(tibia_angle) * u + math.cos(tibia_angle) * v)

        # -------------------------------------------------------------
        # VALIDATION LOGIC
        # -------------------------------------------------------------
        target_pos = np.array([x, -(self.ik_solver.G_C - y), z])  # actual goal in robot coordinates

        if self.check_validation.isChecked():
            l_coxa = np.linalg.norm(p1 - p0)
            l_femur = np.linalg.norm(p2 - p1)
            l_tibia = np.linalg.norm(p3 - p2)

            err_coxa = abs(l_coxa - self.ik_solver.Coxa)
            err_femur = abs(l_femur - self.ik_solver.Femur)
            err_tibia = abs(l_tibia - self.ik_solver.Tibia)
            dist_error = np.linalg.norm(p3 - target_pos)

            is_valid = (dist_error < 0.1) and (err_coxa < 0.1) and (err_femur < 0.1) and (err_tibia < 0.1)

            status_text = "STATUS: " + ("VALID \u2713" if is_valid else "INVALID \u2717") + "\n"
            status_text += f"Target  XYZ: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})\n"
            status_text += f"Final   XYZ: ({p3[0]:.1f}, {p3[1]:.1f}, {p3[2]:.1f}) | Err: {dist_error:.3f}\n"
            status_text += f"Coxa  Stick: {l_coxa:.1f} (Conf: {self.ik_solver.Coxa}) | Err: {err_coxa:.3f}\n"
            status_text += f"Femur Stick: {l_femur:.1f} (Conf: {self.ik_solver.Femur}) | Err: {err_femur:.3f}\n"
            status_text += f"Tibia Stick: {l_tibia:.1f} (Conf: {self.ik_solver.Tibia}) | Err: {err_tibia:.3f}"

            self.lbl_validation.setText(status_text)
            self.lbl_validation.setStyleSheet(
                "font-family: monospace; color: #6bff95;" if is_valid else "font-family: monospace; color: #ff6b6b;")
            self.lbl_validation.show()
        else:
            self.lbl_validation.hide()

        # -------------------------------------------------------------
        # MATPLOTLIB RENDERING
        # -------------------------------------------------------------
        if self.check_workspace.isChecked():
            self.draw_workspace_shell(p1)

        xs = [p0[0], p1[0], p2[0], p3[0]]
        ys = [p0[2], p1[2], p2[2], p3[2]]  # Map Z to Y
        zs = [p0[1], p1[1], p2[1], p3[1]]  # Map Y to Z

        self.ax.plot(xs, ys, zs, '-o', color='#e8e8e8', linewidth=4, markersize=8)
        self.ax.plot([xs[0]], [ys[0]], [zs[0]], 'o', color='#7fc7ff', markersize=10, label="Robot Body (Origin)")
        self.ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'o', color='#ff6b6b', markersize=10, label="End Effector (Toe)")

        self.finalize_axes()
        self.canvas.draw()

    def finalize_axes(self):
        x_min, x_max = -100, 150
        y_min, y_max = -100, 150
        z_min, z_max = -150, 50

        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])
        self.ax.set_zlim([z_min, z_max])
        # True equal-unit aspect ratio so the leg isn't visually stretched
        self.ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

        self.ax.set_xlabel('Robot X Axis', color='#cccccc')
        self.ax.set_ylabel('Robot Z Axis', color='#cccccc')
        self.ax.set_zlabel('Robot Y Axis (Height)', color='#cccccc')
        self.ax.tick_params(colors='#999999')
        #self.ax.view_init(elev=20, azim=45)
        self.ax.legend(facecolor='#2c2f33', labelcolor='#e8e8e8')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())