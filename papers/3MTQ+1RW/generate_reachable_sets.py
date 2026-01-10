import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# ==========================================
# 1. Math & Geometry Utilities
# ==========================================
class GeometryUtils:
    @staticmethod
    def rotation_matrix(phi, theta, psi):
        """
        Creates a 3-2-1 Euler Angle Rotation Matrix (Body to Inertial).
        """
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        Rx = np.array([[1, 0, 0], [0, c_phi, -s_phi], [0, s_phi, c_phi]])
        Ry = np.array([[c_theta, 0, s_theta], [0, 1, 0], [-s_theta, 0, c_theta]])
        Rz = np.array([[c_psi, -s_psi, 0], [s_psi, c_psi, 0], [0, 0, 1]])

        return Rz @ Ry @ Rx

    @staticmethod
    def sph2cart(r, az, el):
        return r * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])

    @staticmethod
    def minkowski_sum(points_A, points_B):
        """
        Calculates the vertex cloud of the Minkowski sum of two polytopes.
        A + B = {a + b | a in A, b in B}
        """
        return (points_A[:, None, :] + points_B[None, :, :]).reshape(-1, 3)

# ==========================================
# 2. Physics & Actuator Modeling
# ==========================================
class ActuatorModel:
    @staticmethod
    def get_mtq_torque_envelope(B_body, m_max):
        """
        Returns the vertices of the torque polygon achievable by a 3-axis MTQ.
        Torque = m x B.
        """
        if np.linalg.norm(B_body) < 1e-15:
            return np.zeros((1, 3))
            
        corners = np.array(np.meshgrid([-m_max, m_max], 
                                       [-m_max, m_max], 
                                       [-m_max, m_max])).T.reshape(-1, 3)
        
        torques = np.cross(corners, B_body)
        return torques

    @staticmethod
    def get_rw_torque_envelope(axis_body, u_max):
        """
        Returns vertices (endpoints) of the torque line for a single Reaction Wheel.
        """
        direction = np.array(axis_body, dtype=float)
        direction /= (np.linalg.norm(direction) + 1e-16)
        
        return np.array([-u_max * direction, u_max * direction])

# ==========================================
# 3. Visualization Engine
# ==========================================
class Visualizer:
    def __init__(self, ax):
        self.ax = ax
        self.artists = []
        
        # Style settings
        self.ax.set_xlabel("X (Inertial)")
        self.ax.set_ylabel("Y (Inertial)")
        self.ax.set_zlabel("Z (Inertial)")
        self.ax.set_box_aspect([1, 1, 1])
        
        # Set a default initial view limit
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

    def clear(self):
        """Remove all dynamically drawn elements."""
        for artist in self.artists:
            try:
                artist.remove()
            except ValueError:
                pass 
        self.artists = []

    def draw_vector(self, origin, vector, color, lw=2, label=None):
        if np.linalg.norm(vector) < 1e-9: return
        line = self.ax.plot(
            [origin[0], origin[0] + vector[0]],
            [origin[1], origin[1] + vector[1]],
            [origin[2], origin[2] + vector[2]],
            color=color, lw=lw, label=label
        )[0]
        self.artists.append(line)

    def draw_convex_hull(self, points, color, alpha=0.2):
        if len(points) < 2: return
        pts = np.unique(points, axis=0)
        if len(pts) < 2: return

        # Center the points
        center = pts.mean(axis=0)
        u, s, vh = np.linalg.svd(pts - center)
        
        # Rank determination
        rank = np.sum(s > 1e-10 * s[0]) if s[0] > 0 else 0

        if rank <= 1:
            # Draw Line
            proj = (pts - center) @ vh[0]
            p_start = pts[np.argmin(proj)]
            p_end = pts[np.argmax(proj)]
            l = self.ax.plot(*zip(p_start, p_end), color=color, lw=4, alpha=0.8)[0]
            self.artists.append(l)

        elif rank == 2:
            # Draw Flat Polygon in 3D
            pts_2d = (pts - center) @ vh[:2].T
            hull = ConvexHull(pts_2d)
            verts_3d = pts[hull.vertices]
            poly = Poly3DCollection([verts_3d], facecolors=color, edgecolors=color, alpha=alpha)
            self.ax.add_collection3d(poly)
            self.artists.append(poly)

        else:
            # Draw 3D Volume
            try:
                hull = ConvexHull(pts, qhull_options='QJ')
                faces = [pts[s] for s in hull.simplices]
                poly = Poly3DCollection(faces, facecolors=color, edgecolors=color, alpha=alpha, linewidths=0.5)
                self.ax.add_collection3d(poly)
                self.artists.append(poly)
            except Exception as e:
                print(f"Hull Error: {e}")

# ==========================================
# 4. Main Application Controller
# ==========================================
class TorqueAnalysisApp:
    def __init__(self):
        # --- State ---
        self.state = {
            'phi': 0.0, 'theta': 0.0, 'psi': 0.0,       # Attitude
            'B_az': 0.0, 'B_el': 0.0, 'B_mag_log': -5,  # Env
            'mtq_m': 1.0, 'rw_u': 0.05,                 # Actuator Specs
            'rw_axis': 0,                               # 0=x, 1=y, 2=z
            'scale_log': 1.0,                           # Viz
            'show_mtq': True, 'show_rw': True, 'show_sum': True,
            'show_frames': True
        }

        # --- Window 1: Visualization ---
        self.fig_viz = plt.figure("Visualization", figsize=(8, 8))
        self.ax_viz = self.fig_viz.add_subplot(1, 1, 1, projection='3d')
        self.viz = Visualizer(self.ax_viz)
        
        # --- Window 2: Controls ---
        # We create a second figure entirely for controls
        self.fig_ctrl = plt.figure("Control Panel", figsize=(4, 8))
        self.widget_mgr = WidgetManager(self.fig_ctrl, [0.05, 0.95, 0.90, 0.90])
        
        self.setup_controls()
        self.update()

    def setup_controls(self):
        wm = self.widget_mgr
        
        wm.add_header("ATTITUDE (Euler 3-2-1)")
        wm.add_slider("Roll (rad)", -3.14, 3.14, self.state['phi'], lambda v: self.set_state('phi', v))
        wm.add_slider("Pitch (rad)", -1.57, 1.57, self.state['theta'], lambda v: self.set_state('theta', v))
        wm.add_slider("Yaw (rad)", -3.14, 3.14, self.state['psi'], lambda v: self.set_state('psi', v))

        wm.add_header("ENVIRONMENT")
        wm.add_slider("B Azimuth", 0, 6.28, self.state['B_az'], lambda v: self.set_state('B_az', v))
        wm.add_slider("B Elevation", -1.57, 1.57, self.state['B_el'], lambda v: self.set_state('B_el', v))
        wm.add_slider("B Mag (log T)", -7, -3, self.state['B_mag_log'], lambda v: self.set_state('B_mag_log', v))

        wm.add_header("ACTUATORS")
        wm.add_slider("MTQ Max (Am²)", 0.1, 5.0, self.state['mtq_m'], lambda v: self.set_state('mtq_m', v))
        wm.add_slider("RW Max (Nm)", 0.001, 0.2, self.state['rw_u'], lambda v: self.set_state('rw_u', v))
        
        def set_rw_axis(label):
            idx = {'X':0, 'Y':1, 'Z':2}[label]
            self.set_state('rw_axis', idx)
            
        wm.add_radio("RW Axis", ('X', 'Y', 'Z'), 0, set_rw_axis)

        wm.add_header("VISUALIZATION")
        wm.add_slider("Plot Scale (log)", 0, 6, self.state['scale_log'], lambda v: self.set_state('scale_log', v))
        
        # Separate Checkboxes for clear distinction
        wm.add_check("Show MTQ (Blue Plane)", self.state['show_mtq'], lambda v: self.set_state('show_mtq', not self.state['show_mtq']), color='blue')
        wm.add_check("Show RW (Red Line)", self.state['show_rw'], lambda v: self.set_state('show_rw', not self.state['show_rw']), color='red')
        wm.add_check("Show Combined (Green Vol)", self.state['show_sum'], lambda v: self.set_state('show_sum', not self.state['show_sum']), color='green')
        wm.add_check("Show Frames", self.state['show_frames'], lambda v: self.set_state('show_frames', not self.state['show_frames']))

    def set_state(self, key, value):
        self.state[key] = value
        self.update()

    def update(self):
        self.viz.clear()
        
        # 1. Compute Physics Transforms
        R_body_to_inertial = GeometryUtils.rotation_matrix(self.state['phi'], self.state['theta'], self.state['psi'])
        
        # B-field (Inertial Frame)
        B_mag = 10**self.state['B_mag_log']
        B_inertial = GeometryUtils.sph2cart(B_mag, self.state['B_az'], self.state['B_el'])
        
        # B-field (Body Frame) for torque calc
        B_body = R_body_to_inertial.T @ B_inertial

        # 2. Compute Torques (in Body Frame)
        T_mtq_body = ActuatorModel.get_mtq_torque_envelope(B_body, self.state['mtq_m'])
        
        rw_vec = np.zeros(3)
        rw_vec[self.state['rw_axis']] = 1.0
        T_rw_body = ActuatorModel.get_rw_torque_envelope(rw_vec, self.state['rw_u'])

        # 3. Transform Torques to Inertial Frame for Plotting
        T_mtq_in = (R_body_to_inertial @ T_mtq_body.T).T
        T_rw_in = (R_body_to_inertial @ T_rw_body.T).T
        
        # Minkowski Sum (Total Capacity)
        T_sum_in = GeometryUtils.minkowski_sum(T_mtq_in, T_rw_in)

        # 4. Scaling for Visualization
        scale_gain = 10**self.state['scale_log']
        
        # 5. Drawing
        if self.state['show_frames']:
            # Inertial (Black)
            self.viz.draw_vector([0,0,0], [0.8,0,0], 'k', lw=1)
            self.viz.draw_vector([0,0,0], [0,0.8,0], 'k', lw=1)
            self.viz.draw_vector([0,0,0], [0,0,0.8], 'k', lw=1)
            
            # Body (RGB) transformed to Inertial
            bx = R_body_to_inertial[:, 0] * 0.5
            by = R_body_to_inertial[:, 1] * 0.5
            bz = R_body_to_inertial[:, 2] * 0.5
            self.viz.draw_vector([0,0,0], bx, 'r', lw=2)
            self.viz.draw_vector([0,0,0], by, 'g', lw=2)
            self.viz.draw_vector([0,0,0], bz, 'b', lw=2)

        # B-Field Vector
        B_dir = B_inertial / np.linalg.norm(B_inertial)
        self.viz.draw_vector([0,0,0], B_dir, 'cyan', lw=3)

        # Torque Envelopes
        if self.state['show_mtq']:
            self.viz.draw_convex_hull(T_mtq_in * scale_gain, 'blue', alpha=0.15)
        
        if self.state['show_rw']:
            self.viz.draw_convex_hull(T_rw_in * scale_gain, 'red', alpha=0.4)
            
        if self.state['show_sum']:
            self.viz.draw_convex_hull(T_sum_in * scale_gain, 'green', alpha=0.15)

        self.ax_viz.set_title(f"Torque Visualization\nScale Gain: 1e{self.state['scale_log']:.1f}")
        
        # IMPORTANT: We do NOT reset axis limits here (xlim, ylim, zlim).
        # This allows the user to zoom in/out interactively without the plot snapping back.
        # We only call draw_idle() to update the geometry.
        self.fig_viz.canvas.draw_idle()

    def run(self):
        plt.show()

# ==========================================
# 5. UI Layout Helper
# ==========================================
class WidgetManager:
    """Helper to stack widgets vertically without magic number math everywhere."""
    def __init__(self, fig, bbox):
        # bbox = [left, top, width, height_available]
        self.fig = fig
        self.left = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2]
        self.row_h = 0.03
        self.pad = 0.015
        self.widgets = [] # Keep refs

    def add_header(self, text):
        self.y -= (self.row_h + 0.02)
        self.fig.text(self.left + self.width/2, self.y, text, 
                      ha='center', va='center', fontweight='bold', fontsize=10)
        self.y -= 0.02

    def add_slider(self, label, vmin, vmax, init, callback):
        self.y -= self.row_h
        ax = self.fig.add_axes([self.left + 0.35 * self.width, self.y, 0.65 * self.width, self.row_h])
        s = Slider(ax, "", vmin, vmax, valinit=init)
        s.on_changed(callback)
        self.fig.text(self.left, self.y + self.row_h/2, label, va='center', fontsize=9)
        self.widgets.append(s)
        self.y -= self.pad

    def add_check(self, label, init, callback, color='black'):
        self.y -= (self.row_h + 0.01)
        ax = self.fig.add_axes([self.left, self.y, self.width, self.row_h])
        ax.set_frame_on(False)
        cb = CheckButtons(ax, [label], [init])
        
        # --- Matplotlib Version Compatibility Fix ---
        try:
            if hasattr(cb, 'labels'):
                cb.labels[0].set_color(color)
                cb.labels[0].set_fontweight('bold')
            if hasattr(cb, 'rectangles'):
                cb.rectangles[0].set_edgecolor(color)
            elif hasattr(cb, 'rects'): 
                cb.rects[0].set_edgecolor(color)
        except Exception:
            pass
        
        cb.on_clicked(callback)
        self.widgets.append(cb)
        self.y -= 0.01

    def add_radio(self, title, labels, active, callback):
        h = len(labels) * 0.03 + 0.02
        self.y -= h
        ax = self.fig.add_axes([self.left + 0.3 * self.width, self.y, 0.5 * self.width, h])
        ax.set_frame_on(False)
        self.fig.text(self.left, self.y + h/2, title, va='center', fontsize=9)
        
        rb = RadioButtons(ax, labels, active=active)
        rb.on_clicked(callback)
        self.widgets.append(rb)
        self.y -= self.pad

# ==========================================
# 6. Entry Point
# ==========================================
if __name__ == "__main__":
    app = TorqueAnalysisApp()
    app.run()