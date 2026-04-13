"""
Chamfer Distance evaluation utilities for photogrammetry output quality.

Dependencies
------------
- trimesh  : mesh / point-cloud I/O and uniform surface sampling
- scipy    : KDTree nearest-neighbour search (already in requirements)
- PyQt5    : viewer window (already in requirements)
- PyOpenGL : OpenGL rendering inside the Qt widget (already installed)

Public API
----------
compute_metrics(ref_pts, rec_pts, log_callback) -> dict
run_evaluation(output_dir, n_samples, log_callback) -> dict
show_mesh_with_metrics(mesh_path, metrics) -> None   (non-blocking subprocess)
"""

import math
import multiprocessing
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core metric computation  (scipy KDTree, parallelised over all CPU cores)
# ---------------------------------------------------------------------------

def compute_metrics(
    ref_pts: np.ndarray,
    rec_pts: np.ndarray,
    log_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Bidirectional Chamfer Distance and RMSE via scipy KDTree.

    Parameters
    ----------
    ref_pts : (N, 3) ndarray  – reference point cloud (dense_point_cloud.ply)
    rec_pts : (M, 3) ndarray  – reconstructed points (sampled from mesh)
    log_callback : callable or None

    Returns
    -------
    dict with keys: chamfer_distance, rmse, n_ref, n_samples
    """
    from scipy.spatial import KDTree

    if log_callback:
        log_callback(
            f"[Eval] ref={len(ref_pts):,} pts  rec={len(rec_pts):,} pts — forward pass..."
        )

    tree_rec = KDTree(rec_pts)
    d_ref_to_rec, _ = tree_rec.query(ref_pts, workers=-1)   # ref → rec

    if log_callback:
        log_callback("[Eval] Backward pass...")

    tree_ref = KDTree(ref_pts)
    d_rec_to_ref, _ = tree_ref.query(rec_pts, workers=-1)   # rec → ref

    chamfer = float(d_ref_to_rec.mean() + d_rec_to_ref.mean())
    all_sq = np.concatenate([d_ref_to_rec ** 2, d_rec_to_ref ** 2])
    rmse = float(math.sqrt(all_sq.mean()))

    return {
        "chamfer_distance": chamfer,
        "rmse": rmse,
        "n_ref": len(ref_pts),
        "n_samples": len(rec_pts),
    }


# ---------------------------------------------------------------------------
# High-level pipeline evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    output_dir: str,
    n_samples: int = 100_000,
    log_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Load pipeline outputs, sample the mesh, compute and return metrics.

    Parameters
    ----------
    output_dir   : str – pipeline output directory
    n_samples    : int – points sampled uniformly from the mesh surface
    log_callback : callable or None

    Returns
    -------
    dict with keys: chamfer_distance, rmse, n_ref, n_samples, mesh_path
    """
    import trimesh

    out = Path(output_dir)
    pcd_path = out / "dense_point_cloud.ply"
    mesh_path = out / "textured_mesh.obj"

    if not pcd_path.exists():
        raise FileNotFoundError(f"Reference point cloud not found: {pcd_path}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    if log_callback:
        log_callback(f"[Eval] Loading reference cloud: {pcd_path.name}")
    ref_cloud = trimesh.load(str(pcd_path))
    ref_pts = np.asarray(ref_cloud.vertices)

    if log_callback:
        log_callback(f"[Eval] Loading mesh: {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), force="mesh")

    if log_callback:
        log_callback(f"[Eval] Sampling {n_samples:,} points from mesh surface...")
    rec_pts, _ = trimesh.sample.sample_surface(mesh, n_samples)

    metrics = compute_metrics(ref_pts, rec_pts, log_callback)
    metrics["mesh_path"] = str(mesh_path)

    if log_callback:
        log_callback(
            f"[Eval] Done — Chamfer: {metrics['chamfer_distance']:.6f} m  "
            f"RMSE: {metrics['rmse']:.6f} m"
        )
    return metrics


# ---------------------------------------------------------------------------
# Viewer — PyQt5 window with an embedded OpenGL mesh widget
# Text overlay is a plain QLabel child parented to the GL widget, which Qt
# composites on top automatically — no GL state manipulation needed.
# Runs in a daemon subprocess so the main Qt event loop is never blocked.
# ---------------------------------------------------------------------------

_MAX_DISPLAY_FACES = 300_000   # decimate above this to stay interactive


def _build_overlay_text(metrics: dict) -> str:
    cd = metrics.get("chamfer_distance", 0.0)
    rmse_val = metrics.get("rmse", 0.0)
    n = metrics.get("n_samples", 0)
    return (
        f"Chamfer Distance: {cd:.6f} m\n"
        f"RMSE:             {rmse_val:.6f} m\n"
        f"Sampled pts:      {n:,}"
    )


def _viewer_process(mesh_path: str, metrics: dict) -> None:
    """Subprocess entry point: PyQt5 + OpenGL mesh viewer."""
    import sys
    import numpy as np
    import trimesh

    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QOpenGLWidget
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QSurfaceFormat
    from OpenGL import GL

    # ── Load and prepare mesh ────────────────────────────────────────────────
    mesh = trimesh.load(mesh_path, force="mesh")
    if len(mesh.faces) > _MAX_DISPLAY_FACES:
        mesh = mesh.simplify_quadric_decimation(_MAX_DISPLAY_FACES)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    norms = np.asarray(mesh.vertex_normals, dtype=np.float32)

    # Normalize to unit box centred at origin for consistent camera distance
    center = verts.mean(axis=0)
    verts = verts - center
    scale = float(np.abs(verts).max()) or 1.0
    verts = np.ascontiguousarray(verts / scale, dtype=np.float32)

    flat_faces = np.ascontiguousarray(faces.ravel(), dtype=np.uint32)
    n_indices = flat_faces.size

    overlay_text = _build_overlay_text(metrics)

    # ── OpenGL widget ────────────────────────────────────────────────────────
    class MeshGLWidget(QOpenGLWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.rot_x = 20.0
            self.rot_y = -30.0
            self.zoom = 1.0
            self._last_pos = None
            self.setMinimumSize(900, 650)

        def initializeGL(self):
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 1.5, 2.0, 0.0])
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE,  [0.85, 0.85, 0.85, 1.0])
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT,  [0.20, 0.20, 0.20, 1.0])
            GL.glEnable(GL.GL_LIGHT1)
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, [-1.0, -0.5, -1.0, 0.0])
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE,  [0.25, 0.25, 0.30, 1.0])
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glEnable(GL.GL_NORMALIZE)
            GL.glClearColor(0.118, 0.118, 0.180, 1.0)   # Catppuccin base

        def resizeGL(self, w: int, h: int):
            GL.glViewport(0, 0, w, h)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            asp = w / h if h else 1.0
            near, far, fov = 0.01, 100.0, 45.0
            f = 1.0 / math.tan(math.radians(fov) / 2.0)
            # Column-major for glMultMatrixf
            GL.glMultMatrixf([
                f / asp, 0,  0,                            0,
                0,       f,  0,                            0,
                0,       0,  (far + near) / (near - far), -1,
                0,       0,  2 * far * near / (near - far), 0,
            ])
            GL.glMatrixMode(GL.GL_MODELVIEW)

        def paintGL(self):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glLoadIdentity()
            GL.glTranslatef(0.0, 0.0, -2.5 / self.zoom)
            GL.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot_y, 0.0, 1.0, 0.0)

            GL.glColor3f(0.537, 0.706, 0.980)   # #89b4fa (Catppuccin blue)

            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, verts)
            GL.glNormalPointer(GL.GL_FLOAT, 0, norms)
            GL.glDrawElements(GL.GL_TRIANGLES, n_indices, GL.GL_UNSIGNED_INT, flat_faces)
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

        def mousePressEvent(self, event):
            self._last_pos = event.pos()

        def mouseMoveEvent(self, event):
            if self._last_pos is not None:
                dx = event.x() - self._last_pos.x()
                dy = event.y() - self._last_pos.y()
                self.rot_y += dx * 0.5
                self.rot_x += dy * 0.5
                self._last_pos = event.pos()
                self.update()

        def mouseReleaseEvent(self, _event):
            self._last_pos = None

        def wheelEvent(self, event):
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else (1.0 / 1.15)
            self.zoom = max(0.05, min(self.zoom * factor, 50.0))
            self.update()

    # ── Main window ──────────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("3D Result — Photogrammetry")

    gl_widget = MeshGLWidget()

    # Metrics label parented to the GL widget — Qt composites it on top
    label = QLabel(overlay_text, gl_widget)
    label.setFont(QFont("Courier New", 10))
    label.setStyleSheet("""
        QLabel {
            color: #f5e0dc;
            background-color: rgba(30, 30, 46, 210);
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 8px 12px;
        }
    """)
    label.adjustSize()
    label.move(12, 12)
    label.setAttribute(Qt.WA_TransparentForMouseEvents)  # clicks pass through

    win = QMainWindow()
    win.setWindowTitle("🔷 3D Result — Photogrammetry")
    win.setCentralWidget(gl_widget)
    win.resize(1100, 750)
    win.show()

    sys.exit(app.exec_())


def show_mesh_with_metrics(mesh_path: str, metrics: dict) -> None:
    """Open the 3-D viewer in a separate daemon process (non-blocking)."""
    p = multiprocessing.Process(
        target=_viewer_process,
        args=(mesh_path, metrics),
        daemon=True,
    )
    p.start()
