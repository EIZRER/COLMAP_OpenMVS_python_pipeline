import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar,
    QPlainTextEdit, QGroupBox, QCheckBox, QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from gui.worker_thread import WorkerThread
from core.paths import ROOT_DIR
from pipeline.pipeline import PhotogrammetryPipeline

CONFIG_PATH = str(ROOT_DIR / "config" / "config.yaml")
TOTAL_STEPS = 9


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photogrammetry Pipeline")
        self.setMinimumSize(820, 680)
        self.worker = None

        self._build_ui()
        self._load_config_to_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # Header
        title = QLabel("Photogrammetry Pipeline")
        title.setObjectName("title_label")
        subtitle = QLabel("COLMAP + OpenMVS · Sparse → Dense → Mesh → Texture")
        subtitle.setObjectName("subtitle_label")
        root.addWidget(title)
        root.addWidget(subtitle)

        # ── Paths ──
        path_group = QGroupBox("Input / Output")
        pg = QVBoxLayout(path_group)
        self.image_dir_edit = self._path_row(pg, "Image Folder:", "Select folder containing photos")
        self.workspace_dir_edit = self._path_row(pg, "Workspace:", "Where to store intermediate & output files")
        root.addWidget(path_group)

        # ── Settings ──
        settings_group = QGroupBox("Settings")
        sg = QHBoxLayout(settings_group)
        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setChecked(True)
        self.refine_check = QCheckBox("Refine Mesh")
        self.refine_check.setChecked(True)
        self.texture_check = QCheckBox("Texture Mesh")
        self.texture_check.setChecked(True)
        self.ml_features_check = QCheckBox("ML Feature Matching")
        self.ml_features_check.setChecked(False)
        sg.addWidget(self.gpu_check)
        sg.addSpacing(20)
        sg.addWidget(self.refine_check)
        sg.addSpacing(20)
        sg.addWidget(self.texture_check)
        sg.addSpacing(20)
        sg.addWidget(self.ml_features_check)
        sg.addStretch()
        root.addWidget(settings_group)

        # ── Executables ──
        exe_group = QGroupBox("Executables (edit config/config.yaml to change)")
        eg = QVBoxLayout(exe_group)
        self.colmap_exe_edit = self._exe_row(eg, "COLMAP:")
        self.interface_exe_edit = self._exe_row(eg, "InterfaceCOLMAP:")
        self.densify_exe_edit = self._exe_row(eg, "DensifyPointCloud:")
        root.addWidget(exe_group)

        # ── Progress ──
        prog_group = QGroupBox("Progress")
        pl = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, TOTAL_STEPS)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Ready.")
        self.status_label.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.progress_bar)
        pl.addWidget(self.status_label)
        root.addWidget(prog_group)

        # ── Log ──
        log_group = QGroupBox("Log")
        ll = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        ll.addWidget(self.log_view)
        root.addWidget(log_group, 1)

        # ── Buttons ──
        btn_row = QHBoxLayout()

        self.save_cfg_btn = QPushButton("💾  Save Config")
        self.save_cfg_btn.clicked.connect(self._save_config)

        self.clean_btn = QPushButton("🗑  Clean Workspace")
        self.clean_btn.setObjectName("clean_btn")
        self.clean_btn.clicked.connect(self._on_clean_workspace)
        self.clean_btn.setToolTip("Delete all intermediate files from the workspace folder")

        self.abort_btn = QPushButton("⛔  Abort")
        self.abort_btn.setObjectName("abort_btn")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._on_abort)

        self.open_btn = QPushButton("📂  Open Output Folder")
        self.open_btn.setObjectName("open_btn")
        self.open_btn.setVisible(False)
        self.open_btn.clicked.connect(self._on_open_output)

        self.start_btn = QPushButton("▶  Start Reconstruction")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self._on_start)

        btn_row.addWidget(self.save_cfg_btn)
        btn_row.addWidget(self.clean_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.abort_btn)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.start_btn)
        root.addLayout(btn_row)

    def _path_row(self, layout, label_text, placeholder):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(110)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(lambda _, e=edit: self._browse_folder(e))
        row.addWidget(lbl)
        row.addWidget(edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)
        return edit

    def _exe_row(self, layout, label_text):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(160)
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setStyleSheet("color: #6c7086;")
        row.addWidget(lbl)
        row.addWidget(edit)
        layout.addLayout(row)
        return edit

    # ── Config ───────────────────────────────────────────────────────────────

    def _load_config_to_ui(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            exes = cfg.get("executables", {})
            self.colmap_exe_edit.setText(exes.get("colmap", ""))
            self.interface_exe_edit.setText(exes.get("interface_colmap", ""))
            self.densify_exe_edit.setText(exes.get("densify", ""))
            paths = cfg.get("paths", {})
            self.workspace_dir_edit.setText(str(ROOT_DIR / paths.get("workspace_dir", "data/workspace")))
            settings = cfg.get("settings", {})
            self.gpu_check.setChecked(settings.get("use_gpu", True))
            self.refine_check.setChecked(settings.get("run_refine", True))
            self.texture_check.setChecked(settings.get("run_texture", True))
            self.ml_features_check.setChecked(settings.get("use_ml_features", False))
        except Exception as e:
            self._log(f"[WARN] Could not load config: {e}")

    def _save_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            cfg["settings"]["use_gpu"] = self.gpu_check.isChecked()
            cfg["settings"]["run_refine"] = self.refine_check.isChecked()
            cfg["settings"]["run_texture"] = self.texture_check.isChecked()
            cfg["settings"]["use_ml_features"] = self.ml_features_check.isChecked()
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            self._log("Config saved.")
        except Exception as e:
            self._log(f"[ERROR] Could not save config: {e}")

    # ── File Dialogs ──────────────────────────────────────────────────────────

    def _browse_folder(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            edit.setText(d)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _on_start(self):
        image_dir = self.image_dir_edit.text().strip()
        workspace_dir = self.workspace_dir_edit.text().strip()

        if not image_dir or not Path(image_dir).is_dir():
            QMessageBox.warning(self, "Missing Input", "Please select a valid image folder.")
            return
        if not workspace_dir:
            QMessageBox.warning(self, "Missing Workspace", "Please select a workspace directory.")
            return

        self._save_config()
        self._reset_ui()
        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.clean_btn.setEnabled(False)
        self.open_btn.setVisible(False)
        self._log("Starting pipeline...")

        self.worker = WorkerThread(image_dir, workspace_dir, CONFIG_PATH)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.aborted.connect(self._on_aborted)
        self.worker.start()

    def _on_abort(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Abort Reconstruction",
                "Stop the reconstruction and clean the workspace?\n\nAll intermediate files will be deleted.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._log("\n⚠ Aborting — stopping current process...")
                self.abort_btn.setEnabled(False)
                self.worker.abort()

    def _on_clean_workspace(self):
        workspace_dir = self.workspace_dir_edit.text().strip()
        if not workspace_dir:
            QMessageBox.warning(self, "No Workspace", "Set a workspace directory first.")
            return

        reply = QMessageBox.question(
            self, "Clean Workspace",
            f"Delete all files in:\n{workspace_dir}\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                PhotogrammetryPipeline.clean_workspace(workspace_dir, self._log)
                self._log("✅ Workspace cleaned.")
                self.progress_bar.setValue(0)
                self.status_label.setText("Workspace cleaned. Ready.")
            except Exception as e:
                self._log(f"[ERROR] Failed to clean workspace: {e}")

    def _reset_ui(self):
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.log_view.clear()

    # ── Signals ───────────────────────────────────────────────────────────────

    def _on_progress(self, step, msg):
        self.progress_bar.setValue(step)
        self.status_label.setText(f"[{step}/{TOTAL_STEPS}] {msg}")
        self._log(f"── [{step}/{TOTAL_STEPS}] {msg}")

    def _on_finished(self, output_dir):
        self.progress_bar.setValue(TOTAL_STEPS)
        self.status_label.setText("✅ Reconstruction complete!")
        self._log(f"\n✅ Done! Output saved to: {output_dir}")
        self._restore_buttons()
        self.output_dir = output_dir
        self.open_btn.setVisible(True)

    def _on_error(self, msg):
        self.status_label.setText("❌ Error — see log for details")
        self._log(f"\n❌ ERROR: {msg}")
        self._restore_buttons()
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _on_aborted(self):
        self.status_label.setText("⛔ Aborted.")
        self._log("\n⛔ Reconstruction aborted.")
        self._restore_buttons()

        # Offer to clean workspace immediately after abort
        workspace_dir = self.workspace_dir_edit.text().strip()
        if workspace_dir:
            reply = QMessageBox.question(
                self, "Clean Workspace?",
                "Reconstruction was aborted.\n\nClean the workspace now to remove partial files?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    PhotogrammetryPipeline.clean_workspace(workspace_dir, self._log)
                    self._log("🗑  Workspace cleaned.")
                    self.progress_bar.setValue(0)
                    self.status_label.setText("Aborted & cleaned. Ready.")
                except Exception as e:
                    self._log(f"[ERROR] Cleanup failed: {e}")

    def _restore_buttons(self):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.clean_btn.setEnabled(True)

    def _log(self, msg):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _on_open_output(self):
        path = getattr(self, "output_dir", None)
        if path and Path(path).exists():
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
