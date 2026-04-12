import threading
import shutil
import yaml
from pathlib import Path
from typing import Callable

from core.paths import ensure_workspace, get_workspace_paths
from core.logger import get_logger
from core.exceptions import ConfigError
from pipeline.colmap_runner import COLMAPRunner
from pipeline.openmvs_runner import OpenMVSRunner
from pipeline.model_converter import collect_outputs
from ml.ml_feature_pipeline import run_ml_feature_pipeline

logger = get_logger("pipeline")

TOTAL_STEPS = 9


def _find_best_reconstruction(sparse_dir: Path, log_callback: Callable = None) -> Path:
    """
    COLMAP mapper can produce multiple sub-reconstructions (0, 1, 2...).
    This picks the one with the most registered images by counting lines
    in images.txt (or falling back to images.bin file size).
    Returns the Path to the best sub-reconstruction folder.
    """
    candidates = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise RuntimeError(f"No reconstruction folders found in {sparse_dir}")

    if len(candidates) == 1:
        return candidates[0]

    best = candidates[0]
    best_count = 0

    for candidate in candidates:
        # Count registered images via images.bin size or images.txt line count
        images_txt = candidate / "sparse" / "images.txt"
        images_bin = candidate / "images.bin"

        count = 0
        if images_txt.exists():
            # Each image in images.txt takes 2 lines; skip comment lines starting with #
            lines = [l for l in images_txt.read_text().splitlines() if not l.startswith("#") and l.strip()]
            count = len(lines) // 2
        elif images_bin.exists():
            count = images_bin.stat().st_size  # proxy: bigger = more images

        if log_callback:
            log_callback(f"  Reconstruction {candidate.name}: {count} images registered")

        if count > best_count:
            best_count = count
            best = candidate

    if log_callback:
        log_callback(f"  → Selected reconstruction: {best.name} ({best_count} images)")

    return best


class PhotogrammetryPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self._validate_executables()

        exes = self.cfg["executables"]
        settings = self.cfg["settings"]

        self.colmap = COLMAPRunner(colmap_exe=exes["colmap"], use_gpu=settings.get("use_gpu", True))
        self.openmvs = OpenMVSRunner(exes)
        self.run_refine = settings.get("run_refine", True)
        self.run_texture = settings.get("run_texture", True)
        self.use_ml_features = settings.get("use_ml_features", False)

    def _validate_executables(self):
        for name, path in self.cfg["executables"].items():
            if not Path(path).exists():
                raise ConfigError(f"Executable not found: {name} → {path}")

    @staticmethod
    def clean_workspace(workspace_dir: str, log_callback: Callable = None):
        """Delete and recreate the workspace — wipes all intermediate files."""
        ws = Path(workspace_dir)
        if ws.exists():
            shutil.rmtree(ws)
            if log_callback:
                log_callback(f"🗑  Cleaned workspace: {ws}")
        ws.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_dir: str,
        workspace_dir: str,
        progress_callback: Callable[[int, str], None] = None,
        log_callback: Callable[[str], None] = None,
        abort_event: threading.Event = None,
    ):
        def progress(step: int, msg: str):
            logger.info(f"[{step}/{TOTAL_STEPS}] {msg}")
            if progress_callback:
                progress_callback(step, msg)

        def log(msg: str):
            logger.debug(msg)
            if log_callback:
                log_callback(msg)

        def check_abort():
            if abort_event and abort_event.is_set():
                raise RuntimeError("ABORTED")

        image_dir = str(Path(image_dir).resolve())
        paths = ensure_workspace(workspace_dir)

        # ── 1. Create Database ────────────────────────────────────────────────
        check_abort()
        progress(1, "Creating Database")
        self.colmap.create_database(paths["database"], log, abort_event)

        # ── 2. Feature Extraction ─────────────────────────────────────────────
        # Always run COLMAP feature_extractor to populate cameras/images tables.
        # When ML features are enabled the keypoints/descriptors will be
        # overwritten by SuperPoint in step 3.
        check_abort()
        label = "Extracting Features (SuperPoint)" if self.use_ml_features else "Extracting Features (SIFT)"
        progress(2, label)
        self.colmap.extract_features(paths["database"], image_dir, log, abort_event)

        # ── 3. Feature Matching ───────────────────────────────────────────────
        check_abort()
        if self.use_ml_features:
            progress(3, "Matching Features (LightGlue)")
            run_ml_feature_pipeline(
                image_dir=image_dir,
                database_path=str(paths["database"]),
                use_gpu=self.colmap.use_gpu,
                max_keypoints=2048,
                log_callback=log,
                abort_event=abort_event,
            )
        else:
            progress(3, "Matching Features (SIFT)")
            self.colmap.match_features(paths["database"], log, abort_event)

        # ── 4. Mapper ─────────────────────────────────────────────────────────
        check_abort()
        progress(4, "Running Mapper (Sparse Reconstruction)")
        self.colmap.run_mapper(paths["database"], image_dir, paths["sparse"], log, abort_event)

        # ── 4b. Pick best reconstruction ──────────────────────────────────────
        # COLMAP may produce multiple sub-reconstructions (0, 1, 2...) when it
        # can't register all images in one pass. Always use the largest one.
        log("Selecting best reconstruction...")
        best_recon = _find_best_reconstruction(paths["sparse"], log)
        sparse_txt_path = best_recon / "sparse"

        # ── 5. Convert binary → TXT ───────────────────────────────────────────
        check_abort()
        progress(5, "Converting COLMAP Model to TXT")
        self.colmap.convert_model_to_txt(best_recon, sparse_txt_path, log, abort_event)

        # ── 6. Convert COLMAP → OpenMVS ──────────────────────────────────────
        # Pass the reconstruction root (e.g. sparse/1) — InterfaceCOLMAP
        # internally appends /sparse/ to find the .txt files.
        check_abort()
        progress(6, "Converting to OpenMVS Format")
        self.openmvs.convert_to_mvs(best_recon, paths["scene"], image_dir, paths["mvs"], log, abort_event)

        # ── 7. Densify ────────────────────────────────────────────────────────
        check_abort()
        progress(7, "Densifying Point Cloud")
        self.openmvs.densify_point_cloud(paths["scene"], paths["mvs"], log, abort_event)

        # ── 8. Reconstruct Mesh ───────────────────────────────────────────────
        check_abort()
        progress(8, "Reconstructing Mesh")
        self.openmvs.reconstruct_mesh(paths["scene_dense"], paths["mvs"], log, abort_event)

        # ── 9. Refine + Texture ───────────────────────────────────────────────
        if self.run_refine:
            check_abort()
            progress(9, "Refining Mesh")
            self.openmvs.refine_mesh(paths["scene_mesh"], paths["mvs"], log, abort_event)

        if self.run_texture:
            check_abort()
            if not self.run_refine:
                progress(9, "Texturing Mesh")
            input_for_texture = paths["scene_refine"] if self.run_refine else paths["scene_mesh"]
            self.openmvs.texture_mesh(input_for_texture, paths["mvs"], log, abort_event)

        if not self.run_refine and not self.run_texture:
            progress(9, "Refine + Texture skipped")

        # ── Collect outputs ───────────────────────────────────────────────────
        log("Collecting output files...")
        copied = collect_outputs(str(paths["mvs"]), str(paths["output"]))
        log(f"Done! {len(copied)} file(s) saved to: {paths['output']}")
        return str(paths["output"])
