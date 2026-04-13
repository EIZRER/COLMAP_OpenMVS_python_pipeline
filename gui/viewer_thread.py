from PyQt5.QtCore import QThread, pyqtSignal

from utils.eval_chamfer import run_evaluation


class ViewerThread(QThread):
    """Background thread that computes Chamfer Distance metrics.

    Signals
    -------
    finished(metrics: dict, mesh_path: str)
        Emitted when evaluation completes successfully.
    error(message: str)
        Emitted when evaluation fails.
    log(message: str)
        Forwarded progress messages from the evaluation routines.
    """

    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, output_dir: str, n_samples: int = 100_000):
        super().__init__()
        self.output_dir = output_dir
        self.n_samples = n_samples

    def run(self):
        try:
            metrics = run_evaluation(
                self.output_dir,
                n_samples=self.n_samples,
                log_callback=self.log.emit,
            )
            mesh_path = metrics.get("mesh_path", "")
            self.finished.emit(metrics, mesh_path)
        except Exception as exc:
            self.error.emit(str(exc))
