"""Rich progress callback for Boltz predictions."""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from boltz.console import console


# Standard beta values for multi-beta mode
STANDARD_BETAS = [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75]


@dataclass
class PredictionConfig:
    """Configuration for prediction display."""

    model: str = "boltz2"
    seed: Optional[int] = None
    diffusion_samples: int = 1
    recycling_steps: int = 3
    sampling_steps: int = 200
    device: str = "cuda:0"
    output_dir: str = ""
    total_predictions: int = 0
    # Multi-beta support
    multi_beta: bool = False
    betas: List[float] = field(default_factory=list)
    current_beta: float = 0.0
    # Input/output paths for display
    input_path: str = ""
    output_path: str = ""


class BoltzProgressCallback(Callback):
    """Rich-based progress callback for Boltz predictions."""

    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        self.start_time: Optional[float] = None
        self.completed = 0
        self.failed = 0
        self.current_id = ""
        self._progress: Optional[Progress] = None
        self._task_id = None

    def _make_header(self) -> Panel:
        """Create header panel with configuration (1-column layout)."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")
        table.add_column(style="cyan")

        table.add_row("Model", self.config.model)
        table.add_row("Device", self.config.device)
        table.add_row("Seed", str(self.config.seed) if self.config.seed else "random")

        if self.config.current_beta != 0.0:
            table.add_row("β (uniform)", f"{self.config.current_beta:+.2f}")

        table.add_row("Diffusion", f"{self.config.diffusion_samples} samples")
        table.add_row("Recycling", f"{self.config.recycling_steps} steps")
        table.add_row("Sampling", f"{self.config.sampling_steps} steps")

        if self.config.input_path:
            table.add_row("Input", self.config.input_path)
        if self.config.output_path:
            table.add_row("Output", self.config.output_path)

        return Panel(
            table,
            title="[bold]Boltz-2 Prediction[/bold]",
            border_style="blue",
            width=52,
        )

    def _make_progress(self) -> Progress:
        """Create progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction starts."""
        self.start_time = time.time()
        self.completed = 0
        self.failed = 0

        # Print header
        console.print()
        console.print(self._make_header())
        console.print()

        # Start progress
        self._progress = self._make_progress()
        total = self.config.total_predictions or 1
        self._task_id = self._progress.add_task("Predicting", total=total)
        self._progress.start()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called after each prediction batch."""
        # Get record ID from batch
        if "record" in batch and len(batch["record"]) > 0:
            record = batch["record"][0]
            self.current_id = getattr(record, "id", "unknown")

        # Update progress
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction ends."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def mark_completed(self, record_id: str) -> None:
        """Mark a prediction as completed."""
        self.completed += 1

    def mark_failed(self, record_id: str) -> None:
        """Mark a prediction as failed."""
        self.failed += 1


class MultiBetaProgress:
    """Progress display for multi-beta runs with Rich progress bar."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.start_time: Optional[float] = None
        self.beta_results: List[dict] = []
        self._progress: Optional[Progress] = None
        self._task_id = None
        self._current_beta: float = 0.0

    def _format_beta(self, beta: float) -> str:
        """Format beta value for display."""
        if beta == 0.0:
            return "β=0"
        return f"β={beta:+.2f}"

    def _format_betas_range(self) -> str:
        """Format beta range for display."""
        if not self.config.betas:
            return "none"
        min_b, max_b = min(self.config.betas), max(self.config.betas)
        return f"{min_b:+.2f} to {max_b:+.2f}"

    def _make_header(self) -> Panel:
        """Create header panel for multi-beta mode."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")
        table.add_column(style="cyan")

        table.add_row("Model", self.config.model)
        table.add_row("Device", self.config.device)
        table.add_row("Seed", str(self.config.seed) if self.config.seed else "random")

        table.add_row("β (uniform)", f"{len(self.config.betas)} values")
        table.add_row("  range", self._format_betas_range())

        table.add_row("Diffusion", f"{self.config.diffusion_samples} samples")
        table.add_row("Recycling", f"{self.config.recycling_steps} steps")
        table.add_row("Sampling", f"{self.config.sampling_steps} steps")

        if self.config.input_path:
            table.add_row("Input", self.config.input_path)
        if self.config.output_path:
            table.add_row("Output", self.config.output_path)

        return Panel(
            table,
            title="[bold]Boltz-2 Sampling[/bold]",
            border_style="blue",
            width=52,
        )

    def start(self) -> None:
        """Start multi-beta progress display."""
        self.start_time = time.time()
        self.beta_results = []

        console.print()
        console.print(self._make_header())
        console.print()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            f"[cyan]{self._format_beta(self.config.betas[0])}[/cyan]",
            total=len(self.config.betas),
        )
        self._progress.start()

    def update_beta(self, beta: float) -> None:
        """Update progress bar with current beta."""
        self._current_beta = beta
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description=f"[cyan]{self._format_beta(beta)}[/cyan]",
            )

    def mark_beta_done(self, beta: float, elapsed: float, success: bool = True) -> None:
        """Mark a beta run as completed."""
        self.beta_results.append({
            "beta": beta,
            "elapsed": elapsed,
            "success": success,
        })

        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

    def get_summary(self) -> dict:
        """Return summary stats (completed, failed, elapsed)."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        completed = sum(1 for r in self.beta_results if r.get("success"))
        failed = sum(1 for r in self.beta_results if not r.get("success"))
        return {"completed": completed, "failed": failed, "elapsed": elapsed}

    def finish(self) -> None:
        """Finish multi-beta progress display."""
        if self._progress:
            self._progress.stop()
            self._progress = None
