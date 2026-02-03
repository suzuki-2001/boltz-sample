"""Unified console output for boltz using rich."""

import pyfiglet
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def header(model: str = "boltz2") -> None:
    """Display startup banner."""
    logo = pyfiglet.figlet_format("boltz-sample", font="slant")
    model_display = "Boltz-2" if model == "boltz2" else "Boltz-1"
    content = (
        f"[bold cyan]{logo}[/bold cyan]"
        f"  [dim]Uniform Scaling for Protein Structure Prediction[/dim]\n"
        f"  [dim]Based on {model_display}[/dim]"
    )
    console.print()
    console.print(Panel(content, border_style="blue", width=72))


def step(phase: int, total: int, msg: str) -> None:
    """Display phase step (e.g. [1/4] Processing inputs)."""
    console.print()
    console.print(f"[bold cyan]\\[{phase}/{total}][/bold cyan] {msg}")


def info(msg: str) -> None:
    console.print(f"  [blue]ℹ[/blue] {msg}")


def success(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow] {msg}")


def error(msg: str) -> None:
    console.print(f"  [red]✗[/red] {msg}")


def status(msg: str) -> None:
    console.print(f"  [dim]>[/dim] {msg}")


def report_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Display a result summary table."""
    table = Table(title=title, border_style="dim")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print()
    console.print(table)


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
