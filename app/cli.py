"""Command-line interface for the Grounded document assistant."""

from __future__ import annotations

import typer

from app.config import load_settings
from app.errors import ConfigError
from app.logging_config import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@app.command()
def index() -> None:
    """Reserve the document indexing command for a later phase."""
    load_settings()
    typer.echo("Document indexing will be implemented in a later phase.")


@app.command()
def ask() -> None:
    """Reserve the question-answering command for a later phase."""
    load_settings()
    typer.echo("Question answering will be implemented in a later phase.")


def main() -> None:
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    app()


if __name__ == "__main__":
    main()

