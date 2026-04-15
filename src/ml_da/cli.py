from __future__ import annotations

from pathlib import Path

import typer

from ml_da import CONFIGS_DIR
from ml_da.orchestrator import run_experiment
from ml_da.tools.config import load_config
from ml_da.tools.io import save_yaml
from ml_da.tools.logger import setup_logging
from ml_da.tools.paths import make_run_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
default_cfg = CONFIGS_DIR / "main.yaml"


def run_command(config: Path, stage: str, log_level: str):
    """Basic function that is called by all commands."""
    typer.echo("Start run ...")
    # load and resolve configs
    experiment_cfg = load_config(config)
    # prepare the dir where we are saving logs and e.g. the resolved configs
    run_dir = make_run_dir(experiment_cfg)
    save_yaml(experiment_cfg.model_dump(mode="json"), run_dir / "resolved_config.yaml")

    # set up logging (done here once, use getLogger in all other files pls)
    setup_logging(
        log_dir=run_dir,
        console_level=log_level,
    )

    # run the orchestrator
    run_experiment(experiment_cfg, stage=[stage], run_dir=run_dir)
    typer.echo("... Run completed.")


@app.command()
def generate(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run data generation from a yaml config."""
    run_command(config, stage="generate", log_level=log_level)


@app.command()  # would be run-model for the actual command
def run_model(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run a model from a yaml config file."""
    run_command(config, stage="run", log_level=log_level)


@app.command()
def evaluate(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Evaluate a model from a yaml config file."""
    run_command(config, stage="evaluate", log_level=log_level)


@app.command()
def aggregate(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Aggregate results, launched from a yaml config file."""
    run_command(config, stage="aggregate", log_level=log_level)


@app.command()
def visualize(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Visualize results from a yaml config file."""
    run_command(config, stage="visualize", log_level=log_level)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
