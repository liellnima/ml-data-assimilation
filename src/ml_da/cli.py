from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ml_da import CONFIGS_DIR
from ml_da.orchestrator import run_experiment
from ml_da.tools.config import ModelConfig, load_config, update_experiment_cfg
from ml_da.tools.io import load_yaml, save_yaml
from ml_da.tools.logger import setup_logging
from ml_da.tools.paths import make_run_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
default_cfg = CONFIGS_DIR / "main.yaml"


def run_command(
    config: Path,
    stage: str,
    log_level: str,
    overwrite_cfg_dict: dict[str, Any] = None,  # cfgs that should be overwritten by the command
):
    """Basic function that is called by all commands."""
    typer.echo("Start run ...")
    # load and resolve configs
    experiment_cfg = load_config(config)
    # overwrite the configs if necessary through the command
    if overwrite_cfg_dict is not None:
        experiment_cfg = update_experiment_cfg(experiment_cfg, overwrite_cfg_dict)

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


@app.command()  # typer cli.py run data-generation
def data_generation(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run data generation from a yaml config."""
    run_command(config, stage="generate", log_level=log_level)


@app.command()  # typer cli.py run model
def model(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    model: Path = typer.Option(None, "--model", "-m", exists=True, readable=True),
    dataset: str = typer.Option(None, "--data", "-d"),
    id: str = typer.Option(None, "--data-id", "-i"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Run a model from a yaml config file.

    Can overwrite model and data params
    """
    override: dict[str, object] | None = None

    if model is not None or dataset is not None or id is not None:
        model_override: dict[str, object] = {}

        if dataset is not None:
            model_override["dataset_name"] = dataset

        if id is not None:
            model_override["data_id"] = id

        if model is not None:
            model_cfg_dict = load_yaml(model)
            model_override["model"] = ModelConfig.model_validate(model_cfg_dict)

        override = {"run_model": model_override}

    run_command(config, stage="run", log_level=log_level, overwrite_cfg_dict=override)


@app.command()
def evaluation(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Evaluate a model from a yaml config file."""
    run_command(config, stage="evaluate", log_level=log_level)


@app.command()
def aggregation(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Aggregate results, launched from a yaml config file."""
    run_command(config, stage="aggregate", log_level=log_level)


@app.command()
def visualization(
    config: Path = typer.Option(default_cfg, "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Visualize results from a yaml config file."""
    run_command(config, stage="visualize", log_level=log_level)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
