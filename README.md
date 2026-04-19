## Quick Start

These steps are for anyone cloning this project to set it up for development.

If you do not have uv installed yet, run `pip install uv` first.

1. **Create and Activate Virtual Environment:**
   This command creates a `.venv` folder using the Python version specified in the
   project.

   On Linux:

   ```bash
   # Create virtualenv with UV, specifying the Python version
   uv venv -p 3.12

   # Activate the virtual environment
   source .venv/bin/activate

   # To deactivate, simply run: deactivate

   # or use directly while inside the repository
   uv run <command>
   ```

   On Windows:

   ```bash
   # Create virtualenv with UV, specifying the Python version
   uv venv -p 3.12

   # Allow the executation of your venv scripts
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

   # Activate the virtual environment
   .venv\\Scripts\\activate.ps1

   # To deactivate, simply run: deactivate
   ```

2. **Install Dependencies:**
   This command installs all dependencies from `pyproject.toml` and locks them using
   `uv.lock`. It also installs your local package (e.g., `src/ml-da`) in
   editable mode.

   ```bash
   uv sync
   ```

3. **Set Up Pre-commit Hooks:**
   This will run automated code quality checks (like `ruff` and `black`) before each
   commit.

   Linux / Windows:

   ```bash
   pre-commit install
   ```

You are now ready to start development!

## Project Usage

1. **Run Data Generation**
   ```bash
   # To run the default main.yaml config
   typer src/ml_da/cli.py run data-generation

   # If you want to run specific configs 
   typer src/ml_data/cli.py run data-generation --config configs/your_experiments/template.yaml
   ```
2. **Run Models**
   ```bash
   typer src/ml_da/cli.py run model

   # you can also provide the exact model config and the dataset id
   typer src/ml_da/cli.py run model -m configs/models/da/var4d.yaml -d 002
   ```

   You can also check out ``srcipts/`` to see how we run some of the models on the Mila cluster.

## References
We are building upon the following repos:
- [DAPPER](https://github.com/nansencenter/DAPPER)
- [dabench](https://github.com/StevePny/DataAssimBench)
- [Rolnick Lab Template](https://github.com/RolnickLab/lab-uv-template)
