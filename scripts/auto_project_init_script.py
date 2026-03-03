#!/usr/bin/env python3
"""
Initialization Script for Lab Advanced Template.

This script customizes the project based on user input and removes template-specific placeholders.
It handles:
- User input via CLI flags or interactive prompts.
- Automatic detection of repository URL.
- Variable replacement in configuration files.
- Directory renaming and import updates.
- README customization.
"""

import argparse
import json
import re
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from typing import LiteralString
except ImportError:
    # Workaround for python<3.11
    LiteralString = str

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve(strict=False).parent.parent

# Placeholders (These will be updated by the script itself after the first run)
PLACEHOLDER_PACKAGE_NAME = "core"
PLACEHOLDER_IMPORT_NAME = "my_awesome_project"
PLACEHOLDER_PROJECT_NAME = "My Awesome Project"
PLACEHOLDER_README_PROJECT_NAME = "\\<YOUR_PROJECT_NAME_HERE>"
PLACEHOLDER_DESCRIPTION_TOML = ""
PLACEHOLDER_DESCRIPTION_README = "[Provide a brief, one-sentence description of your project here.]"
PLACEHOLDER_PYTHON_VERSION = "3.12"
PLACEHOLDER_REPO_URL = "REPOSITORY_URL"
PLACEHOLDER_AUTHOR = "Author"
PLACEHOLDER_EMAIL = "author@example.com"

DEFAULT_PYTHON_VERSION = "3.12"
VALID_PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]

# Files to modify
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
README_MD = PROJECT_ROOT / "README.md"
INIT_MARKER_FILE = PROJECT_ROOT / "scripts" / ".init_completed"

# --- Helper Functions ---


def prompt_user(prompt: str, default: Optional[str] = None, choices: Optional[List[str]] = None) -> str | None:
    """
    Prompts the user for input, with optional default and validation.

    Args:
        prompt: The text to display to the user.
        default: The default value to return if the user enters nothing.
        choices: A list of valid choices. If provided, input is validated against this list.

    Returns:
        The user's input or the default value.
    """
    while True:
        prompt_text = f"{prompt}"
        if default:
            prompt_text += f" - [{default}]"
        if choices:
            prompt_text += f" - Available choices: ({', '.join(choices)})"
        prompt_text += ": "

        value = input(prompt_text).strip()

        if not value and default:
            return default

        if not value and not default:
            print("Value is required.")
            continue

        if choices and value not in choices:
            print(f"Invalid choice. Must be one of: {', '.join(choices)}")
            continue

        return value


def update_pyproject_toml(
    filepath: Path,
    package_name: str,
    description: str,
    author: str,
    email: str,
    python_version: str,
    dry_run: bool,
) -> None:
    """
    Updates pyproject.toml with project metadata.

    Args:
        filepath: Path to pyproject.toml.
        package_name: The new package name.
        description: Project description.
        author: Author name.
        email: Author email.
        python_version: Python version string.
        dry_run: If True, prints changes without writing to file.
    """
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return

    content = filepath.read_text(encoding="utf-8")

    # Basic replacements
    # Note: Using regex for more robust replacement where simple string replace might be ambiguous

    # Update name = "core" -> name = "package_name"
    # We look for name = "core" specifically in the [project] section usually at the top
    content = re.sub(
        pattern=f'name = "{PLACEHOLDER_PACKAGE_NAME}"', repl=f'name = "{package_name}"', string=content, count=1
    )

    # Update description
    content = re.sub(
        pattern=f'description = "{PLACEHOLDER_DESCRIPTION_TOML}"',
        repl=f'description = "{description}"',
        string=content,
        count=1,
    )

    # Update authors
    # authors = [{ name = "Your Name", email = "you@example.com" }]
    new_authors = f'authors = [{{ name = "{author}", email = "{email}" }}]'
    content = re.sub(pattern=r"authors = \[.*]", repl=new_authors, string=content, count=1)

    # Update python version
    # requires-python = ">=3.12,<3.13"
    # We assume the user provides "3.12", we want ">=3.12,<3.13" logic or just update the base
    # For simplicity, let's try to construct the range if it matches X.Y format
    match = re.match(pattern=r"(\d+)\.(\d+)", string=python_version)
    if match:
        major, minor = map(int, match.groups(default=None))
        next_minor = minor + 1
        new_requires = f'requires-python = ">={major}.{minor},<{major}.{next_minor}"'
        content = re.sub(pattern=r'requires-python = ".*"', repl=new_requires, string=content, count=1)

    # Update tool.hatch.build.targets.wheel packages
    # packages = ["src/core"]
    content = content.replace(f'packages = ["src/{PLACEHOLDER_PACKAGE_NAME}"]', f'packages = ["src/{package_name}"]')

    # Update black target-version
    new_target_version = f'target-version = ["py{python_version.replace(".", "")}"]'
    content = re.sub(pattern=r"target-version = .*", repl=new_target_version, string=content, count=1)

    if dry_run:
        print(f"[Dry Run] Would update {filepath} with project metadata.")
    else:
        filepath.write_text(data=content, encoding="utf-8")
        print(f"Updated {filepath}")


def rename_package_directory(package_name: str, dry_run: bool) -> None:
    """
    Renames src/core to src/<package_name> and updates imports.

    Args:
        package_name: The new name for the package directory.
        dry_run: If True, prints changes without moving files or updating imports.
    """
    src_previous = PROJECT_ROOT / "src" / PLACEHOLDER_PACKAGE_NAME
    src_new = PROJECT_ROOT / "src" / package_name

    if not src_previous.exists():
        print(f"Warning: {src_previous} does not exist. Skipping rename.")
        return

    if dry_run:
        print(f"[Dry Run] Would rename {src_previous} to {src_new}")
    else:
        shutil.move(src=src_previous, dst=src_new)
        print(f"Renamed {src_previous} to {src_new}")

    # Update imports in all .py files
    # We need to walk through the project and replace "from my_awesome_project" with "from package_name"
    # and "import my_awesome_project" with "import package_name"

    # Directories to skip
    skip_dirs = {".git", ".venv", "__pycache__", ".nox", ".idea"}

    for file_path in PROJECT_ROOT.rglob(pattern="*.py"):
        # Check if any part of the path is in skip_dirs
        if any(part in skip_dirs for part in file_path.parts):
            continue

        # Skip this script itself to avoid self-modification during this step
        if file_path.resolve(strict=False) == Path(__file__).resolve(strict=False):
            continue

        content = file_path.read_text(encoding="utf-8")

        # Simple replacements for imports
        # This is a basic heuristic and might need refinement for complex cases
        new_content = content.replace(f"from {PLACEHOLDER_IMPORT_NAME}", f"from {package_name}")
        new_content = new_content.replace(f"import {PLACEHOLDER_IMPORT_NAME}", f"import {package_name}")

        if content != new_content:
            if dry_run:
                print(f"[Dry Run] Would update imports in {file_path}")
            else:
                file_path.write_text(data=new_content, encoding="utf-8")
                print(f"Updated imports in {file_path}")


def update_readme(
    readme_path: Path, project_name: str, description: str, package_name: str, python_version: str, dry_run: bool
) -> None:
    """
    Updates README.md content.

    Args:
        readme_path: Path to README.md.
        project_name: The project name.
        description: The project description.
        package_name: The package name.
        python_version: The Python version used.
        dry_run: If True, prints changes without writing to file.
    """
    if not readme_path.exists():
        return

    content = readme_path.read_text(encoding="utf-8")

    # Replace Title and Description
    content = re.sub(
        pattern=f"# {re.escape(pattern=PLACEHOLDER_README_PROJECT_NAME)}", repl=f"# {project_name}", string=content
    )
    content = re.sub(pattern=re.escape(pattern=PLACEHOLDER_DESCRIPTION_README), repl=description, string=content)

    # Remove Template Initialization Section
    # We look for the section start and end
    start_marker = "## 🚀 Template Initialization"
    end_marker = "## 🐍 Python Version"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        # Keep the end marker section
        content = content[:start_idx] + content[end_idx:]

    # Update package name in README (src/core)
    content = re.sub(
        pattern=f"src/{PLACEHOLDER_PACKAGE_NAME}",
        repl=f"src/{package_name}",
        string=content,
    )

    # Update Python Version section text
    content = re.sub(
        pattern=f"This project uses \\*\\*Python {re.escape(pattern=PLACEHOLDER_PYTHON_VERSION)}\\*\\*",
        repl=f"This project uses **Python {python_version}**",
        string=content,
    )
    content = re.sub(
        pattern=f"The virtual environment created by `uv venv -p {re.escape(pattern=PLACEHOLDER_PYTHON_VERSION)}`\\*\\*",
        repl=f"The virtual environment created by `uv venv -p {python_version}`",
        string=content,
    )
    content = re.sub(
        pattern=f"uv venv -p {re.escape(pattern=PLACEHOLDER_PYTHON_VERSION)}",
        repl=f"uv venv -p {python_version}",
        string=content,
    )
    if dry_run:
        print(f"[Dry Run] Would update {readme_path}")
    else:
        readme_path.write_text(data=content, encoding="utf-8")
        print(f"Updated {readme_path}")


def update_self(script_path: Path, replacements: Dict[str, str], dry_run: bool) -> None:
    """Updates the script's own constants to match the new project state."""
    if not script_path.exists():
        return

    content = script_path.read_text(encoding="utf-8")

    for key, value in replacements.items():
        # Look for KEY = "..." or KEY = '...'
        # We use json.dumps to generate a safe string representation (e.g. "value")
        # and replace the existing assignment.
        # We assume the constants are defined at the top level.

        # Regex explanation:
        # ^: Start of line
        # {key}: The constant name
        # \s*=\s*: Assignment operator with optional whitespace
        # .*$: Match the rest of the line (the value)

        new_line = f"{key} = {json.dumps(obj=value)}"
        content = re.sub(pattern=rf"^{key}\s*=\s*.*$", repl=new_line, string=content, flags=re.MULTILINE)

    if dry_run:
        print(f"[Dry Run] Would update {script_path} constants.")
    else:
        script_path.write_text(data=content, encoding="utf-8")
        print(f"Updated {script_path} constants for future runs.")


def self_update_for_next_run_of_script(
    args: Namespace,
    author: str | Any,
    description: str | Any,
    email: str | Any,
    package_name: str | Any,
    project_name: str | Any,
    python_version: str | Any,
):
    self_replacements = {
        "PLACEHOLDER_PACKAGE_NAME": package_name,
        "PLACEHOLDER_IMPORT_NAME": package_name,
        "PLACEHOLDER_PROJECT_NAME": project_name,
        "PLACEHOLDER_README_PROJECT_NAME": project_name,
        "PLACEHOLDER_DESCRIPTION_TOML": description,
        "PLACEHOLDER_PYTHON_VERSION": python_version,
        "DEFAULT_PYTHON_VERSION": python_version,
        "PLACEHOLDER_AUTHOR": author,
        "PLACEHOLDER_EMAIL": email,
    }

    update_self(script_path=Path(__file__), replacements=self_replacements, dry_run=args.dry_run)


def gather_metadata_fields(args: Namespace) -> tuple[str | Any, str | Any, str | Any, str | Any, str | Any, str | Any]:
    project_name = args.project_name or prompt_user(prompt="Project Name", default=PLACEHOLDER_PROJECT_NAME)

    default_package_name = PLACEHOLDER_IMPORT_NAME.lower().replace(" ", "_").replace("-", "_")
    if project_name != PLACEHOLDER_PROJECT_NAME:
        default_package_name = project_name.lower().replace(" ", "_").replace("-", "_")

    package_name = args.package_name or prompt_user(prompt="Package Name (snake_case)", default=default_package_name)

    description = args.description or prompt_user(prompt="Project Description", default=PLACEHOLDER_DESCRIPTION_TOML)
    author = args.author or prompt_user(prompt="Author Name", default=PLACEHOLDER_AUTHOR)
    email = args.email or prompt_user(prompt="Author Email", default=PLACEHOLDER_EMAIL)
    python_version = args.python_version or prompt_user(
        prompt="Python Version", default=DEFAULT_PYTHON_VERSION, choices=VALID_PYTHON_VERSIONS
    )
    return author, description, email, package_name, project_name, python_version


def generate_args_and_parser() -> Namespace:
    parser = argparse.ArgumentParser(description="Initialize the project from the template.")

    # Project Metadata
    parser.add_argument("--project-name", help="Name of the project")
    parser.add_argument("--package-name", help="Python package name (snake_case)")
    parser.add_argument("--description", help="Brief project description")
    parser.add_argument("--author", help="Author's name")
    parser.add_argument("--email", help="Author's email")

    # Technical Configuration
    parser.add_argument("--python-version", choices=VALID_PYTHON_VERSIONS, help="Target Python version")

    # Flags
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to disk")

    args = parser.parse_args(args=None, namespace=None)
    return args


# --- Main Execution ---


def main() -> None:
    """
    Main execution function for the initialization script.

    Parses arguments, gathers user input, and triggers file updates.
    """
    args = generate_args_and_parser()

    if INIT_MARKER_FILE.exists() and not args.dry_run:
        print("\n⚠️  WARNING: It looks like this project has already been initialized.")
        print(f"Marker file exists: {INIT_MARKER_FILE}")
        print("Re-running this script might overwrite your changes or cause unexpected behavior.")

        should_continue = prompt_user(prompt="Do you want to continue anyway?", default="no", choices=["yes", "no"])
        if should_continue != "yes":
            print("Aborting initialization.")
            return

    print("🚀 Starting Project Initialization...")

    # 1. Gather Inputs
    author, description, email, package_name, project_name, python_version = gather_metadata_fields(args=args)

    # 3. Execution
    print("\nApplying changes...")

    # Update pyproject.toml
    update_pyproject_toml(
        filepath=PYPROJECT_TOML,
        package_name=package_name,
        description=description,
        author=author,
        email=email,
        python_version=python_version,
        dry_run=args.dry_run,
    )

    # Rename Directory and Update Imports
    if package_name != PLACEHOLDER_PACKAGE_NAME:
        rename_package_directory(package_name=package_name, dry_run=args.dry_run)

    # Update README.md
    update_readme(
        readme_path=README_MD,
        project_name=project_name,
        description=description,
        package_name=package_name,
        python_version=python_version,
        dry_run=args.dry_run,
    )

    # 4. Update self (the script itself) to prepare for next run
    self_update_for_next_run_of_script(
        args=args,
        author=author,
        description=description,
        email=email,
        package_name=package_name,
        project_name=project_name,
        python_version=python_version,
    )

    print("\n✅ Initialization Complete!")
    if args.dry_run:
        print("(This was a dry run. No files were modified.)")
    else:
        INIT_MARKER_FILE.touch(exist_ok=True)
        print("\n🔍  Please review the changes and commit them.")
        print("\n📦  You can now process to installing the package")


if __name__ == "__main__":
    main()
