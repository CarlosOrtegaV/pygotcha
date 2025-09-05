# Project overview
This repository contains **`pygotcha`**, a Python package with ML methods for weakly supervised learning.  
It provides both a library and a CLI (`pygotcha.cli:app`) built with Typer.  

# Repository structure
- `src/` → main source code
- `tests/` → test suite
- `README.md` → project description
- `pyproject.toml` → build system, dependencies, lint/type/test config
- `docs/` → generated documentation

# Tools & dependencies
- Python: >=3.11,<4.0
- Core libs: `numpy`, `pandas`, `scikit-learn`, `scipy`, `typer`
- Dev tools: `pytest`, `mypy`, `ruff`, `coverage`, `pre-commit`, `pdoc`

# Coding standards
- Use **absolute imports only**, never relative imports.
- Target **line length = 100**.
- Use **NumPy-style docstrings** for public APIs.
- All functions and methods must have **type hints** (strict mypy).
- **Never use `print` or `pprint`; always use the `logging` module**.
- Use timezone-aware `datetime` objects.
- Prefer vectorized NumPy/pandas operations over loops.

# Testing
- Tests live in `tests/` and `src/` (doctests).
- Use **pytest** with strict config (`--strict-config --strict-markers`).
- Treat warnings as errors.
- Ensure docstring examples pass doctests.

# Documentation
- Generated with `pdoc --docformat numpy`.
- Docstrings should be runnable and conform to the NumPy style guide.