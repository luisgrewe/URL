# URL

### Testing Docstring
'''uv run flake8 qbca.py --config .flake8 || uv run python -m flake8 qbca.py --config .flake8 '''

## Linting (Flake8 via uv)

- Install linters into the `uv` environment:
```bash
uv add flake8 flake8-docstrings flake8-bugbear
```

- Run Flake8 for the whole repo (uses `.flake8`):
```bash
uv run flake8 --config .flake8
```

- Run Flake8 for a single file:
```bash
uv run flake8 qbca.py --config .flake8
```

- Run the flake8 binary directly from the venv (if needed):
```bash
.venv/bin/flake8 --config .flake8
```

- VS Code visual linting: select the project interpreter (Command Palette → `Python: Select Interpreter` → choose the `.venv`), reload the window, then open files — Problems panel and editor squiggles will show Flake8 findings.

Notes:
- The repo-level `.flake8` controls rules such as blank-line enforcement and docstring checks.
- To suppress a specific check for a file temporarily, use a `# noqa: CODE` comment or add a `per-file-ignores` entry in `.flake8`.

- Quick reference: see `FLAKE8-CHEATSHEET.md` for common codes and fixes.

### Linting notebooks (ipynb)

- Install `nbqa` into the `uv` environment:
```bash
uv add nbqa
```

- Run Flake8 on a notebook (checks code cells using your `.flake8`):
```bash
uv run nbqa flake8 path/to/notebook.ipynb --config .flake8
```

- Autoformat or fixable issues: run an autoformatter via `nbqa`, for example:
```bash
uv run nbqa autopep8 path/to/notebook.ipynb --in-place
```

Notes:
- `nbqa` applies tools to code cells only and preserves notebook structure.
- You can use the same `.flake8` config for notebooks and .py files to keep rules consistent.
