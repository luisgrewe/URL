FLAKE8 CODES CHEAT-SHEET

This short cheat-sheet maps common Flake8 / pydocstyle codes to plain English and quick fixes.

- D100: Missing docstring in public module
  - Fix: Add a short module-level docstring at top of the file (one-line summary).

- D101: Missing class docstring
  - Fix: Add a one-line docstring inside the class immediately after the class line.

- D102 / D103 / D104 / D105 / D107: Missing docstrings for public functions/methods or return/param docs
  - Fix: Add short docstrings for functions, or use `# noqa: D102` for a specific function.

- D204 / D205: Blank line formatting around docstrings
  - D204: 1 blank line required after class docstring.
  - D205: 1 blank line required between summary line and description in docstring.
  - Fix: Ensure correct blank-line placement.

- W291: Trailing whitespace
  - Fix: Remove extra spaces at line ends.

- W292: No newline at end of file
  - Fix: Ensure file ends with a single newline.

- W293: Blank line contains whitespace
  - Fix: Clean blank lines so they contain no spaces/tabs.

- E501: Line too long
  - Fix: Wrap the expression or increase `max-line-length` in `.flake8` if long literal lines are acceptable.

- E302 / E301 / E303: Blank-line rules
  - E301/E302 enforce number of blank lines before functions/classes.
  - Fix: Add 1 (nested) or 2 (top-level) blank lines as appropriate.

- E741: Ambiguous variable name (e.g., `l`)
  - Fix: Rename ambiguous single-letter variable to descriptive name (e.g., `dim`).

Tips
- To silence a single occurrence: add `# noqa: CODE` at the end of the line.
- To ignore rules for a file: add a `per-file-ignores` line in `.flake8`.
- Use `nbqa` to run Flake8 on notebooks: `uv run nbqa flake8 notebook.ipynb --config .flake8`.

See `.flake8` in the repo root for configured rules.
