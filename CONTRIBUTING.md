# Contributing to AutoParquet

Thank you for your interest in contributing to AutoParquet! We welcome all contributions, from bug reports and feature requests to code improvements and documentation.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/autoparquet
   cd autoparquet
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,polars]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Project Structure

```
autoparquet/
├── src/autoparquet/          # Main package source code
│   ├── __init__.py          # Package exports
│   ├── cli.py               # Command-line interface
│   ├── converters.py        # DataFrame to Arrow conversion
│   ├── constants.py         # Configuration constants
│   ├── io.py                # Parquet read/write operations
│   ├── schema.py            # Schema inference and optimization
│   └── transforms.py        # Data transformations (kmer tools, etc.)
├── tests/                    # Test suite
├── benchmarks/               # Performance benchmarks
├── docs/                     # Documentation
└── pyproject.toml           # Project configuration
```

## Running Tests

We use `pytest` for testing. Run the full test suite:

```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/autoparquet
```

Run a specific test:
```bash
pytest tests/test_io.py::test_basic_io
```

## Code Quality

### Linting and Formatting

We use `ruff` for linting and automatic code formatting:

```bash
# Check for linting issues
ruff check .

# Automatically format code
ruff format .
```

### Type Checking

We use `mypy` for static type checking:

```bash
mypy src/autoparquet tests
```

All code should be fully type-annotated.

### Using Make Commands

For convenience, use the provided Makefile:

```bash
make help          # Show available commands
make lint          # Run ruff linting
make format        # Auto-format with ruff
make check         # Run mypy type checking
make test          # Run pytest
make clean         # Remove build artifacts
```

## Code Style Guidelines

- **Type hints**: All functions must have complete type annotations
- **Docstrings**: Use clear, descriptive docstrings for public functions
- **Comments**: Comment why, not what. The code should be self-documenting
- **Line length**: Maximum 88 characters (enforced by ruff)
- **Imports**: Organize imports in groups: stdlib, third-party, local

## Submitting Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: clear description of what was added"
   ```

3. Run tests and linting before pushing:
   ```bash
   make test
   make lint
   make check
   ```

4. Push to your fork and create a pull request on GitHub

5. Ensure the CI checks pass and address any review feedback

## Cutting a Release

Releases are cut manually from `main`. The `Makefile` has targets for every
step; the flow below assumes you're on a clean checkout of `main` with all
tests passing.

1. **Bump the version.** Update both `pyproject.toml` and
   `src/autoparquet/__init__.py` in one shot:
   ```bash
   make version VERSION=0.2.0
   ```
   We follow [Semantic Versioning](https://semver.org/): bump the patch for
   bug fixes, the minor for backward-compatible features, the major for
   breaking changes.

2. **Verify the working tree is clean.**
   ```bash
   make lint
   make check
   make test
   ```
   Everything must pass before tagging.

3. **Commit and tag.** The tag drives the release; create it on the version-
   bump commit so `git describe` and PyPI agree.
   ```bash
   git add pyproject.toml src/autoparquet/__init__.py
   git commit -m "Release v0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin main --follow-tags
   ```

4. **Build the distribution.** This produces a wheel and sdist in `dist/`.
   ```bash
   make build
   ```

5. **Smoke-test on TestPyPI first.** This catches packaging mistakes before
   they hit the real index, where each version number can only be used once.
   ```bash
   make upload   # uploads to TestPyPI
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ autoparquet==0.2.0
   ```
   Try a quick `python -c "import autoparquet; print(autoparquet.__version__)"`
   in a fresh virtualenv.

6. **Publish to PyPI.** Requires typing `yes` at the confirmation prompt.
   ```bash
   make prod
   ```

7. **Cut the GitHub release.** Either via the web UI or the CLI:
   ```bash
   gh release create v0.2.0 --title "v0.2.0" --generate-notes
   ```
   `--generate-notes` autopopulates the changelog from PR titles since the
   previous tag; edit it down to a short human-friendly summary.

### Credentials

`twine` reads from `~/.pypirc`, environment variables (`TWINE_USERNAME` /
`TWINE_PASSWORD`), or the keyring. Use API tokens scoped to this project, not
your account password — see [PyPI's token docs](https://pypi.org/help/#apitoken).

### If something goes wrong

* **Wrong version on PyPI:** you can't reuse a version. Yank it
  (`pip install pkginfo twine` then `twine yank ...`) and release the next
  patch version.
* **Built artifact is wrong:** `make clean && make build` and try again. The
  `clean` target removes `dist/` so you don't accidentally re-upload a stale
  wheel.

## Branch Protection (for maintainers)

`main` should be protected so all changes go through CI and PR review. The
recommended minimum is:

| Rule | Why |
|------|-----|
| Require a pull request before merging | Forces review and prevents accidental direct pushes. |
| Require status checks to pass before merging | `main` always has green CI. |
| Require branches to be up to date before merging | Catches "passes alone, fails together" semantic conflicts. |
| Block force pushes | Preserves history; force-pushes to `main` rewrite shared commits. |
| Block branch deletion | Prevents accidentally deleting `main`. |

### Setting it up via the GitHub UI

1. Go to **Settings → Branches → Branch protection rules → Add rule**
   (or **Settings → Rules → Rulesets → New branch ruleset** for the newer UI).
2. **Branch name pattern:** `main`
3. Tick the boxes:
   - **Require a pull request before merging** — leave "Required approvals" at
     `0` for a solo project; bump to `1+` once you have other maintainers.
   - **Require status checks to pass** — search for and add the
     `test (3.10)`, `test (3.11)`, and `test (3.12)` checks from `.github/workflows/ci.yml`.
   - **Require branches to be up to date before merging** (sub-option of the above).
   - **Do not allow bypassing the above settings** — apply the rules to
     admins too. You can always loosen this later.
4. Leave **Allow force pushes** and **Allow deletions** unchecked.
5. Save.

For the release flow, this means: bump version on a branch (`release/v0.2.0`),
open a PR, let CI go green, merge, then tag and publish from the merged
commit on `main`.

### Optional next steps

* Add a `CODEOWNERS` file to auto-request review from specific people.
* Require signed commits if you publish security-sensitive releases.
* Add a `release.yml` GitHub Action that runs on tag push, builds, and
  publishes to PyPI using a [trusted publisher](https://docs.pypi.org/trusted-publishers/)
  — removes the need for API tokens on a developer machine.

## Reporting Issues

When reporting bugs or requesting features:

- Check if the issue already exists
- Include your Python version and OS
- For bugs, provide a minimal reproducible example
- For features, explain the use case and expected behavior

## Benchmarking

If you're making performance-related changes, run the benchmark suite:

```bash
python benchmarks/bench.py
```

For custom compression configurations:
```bash
python benchmarks/bench.py --compression zstd:10 snappy gzip:6
```

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!
