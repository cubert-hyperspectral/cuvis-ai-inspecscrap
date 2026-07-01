# Contributing to cuvis-ai-inspecscrap

We welcome contributions: bug reports, fixes, new nodes, docs.

## Workflow

1. Fork the repo and create a feature branch from `main`. Never push directly to `main`.
2. If you add code that should be tested, add tests under `tests/`.
3. If you add a new node, author it through the **cuvis-ai-node** conventions (inherit
   `cuvis_ai_core.node.Node`, declare `INPUT_SPECS` / `OUTPUT_SPECS`, keep `forward`
   torch-native, ship golden-reference + port-contract tests). Add its fully-qualified
   `class_name` to `configs/plugins/cuvis_ai_inspecscrap.yaml` under `capabilities:`, then
   regenerate the manifest's palette metadata (see below).
4. Make sure CI is green locally:

   ```bash
   uv run --no-sources --extra dev --extra tiff pytest tests/ -q
   uv run --no-sources --extra dev ruff format --check cuvis_ai_inspecscrap tests
   uv run --no-sources --extra dev ruff check cuvis_ai_inspecscrap tests
   uv run --no-sources --extra dev mypy cuvis_ai_inspecscrap/
   ```

5. Update `CHANGELOG.md` under `## [Unreleased]`.
6. Open a PR. Watch CI go green, then squash-merge.

## Code style

- `ruff format` + `ruff check` (config in `pyproject.toml`, line length 100).
- Type-annotate the public surface. `mypy cuvis_ai_inspecscrap/` is **non-blocking in CI**
  (`|| true`): the `cuvis-ai-core` / `cuvis-ai-schemas` / torch stubs are incomplete, so a
  green run typically reports a few `import-untyped` notes. Treat those as warnings; treat any
  other mypy error as a hard fail.
- New node classes omit the `Node` suffix. Anomaly-style nodes expose `scores` / `anomaly_score`,
  never `anomaly_map` / `image_score`.

## Regenerating manifest palette metadata

The `capabilities:` list in `configs/plugins/cuvis_ai_inspecscrap.yaml` carries palette metadata
(port specs, category, tags, icon, doc summary) that the cuvis-ai UI and the static node catalog
read. It is generated, not hand-written: after changing a node's ports or docstring, run the
`emit_metadata` tool shipped with cuvis-ai-core against the manifest and commit the diff. Release
CI runs the same tool with `--check` as a drift guard.

## Running tests

Tests live under `tests/`. `pythonpath = ["."]` in `[tool.pytest.ini_options]` puts the repo root
on `sys.path` so `cuvis_ai_inspecscrap` is importable. Tests that need the real InSpecScrap dataset
skip automatically when it is absent, so the suite is green without the data download. The TIFF data
modules require the `tiff` extra (`--extra tiff`), which pulls `cuvis-ai-dataloader[tiff]`.

## How the workflows work

The repo ships three GitHub Actions workflows:

- **`.github/workflows/ci.yml`** runs on every push and PR: five independent jobs (tests with
  coverage, ruff format/lint, non-blocking mypy, security scanning with pip-audit/detect-secrets/
  bandit, and a build-and-validate). `--no-sources` installs from normal indexes the way an
  external user would; `--locked` pins to the committed `uv.lock`.
- **`.github/workflows/cuvis_ai_compat.yml`** runs on dependency PRs, a weekly cron, and manual
  dispatch. It audits this plugin's dependency specifiers against the `cuvis-ai-core` lock, so a
  core release that tightens a shared dependency turns red as a "core moved, react" signal rather
  than surfacing at a user's first pipeline run.
- **`.github/workflows/release.yml`** runs only on `v*.*.*` tags: validate, security, build (with a
  tag-equals-package-version check), and a GitHub Release whose body is this version's `CHANGELOG.md`
  section.

## Release process

Releases follow semver. The package version is derived from the git tag by setuptools-scm, so there
is no static `version` to bump.

1. Move `## [Unreleased]` entries into a `## X.Y.Z - YYYY-MM-DD` section in `CHANGELOG.md`
   (this section becomes the GitHub Release body).
2. Open a PR with the changelog stamp. Merge once CI is green.
3. From `main`: `git tag -a vX.Y.Z -m "Release vX.Y.Z: <summary>"` then `git push origin vX.Y.Z`.
4. The release workflow runs. If it fails: delete the remote tag, fix on a branch, re-tag to
   `vX.Y.(Z+1)`. Tags are immutable, do not repoint.
5. Verify the released tag clones and loads cleanly against a fresh env, then open a PR against
   `cubert-hyperspectral/cuvis-ai` bumping the `tag:` in `configs/plugins/cuvis_ai_inspecscrap.yaml`.

## Issues

Use GitHub issues. Include the cuvis-ai-core version, torch version, a minimal repro pipeline YAML,
and the failing tensor shape if applicable.

## License

By contributing you agree your contributions are licensed under Apache-2.0 (see [LICENSE](LICENSE)).
