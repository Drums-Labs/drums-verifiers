# Changelog

## 0.2.1 - 2026-11-13

First release on PyPI: `pip install drums-verifiers`.

- Depends on `verifiers>=0.4.1,<0.5` from PyPI (prime-rl main requires 0.4.1 too) instead of a git-pinned commit, and
  on `drums-sdk>=0.3.0`.
- `vf-eval` was removed upstream in 0.4.1; the README, CONTRIBUTING and the end-to-end tests use `vf-run`, which takes
  the same `-m`, `-n`, `-r`, `--client.*` and `--env.*` options (there is no `--no-push`). `vf-run` exits non-zero when
  every episode errored; the tests allow that for the deliberate env-crash scenario.
- Package metadata for PyPI: readme, classifiers, keywords, project URLs (docs at docs.drums.run).
- Release workflow (`.github/workflows/release.yml`): a `v*` tag builds, tests the wheel on Python 3.11 to 3.13 and
  publishes from the `pypi` environment.

## 0.2.0 - 2026-11-11

Repository tag `v0.2.0` (the rc1 code plus the SDK 0.2.5 API); not published to PyPI.

## 0.1.0 - 2026-10-31

Repository tag `rc1`: the first internal release.
