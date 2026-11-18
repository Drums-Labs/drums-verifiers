# Changelog

## Unreleased

- Throughput options for the drums-chat screenshot window (A-39 correction), all off by default: the default request
  is byte-identical to 0.2.2's.
  - `env.agent.harness.screenshots_full` (default: `images_to_keep`): the newest screenshots sent at full size.
  - `screenshots_total` (default: `screenshots_full`): screenshots kept at all. Those older than `screenshots_full`
    are downscaled by `downscale` (default 0.5, a width and height factor; 1 = full size).
  - `screenshots_stride` (default 1): move the window in steps of N screenshots, so on N - 1 turns of N the request
    is the previous one plus the new messages, byte for byte, and vLLM's prefix cache covers the whole history.
  - The system prompt and the first user message (task text and first screenshot) are never changed. The chat
    program declares Pillow only when it downscales. The window is its own module (`src/drums_chat/window.py`),
    embedded verbatim in the chat program.

## 0.2.2 - 2026-11-16

- Three task families for the D1 study, one per app (`src/drums_tasks/families/d1.py`): `d1_files_collect` (copy
  every file matching a pattern from a folder tree into a target folder under a renaming scheme), `d1_calc_aggregate`
  (a sum, average, count or maximum of a column under a condition, written to a cell) and `d1_writer_style` (bold every
  occurrence of a word, save under a new name). 96 variants each at scale 1.0; four near misses each.
- `python -m drums_tasks.split`: a training set and a held-out set of one family from two seeds, `pool.json`, and an
  overlap check (instruction text, seeded file path and content, params) that fails the split on any collision.
  `drums_tasks.build.generate` takes `count` and `start` for that.
- `GENERATOR_VERSION` is `t6-factory-2`: the default build is now 40 families, 1,928 tasks, 5,437 near misses.
- `drums_tasks.validate` retries drumsd's 429 (admission) on step and verify as well, not only on create and fork.
- Factory action steps chunk file writes at 8,000 base64 characters (drumsd caps a step argument at 8,192); setup
  steps keep 60,000.
- The built wheel and sdist are no longer tracked in the repository (`dist/` is ignored).

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
