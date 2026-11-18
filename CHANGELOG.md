# Changelog

## Unreleased

## 0.3.0 - 2026-11-18

The post-D1 fixes integrated on one branch (drums-verifiers #12 to #18). Not on PyPI and not tagged; the GPU templates
install it by commit (`[image] drums_verifiers` in `templates/rl-config.toml`).

- The GPU templates no longer install `main`: `[image] drums_verifiers` is a commit placeholder (`PIN_A_COMMIT`) that
  `run.sh build`, the Modal app and `check` refuse until it is a 40-character sha or a release (`==X.Y.Z`).
- `drums_tasks.validate` takes its token from `$DRUMS_TOKEN` by default, like the SDK and the docs (A-14).
  `--token-file` still works and wins when given; without either it falls back to `$DRUMS_TOKEN_FILE`, then
  `~/.drums-token`. New `--url` (default `$DRUMS_URL`).
- Episode transcripts (A-42): each `DRUMS_RECORD` / `record_path` row now has `transcript`, the episode's tool calls in
  order (`tool`, `args` as JSON cut to 200 characters, `status` = ok / invalid / env_failed / quarantined / skipped /
  no_action / error, `latency_s` for the whole action + pause + screenshot, `t`). It also has `turn_tokens`, the prompt
  (cache reads included), cached and completion tokens and seconds of each model call; the token fields are null when
  the client reports no usage. The action list is also in `trace.info["drums"]["transcript"]`. On by default;
  `env.transcript = false` gives the 0.2.2 rows.
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
- Train on your own GPUs: `templates/` and `docs/train-on-your-gpus.md`. One config file (`templates/rl-config.toml`:
  model, GPU type and split, Drums secret, task set, budget, prices, image pins) drives one runner
  (`templates/common/drums_rl.py`). The runner works in your own Modal account (`templates/modal/app.py`: volumes, a
  Secret, GPU fallback lists, a start deadline that cancels a launch with no container, retries that resume) or on
  any 8-GPU Linux machine (`templates/gpu-box/run.sh`, the same image under Docker).
  - Commands: download (writes `refs/main` for a pinned revision), check, scan (step-0 rates and the pool report),
    smoke, train, evaluate, export, spend.
  - `train` moves partial checkpoint directories aside before it resumes, and stops itself at the budget (a GPU-spend
    ledger across all launches).
  - The prime-rl recipe (`templates/common/prime-rl-drums.toml`) is D1's validated config (IPO loss, no reference KL)
    plus throughput settings to be validated on D2.
  - `drums_rl.py pool-report` reads the expected zero-advantage share from a record file.
- A task can name its lab verifier by hash (drumsd's verifier sandbox, Drums-Labs/drumsd#27).
  - `python -m drums_tasks.build --verifier-ref hash` adds `evaluator.verifier_sha256` (and the redundant checker's) to every task. The default `name` keeps the bytes.
  - `index.json` records `verifier_sha256` and `verifier_ref`.
  - `drums_tasks.upload.ensure_verifiers` asks the host for each hash a task set names and uploads the local file when it is missing (`POST /v1/verifiers`). `drums_tasks.validate` runs it before the first create (`--verifier FILE` for your own). `DrumsEnv` runs it before a task's first create, once per host (`verifier_files` in the env config).
- The task library (`docs/task-library.md`): a rubric for what makes a computer-use task good for RL (end-state
  checker with near misses and a no-op that scores 0, a learnable band measured by a K = 2 step-0 scan, lossless
  params, horizon tiers, named skills, binary reward by default, contamination rules), what each check kind does and
  does not verify (A-15), and an audit of the 40 existing families: 21 keep, 5 fix, 3 drop, 11 park (writer_bold moves to fix: in 3 to 7 of 96 variants its phrase occurs twice and
  the oracle scores 0) (Chrome, GIMP and
  VS Code are not proven on Drums).
- 11 new families, 96 variants each, all on existing check kinds, each with an oracle, three to five near misses and
  lossless params: medium-horizon `medium_files_rules`, `medium_calc_pipeline`, `medium_writer_restructure`;
  multi-app `multi_csv_report`, `multi_letter_fill`; hidden-state `hidden_files_dotfiles`, `hidden_calc_rows`,
  `hidden_writer_readonly`; recovery (the setup creates the fault) `recover_files_interrupted`, `recover_calc_column`,
  `recover_writer_steps`. Modules `families/medium.py`, `workflow.py`, `hidden.py`, `recovery.py`, helpers in
  `lib.py`. None has run on a live desktop yet.
- `params` is lossless in the 26 pre-D1 families on the apps Drums runs today: each records every drawn parameter and
  a digest of the drawn content, so the split's three-way overlap check (instruction, file, params) passes for the
  kept families without skipping the params check. The tasks are byte-identical to 0.2.2 (pinned in
  `tests/test_library.py`); only `solutions.jsonl` changes.
- A third near miss for files_delete_pattern, calc_rename_sheet, writer_delete_para, writer_bold, writer_heading,
  impress_title, impress_delete_slide and impress_add_slide.
- `drums_tasks.split --start N`: both index ranges start at N, so a later pool of a family never reuses an earlier
  pool's task ids.
- `GENERATOR_VERSION` is `t6-factory-3`.
- The D2 pool proposal (`docs/d2-pool.md`): candidates, difficulty tiers, about 672 training variants over about 14
  families, held-out by seed for every family plus two whole families for transfer, D1's held-out set as an anchor,
  and the step-0 scan that decides before the freeze.
- `tests/test_library.py`: the new families and the kept ones offline (oracle 1, near misses 0, no-op 0 under both
  checkers; zero overlap at the default seeds; unchanged task bytes).

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
