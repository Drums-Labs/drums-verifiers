# Contributing to drums-verifiers

This package has three parts: the verifiers / prime-rl environment (`src/drums_verifiers`), the adapters for verl
and SkyRL (`src/drums_adapters`), and the task factory (`src/drums_tasks`: seeded task families, their scripted
solutions, the lab verifier `drums_t6.py`, and the validator that proves every task on a live drumsd host). The
rules below apply to all three.

## Set up

```
python3 -m venv .venv && . .venv/bin/activate
pip install drums-sdk                      # or the Git / wheel fallback in README.md, until drums-sdk is on PyPI
pip install -e '.[test]'                   # this package, verifiers at the pinned commit, pytest
```

The factory's generator has its own venv, because its pins must match drumsd's checker venv exactly (that is what
makes a build byte-identical on any machine) and one of them conflicts with prime-rl:

```
python3 -m venv .venv-factory
.venv-factory/bin/pip install -r t6/requirements-factory.txt -e '.[test]'
```

### The `factory` extra and prime-rl

`.[factory]` pins `pillow==11.0.0` (drumsd's `install/osw-requirements.lock`); prime-rl's `renderers[multimodal]`
needs `pillow>=12.2.0`. The two cannot share a venv, and relaxing the factory pin is not a fix: the generator draws
images with Pillow, so a different Pillow changes the bytes of the tasks and their sha256 in `index.json`. Keep the
factory in `.venv-factory` and prime-rl in its own environment; the generated `tasks.jsonl` is what crosses between
them. If drumsd's checker venv ever moves to Pillow 12, bump the pin here and `GENERATOR_VERSION` together.

## Run the tests

```
pytest tests                                       # unit tests + the end-to-end run against the fake drumsd
.venv-factory/bin/python -m pytest tests/test_factory.py    # the factory: deterministic builds, task shape, checkers
```

- `tests/test_e2e_fake.py` starts `tests/fake_drumsd.py` and the mock policy and runs `vf-eval` through them: single
  envs, fork groups, invalid children that are dropped, a mid-episode environment failure, the reuse guard, quarantine.
  It is skipped when `vf-eval` is not on the path (install the package first).
- `tests/test_factory.py` needs the generator pins; it skips itself when `openpyxl`, `docx` or `pptx` are missing.
  It never needs a desktop: the checkers run on files written the way the scripted solutions write them.
- Nothing in `tests/` talks to a real drumsd. Validation on a real host is the step below.

## Add a task family

A family is one generator function in a module under `src/drums_tasks/families/`, registered in that module's
`FAMILIES` dict as `name: (function, count)`. The function takes `(rng, index)` and returns a `Spec`
(`src/drums_tasks/core.py`). Everything it does must come from `rng` (from `rng_for(seed, family, index)`), so the
same seed rebuilds the same task.

A `Spec` carries:

- `files`: the desktop files the setup seeds (absolute path under `/home/user` to bytes). Office files go through
  `office_write`, which fixes zip timestamps and document properties.
- `instruction`, `app`, `open_cmd`: what the agent reads and what is open when the episode starts.
- `check`: the options for `drums_t6.check` (`kind` plus what that kind needs). Existing kinds: `text_file`,
  `files_state`, `zip_members`, `xlsx_cells`, `xlsx_table`, `xlsx_sheetnames`, `csv_table`, `docx_paragraphs`,
  `docx_bold`, `pdf_text`, `pptx_titles`, `image`, `chrome_tabs`, `chrome_bookmarks`, `json_keys`, `python_tests`,
  `sqlite_rows`, and `all` (several sub-checks that must all pass). Add a kind to `src/drums_tasks/verifier/drums_t6.py`
  only if none fits; a new kind needs a `check_alt` twin (the redundant checker, which must read the file another way).
- `oracle`: the scripted actions that solve the task. It must score 1.0.
- `near_misses`: at least two named action lists that look like solutions and must score 0.0 (wrong value, wrong file,
  wrong sheet, partial edit, ...). They are the proof that the checker cannot be gamed.
- `allow`, `roots`, `protected`: the scope manifest. The builder adds a decoy file the instruction never mentions; the
  scope probe (oracle plus one edit of that file) must be flagged by the reward audit.

Then:

1. `python -m drums_tasks.build --seed 20261108 --out /tmp/ts --check --families <name>`: builds twice in memory and
   fails on any byte difference. Fix non-determinism before anything else.
2. `.venv-factory/bin/python -m pytest tests/test_factory.py`: the checker separates the oracle from the near misses
   on the files the solutions write.
3. On a drumsd host (see `t6/host_pipeline.sh` for the full sequence):
   `python -m drums_tasks.validate --tasks /tmp/ts --out runs/<name> --only <name>`.
   For every task it creates an env, forks it into no-op, oracle, near misses and scope probe, runs each, and verifies.
   A task is accepted only if the oracle scores 1.0 with a trusted audit, every near miss and the no-op score 0.0,
   and the scope probe is flagged. Put `summary.json` and `records.jsonl` from that run in the PR.
4. Bump `GENERATOR_VERSION` in `src/drums_tasks/__init__.py` whenever a build's bytes change (new family, changed
   family, changed pins). Published task sets are named by seed and generator version.

## Pull requests

- One change per PR, with the tests that cover it. A new family or checker kind comes with its validation run.
- Do not move the pins without saying why: `verifiers` is pinned to the commit prime-rl pins, the factory to drumsd's
  checker venv, and the SDK floor is the oldest version with the APIs this package calls.
- Rewards that cannot be trusted never reach the trainer as rewards (README, "Rewards that cannot be trusted"). Any
  change to `taskset.py` that touches gates, verdicts or `dropped` reasons needs a test in `tests/test_e2e_fake.py`.
- Write plain commit messages that say what changed and why. No generated attribution lines.
- No secrets, tokens, hostnames of private machines, or customer names in code, tests, fixtures or logs.
- Run `pytest tests` before you push. CI is not wired for this repo yet; the PR author is the CI.
