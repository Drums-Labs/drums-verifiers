## What changed and why

<!-- One change per pull request. Say what changed and why in plain words. -->

## How it was tested

- [ ] `ruff check .` and `python -m pytest tests` pass locally
- [ ] A new task family or checker kind comes with its validation run on a live host (`summary.json` and `records.jsonl` attached or linked)
- [ ] A change to `taskset.py` that touches gates, verdicts or `dropped` reasons has a test in `tests/test_e2e_fake.py`

## Checklist

- [ ] Commits are signed off (`git commit -s`, Developer Certificate of Origin)
- [ ] No pins moved without a reason in the description (`verifiers`, the factory extra, the SDK floor)
- [ ] No secrets, tokens, private hostnames or customer names in code, tests, fixtures or logs
