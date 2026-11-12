# drums-verifiers

Drums computer-use environments for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) (v1 API) and
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). Each rollout is one isolated Ubuntu desktop from a drumsd
host running one OSWorld task. The model acts through `computer_*` tools and gets a screenshot back after every action.
The reward is OSWorld's own checker, returned together with the episode's gate verdict.

Tested against: verifiers `60290d43` (the commit prime-rl `a566e5d8` pins), drumsd v0.2.0 and the Drums Python SDK 0.2.5 ([drums-python-sdk](https://github.com/Drums-Labs/drums-python-sdk) `main`, `f8124ac`; the commit drumsd's `install/lock.json` pins). The SDK must be 0.2.5 or later: this package uses `drums.hybrid` and `drums.Overloaded`.

## Install

On the machine that runs the env workers (it must reach drumsd; a tunnel is fine):

```
pip install drums-sdk                     # the Drums SDK (import name `drums`; standard library only)
pip install -e .                          # this package (pulls verifiers at the pinned commit)
export DRUMS_URL=http://127.0.0.1:7070 DRUMS_TOKEN_FILE=/etc/drums/token
```

Until `drums-sdk` is on PyPI, install the SDK from Git or from the wheel a drumsd release ships, then this package:

```
# from Git (the branch that carries the drums-sdk distribution name; needs read access while the repo is private)
pip install "drums-sdk @ git+https://github.com/Drums-Labs/drums-python-sdk@pypi-release"
# or from an installed drumsd host (install.sh puts the release's wheel in /opt/drums/sdk)
pip install /opt/drums/sdk/drums*.whl
pip install -e .
```

`pip install -e '.[test]'` adds pytest. `.[factory]` (the task generator's pins) does not install next to prime-rl: see
[CONTRIBUTING.md](CONTRIBUTING.md#the-factory-extra-and-prime-rl).

## Use with prime-rl

```toml
[[orchestrator.train.source]]
name = "drums"
env.taskset.id = "drums-verifiers"
env.taskset.task_ids = ["e0df059f-28a6-4169-924f-b9623e7184cc"]   # OSWorld ids; served by drumsd's pinned checkout
# optional: GRPO from one reached state. One episode = one parent env, `prefix`, then N forked children
env.fork_n = 8
env.prefix = ["pyautogui.click(960, 540)"]
```

With `fork_n = N`, the N child traces of an episode form the comparison group. Use `group_size = 1` with `grpo`, or use
`hierarchical_grpo` with `episode_agents = ["agent"]`. With `fork_n = 0` (the default), use `group_size` as usual.

Eval without a GPU, using the scripted stand-in model in `examples/mock_policy.py`:

```
python examples/mock_policy.py --port 8011 &
MOCK_KEY=x uv run eval drums-verifiers -n 1 -r 2 -m mock --client.base-url http://127.0.0.1:8011/v1 \
    --client.api-key-var MOCK_KEY --no-monitors.prime
```

## Rewards that cannot be trusted never reach the trainer as rewards

| what happened | trace | prime-rl |
|---|---|---|
| all gates passed, checker ran | reward = checker score (0 or 1) | trained |
| a gate failed (isolation, mounts, network, cgroup limits, state hash, processes alive after restore), at the start or mid-episode (drumsd quarantines the env) | no reward; `dropped = "drums_invalid:<verdict>:<gates>"` | skipped (`iter_trainable_traces`), left out of eval reward means |
| the checker failed, or drumsd or the desktop failed mid-episode | error (`ok = False`) | skipped |

The checker's score for a flagged episode is kept in `trace.info["drums"]` for audit. Set `DRUMS_RECORD=path.jsonl` to get
one line per scored trace: env, fork parent, verdict, failed gates, score, reward handed out, drop reason.

## Tests

`pytest tests` runs the unit tests, plus an end-to-end run of `vf-eval` against a fake drumsd (`tests/fake_drumsd.py`)
and the mock policy. That run covers single envs, fork groups, invalid children that get dropped, and a mid-episode
environment failure.

How to run them, add a task family, and send a change: [CONTRIBUTING.md](CONTRIBUTING.md).

## Acceptance (fresh host, real model)

```
acceptance/launch-host.sh                                   # SUBNET, SG, KEY in the environment; adds ssh alias drums-acceptance
acceptance/acceptance.sh --host drums-acceptance --release drums-release-<commit>.tar.gz \
    --model-url http://<vllm-host>:8000/v1 --tasks 4 --fork 4 --rollouts 2 --tool-style scalecua
```

This installs Drums with the release's installer, installs prime-rl (pinned) and this adapter, then runs three things through
prime-rl's eval: single-env rollouts, fork groups, and a negative control whose children must all be dropped. It pulls back
the records, prime-rl's trace stream, screenshots, drumsd metrics and `summary.json` (pass/fail). Use `--model-url mock` for a
plumbing run with no model. The model endpoint must be reachable from the host and must serve tool calls.
