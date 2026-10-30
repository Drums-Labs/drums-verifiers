"""drums-verifiers: Drums computer-use environments for verifiers (v1) and prime-rl.

    [[orchestrator.train.source]]
    env.taskset.id = "drums-verifiers"
    env.taskset.task_ids = ["e0df059f-28a6-4169-924f-b9623e7184cc"]
    env.fork_n = 8                      # optional: one reached state, 8 children = one GRPO group
"""
from drums_verifiers.taskset import DrumsEnv, DrumsTaskset

__all__ = ["DrumsTaskset", "DrumsEnv"]
