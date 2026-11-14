# Security

## Supported versions

The `main` branch. There are no release branches yet; a drumsd release pins the commit of this package it was tested
with.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: <https://github.com/Drums-Labs/drums-verifiers/security/advisories/new>.
Do not open a public issue for a security problem. You will get an acknowledgement within 3 business days and a fix or
a plan within 30 days. Please include the package commit, the drumsd version (`client.health()["version"]`), the
prime-rl or verifiers command line, and a `DRUMS_RECORD` line or trace if the problem is in scoring.

## What this package does and does not protect

This package is a client of drumsd. It does not run anything privileged and it holds one secret, the drumsd token.

- The token comes from `DRUMS_TOKEN_FILE` (default `/etc/drums/token`) or `DRUMS_TOKEN`. It is never written into
  task data, traces, records or logs. The SDK sends it over HTTPS, or over plain HTTP to loopback addresses only
  (`localhost`, `127.0.0.0/8`, `::1`); set `DRUMS_URL` to a tunnel endpoint on loopback for a remote host.
- The model's tool calls are turned into actions in the desktop (pyautogui code, shell commands in the `hybrid` style)
  and run as the desktop user inside the environment. They never run on the machine that hosts the env workers.
- Rewards that fail a gate are dropped, and the reward audit's flags are passed through; neither is a security
  boundary. The scope scan in particular runs with the desktop user's rights and is a reward-hacking signal, not
  isolation.

What the desktop isolates is drumsd's responsibility. In short, and honestly:

- Each environment is a process tree in its own user, pid, mount and network namespaces, under cgroup limits, with
  `no_new_privs` set on its init. The desktop user has no privilege escalation: `sudo`, `su`, `pkexec`, setuid bits
  and file capabilities give nothing, and tasks whose setup needs root run that step as drumsd's provisioning, never
  through the agent.
- Environments restored from one checkpoint share the host's kernel, and in the default profile sibling environments
  share the origin's uid range. That is accepted inside one trusted customer's boundary. It is not isolation between
  hostile tenants; per-tenant isolation is a VM boundary around the host.
- The hardened execution profile is not the default yet.
- Known open item: a Chrome crash in forked children was traced to memory protection keys and fixed in the Drums CRIU
  fork; the compatibility regression it exposed in the hardened profile is still being worked through.

Details: the `security` page of the [docs](https://docs.drums.run) and the drumsd repository.
