"""drums-chat: verifiers' `null` chat harness with a visual-context window, for long computer-use episodes.

Before every model call, image parts are removed from all but the newest `images_to_keep` tool messages that carry an image
(the first user turn keeps its screenshot). This is the rule of the scalecua_rl trainer (`images_to_keep`, default 5) as
drums-lab's lab-scale and h2h harnesses apply it to this endpoint (labbench/policy.py `trim_images` at 933df2b, copied
unchanged). The tool message keeps its text ("Success"); no placeholder is added, as in that rule. The program's own history
keeps every image; the trace records what was actually sent.

    env.agent.harness.id = "drums-chat"
    env.agent.harness.images_to_keep = 5        # None = send every image
"""
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE, compaction_args, launch_chat_program

ENTRY = 'if __name__ == "__main__":\n    asyncio.run(main())'
WINDOW = '''
import copy as _copy
import os as _os


def trim_images(messages: list, keep: int | None) -> list:
    if keep is None:
        return messages
    out = _copy.copy(messages)
    idx = [i for i, m in enumerate(out) if m.get("role") == "tool" and isinstance(m.get("content"), list)
           and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])]
    for i in idx[:-keep] if keep > 0 else idx:
        m = dict(out[i])
        m["content"] = [p for p in m["content"] if not (isinstance(p, dict) and p.get("type") == "image_url")]
        out[i] = m
    return out


_IMAGES_TO_KEEP = _os.environ.get("DRUMS_IMAGES_TO_KEEP", "")
_IMAGES_TO_KEEP = int(_IMAGES_TO_KEEP) if _IMAGES_TO_KEEP not in ("", "None") else None
_vf_chat = chat


async def chat(client, model, messages, tools, **kwargs):
    return await _vf_chat(client, model, trim_images(messages, _IMAGES_TO_KEEP), tools, **kwargs)


'''
assert CHAT_PROGRAM_SOURCE.rstrip().endswith(ENTRY), "verifiers chat program layout changed"
PROGRAM_SOURCE = CHAT_PROGRAM_SOURCE.rstrip()[: -len(ENTRY)] + WINDOW + ENTRY + "\n"


class DrumsChatConfig(NullHarnessConfig):
    images_to_keep: int | None = 5
    """Screenshots kept in tool messages sent to the model (newest first); None keeps all."""


class DrumsChatHarness(Harness[DrumsChatConfig]):
    APPENDS_SYSTEM_PROMPT = NullHarness.APPENDS_SYSTEM_PROMPT
    SUPPORTS_MCP = NullHarness.SUPPORTS_MCP
    SUPPORTS_RESUME = NullHarness.SUPPORTS_RESUME
    SUPPORTS_TOOL_INTERCEPTION = NullHarness.SUPPORTS_TOOL_INTERCEPTION
    EXECUTES_CODE = NullHarness.EXECUTES_CODE
    NEEDS_CONTAINER = NullHarness.NEEDS_CONTAINER

    async def setup(self, runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=None):
        system_prompt, prompt = self.resolve_prompt(data)
        args = compaction_args(self.config.compaction)
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        env = {**self.config.resolved_env, "DRUMS_IMAGES_TO_KEEP": str(self.config.images_to_keep)}
        return await launch_chat_program(PROGRAM_SOURCE, self.config, ctx, trace, runtime, endpoint, secret, mcp_urls,
                                         system_prompt, prompt, extra_args=args, env=env)


__all__ = ["DrumsChatHarness"]
