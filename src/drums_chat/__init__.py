"""drums-chat: verifiers' `null` chat harness with a visual-context window, for long computer-use episodes.

Before every model call the window (drums_chat/window.py) decides which screenshots of the history are sent and at what
size. The default is the scalecua_rl trainer's rule (`images_to_keep`, default 5), as drums-lab's lab-scale and h2h
harnesses apply it to this endpoint (labbench/policy.py `trim_images` at 933df2b): image parts are removed from all but
the newest 5 tool messages that carry an image; the first user turn keeps its screenshot; the tool message keeps its text
("Success"); no placeholder is added. The program's own history keeps every image; the trace records what was sent.

Throughput options (all off by default, so the default request is byte-identical to 0.2.2's):

    env.agent.harness.id = "drums-chat"
    env.agent.harness.images_to_keep = 5        # the 0.2.2 knob; None = send every image
    env.agent.harness.screenshots_full = 3      # newest screenshots sent at full size (default: images_to_keep)
    env.agent.harness.screenshots_total = 6     # screenshots kept at all; the ones past `full` are downscaled
    env.agent.harness.downscale = 0.5           # width and height factor for those (1 or None: sent full size)
    env.agent.harness.screenshots_stride = 3    # move the window every 3 screenshots, not every one, so most turns
                                                # extend the previous request byte for byte (prefix-cache hits)
"""
import inspect

from pydantic import Field, model_validator
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE, compaction_args, launch_chat_program

from . import window as _window

ENTRY = 'if __name__ == "__main__":\n    asyncio.run(main())'
DEPS = '# dependencies = ["'
HOOK = '''

def _env_num(name, cast):
    v = _os.environ.get(name, "")
    return cast(v) if v not in ("", "None") else None


_FULL = _env_num("DRUMS_SCREENSHOTS_FULL", int)
_TOTAL = _env_num("DRUMS_SCREENSHOTS_TOTAL", int)
_DOWNSCALE = _env_num("DRUMS_DOWNSCALE", float)
_STRIDE = _env_num("DRUMS_SCREENSHOTS_STRIDE", int) or 1
_vf_chat = chat


async def chat(client, model, messages, tools, **kwargs):
    return await _vf_chat(client, model, window(messages, _FULL, _TOTAL, _DOWNSCALE, _STRIDE), tools, **kwargs)


'''
assert CHAT_PROGRAM_SOURCE.rstrip().endswith(ENTRY), "verifiers chat program layout changed"
assert CHAT_PROGRAM_SOURCE.count(DEPS) == 1, "verifiers chat program metadata changed"
WINDOW = "\nimport os as _os\n" + inspect.getsource(_window) + HOOK


def program_source(pillow: bool = False) -> str:
    """The chat program with the window. `pillow`: declare Pillow in the script's dependencies (needed to downscale)."""
    src = CHAT_PROGRAM_SOURCE
    if pillow:
        src = src.replace(DEPS, '# dependencies = ["pillow", "', 1)
    return src.rstrip()[: -len(ENTRY)] + WINDOW + ENTRY + "\n"


PROGRAM_SOURCE = program_source()


class DrumsChatConfig(NullHarnessConfig):
    images_to_keep: int | None = Field(5, ge=0)
    """Screenshots kept in tool messages sent to the model (newest first); None keeps all. The default of
    screenshots_full."""
    screenshots_full: int | None = Field(None, ge=0)
    """Newest screenshots sent at full size; None = images_to_keep."""
    screenshots_total: int | None = Field(None, ge=0)
    """Screenshots kept at all (>= screenshots_full); the older ones of them are downscaled. None = screenshots_full."""
    downscale: float | None = Field(0.5, gt=0, le=1)
    """Width and height factor of the screenshots between screenshots_full and screenshots_total; 1 or None = full size."""
    screenshots_stride: int = Field(1, ge=1)
    """The window's boundaries move in steps of this many screenshots. 1 = the 0.2.2 sliding window."""

    @model_validator(mode="after")
    def _total_covers_full(self):
        full = self.screenshots_full if self.screenshots_full is not None else self.images_to_keep
        if full is not None and self.screenshots_total is not None and self.screenshots_total < full:
            raise ValueError(f"screenshots_total ({self.screenshots_total}) is below screenshots_full ({full})")
        return self

    def window_args(self) -> tuple:
        """(full, total, downscale, stride) as the program applies them."""
        full = self.screenshots_full if self.screenshots_full is not None else self.images_to_keep
        total = self.screenshots_total if self.screenshots_total is not None else full
        return full, total, self.downscale, self.screenshots_stride

    def needs_pillow(self) -> bool:
        full, total, downscale, _ = self.window_args()
        return full is not None and total is not None and total > full and downscale is not None and downscale < 1


class DrumsChatHarness(Harness[DrumsChatConfig]):
    APPENDS_SYSTEM_PROMPT = NullHarness.APPENDS_SYSTEM_PROMPT
    SUPPORTS_MCP = NullHarness.SUPPORTS_MCP
    SUPPORTS_RESUME = NullHarness.SUPPORTS_RESUME
    SUPPORTS_TOOL_INTERCEPTION = NullHarness.SUPPORTS_TOOL_INTERCEPTION
    EXECUTES_CODE = NullHarness.EXECUTES_CODE
    NEEDS_CONTAINER = NullHarness.NEEDS_CONTAINER

    @property
    def program(self) -> str:
        return program_source(self.config.needs_pillow())

    async def setup(self, runtime) -> None:
        await runtime.prepare_uv_script(self.program, self.config.resolved_env)

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=None):
        system_prompt, prompt = self.resolve_prompt(data)
        args = compaction_args(self.config.compaction)
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        full, total, downscale, stride = self.config.window_args()
        env = {**self.config.resolved_env, "DRUMS_SCREENSHOTS_FULL": str(full), "DRUMS_SCREENSHOTS_TOTAL": str(total),
               "DRUMS_DOWNSCALE": str(downscale), "DRUMS_SCREENSHOTS_STRIDE": str(stride)}
        return await launch_chat_program(self.program, self.config, ctx, trace, runtime, endpoint, secret, mcp_urls,
                                         system_prompt, prompt, extra_args=args, env=env)


__all__ = ["DrumsChatHarness"]
