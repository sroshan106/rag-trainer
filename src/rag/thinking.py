"""Thinking token filtering and suppression for reasoning models."""

# "/no_think" is Qwen3's soft switch to immediately close the think block.
NO_THINK_SUFFIX = " /no_think"
THINKING_MODEL_PREFIXES = ("qwen3",)
THINK_CLOSE = "</think>"


def wants_no_think(model: str) -> bool:
    """Whether the model requires thinking prompt directives."""
    return model.startswith(THINKING_MODEL_PREFIXES)


def strip_thinking(text: str) -> str:
    """Remove leading <think> block from text if present."""
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[1].strip()
    return text.strip()


class ThinkFilter:
    """Drop a leading ``<think>`` block from a stream, token by token."""

    _OPEN = "<think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._passthrough = False

    def feed(self, chunk: str) -> str:
        if self._passthrough:
            return chunk
        self._buffer += chunk
        text = self._buffer.lstrip()
        if THINK_CLOSE in text:
            self._passthrough = True
            return text.split(THINK_CLOSE, 1)[1].lstrip()
        if text.startswith(self._OPEN) or self._OPEN.startswith(text):
            return ""
        self._passthrough = True
        return text
