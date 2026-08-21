"""Re-export from src.benchmark.runner for test and CLI compatibility."""

import sys
from src.benchmark import runner

# Direct module dictionary sync
for attr in dir(runner):
    if not attr.startswith("__"):
        globals()[attr] = getattr(runner, attr)

class _Proxy(sys.modules[__name__].__class__):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        setattr(runner, name, value)

sys.modules[__name__].__class__ = _Proxy

if __name__ == "__main__":
    sys.exit(runner.main())
