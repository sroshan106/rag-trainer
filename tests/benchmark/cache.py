import sys
from src.benchmark import cache

for attr in dir(cache):
    if not attr.startswith("__"):
        globals()[attr] = getattr(cache, attr)

class _Proxy(sys.modules[__name__].__class__):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        setattr(cache, name, value)

sys.modules[__name__].__class__ = _Proxy
