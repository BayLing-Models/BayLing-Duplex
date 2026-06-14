from typing import TYPE_CHECKING

__all__ = ["BayLingDuplex", "DuplexResult", "ResponseSegment"]

if TYPE_CHECKING:
    from .duplex import BayLingDuplex, DuplexResult, ResponseSegment


def __getattr__(name):
    if name in __all__:
        from .duplex import BayLingDuplex, DuplexResult, ResponseSegment

        exports = {
            "BayLingDuplex": BayLingDuplex,
            "DuplexResult": DuplexResult,
            "ResponseSegment": ResponseSegment,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
