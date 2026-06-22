from .base import BaseParser
from .drain import DrainParser
from .spell import SpellParser
from .ael import AELParser
from .lenma import LenMaParser

__all__ = ["BaseParser", "DrainParser", "SpellParser", "AELParser", "LenMaParser", "get_parser"]

def get_parser(name: str, **kwargs) -> BaseParser:
    parsers = {
        "drain": DrainParser,
        "spell": SpellParser,
        "ael": AELParser,
        "lenma": LenMaParser,
    }
    if name.lower() not in parsers:
        raise ValueError(f"Parser {name} not supported. Available: {list(parsers.keys())}")
    return parsers[name.lower()](**kwargs)
