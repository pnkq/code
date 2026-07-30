from dataclasses import dataclass

@dataclass
class Token:
    text: str
    lang: str
    start: int
    end: int

