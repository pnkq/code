from dataclasses import dataclass

@dataclass
class BuildStats:
    lines: int = 0
    pieces: int = 0
    sequences: int = 0
