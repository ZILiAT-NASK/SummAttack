from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Experiment:
    name: str
    base_config: Dict[str, Any]
