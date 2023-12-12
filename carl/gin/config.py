from typing import Any, Dict, List


def make_gin_bindings(bindings: Dict[str, Any]) -> List[str]:
    """Converts a list of gin bindings to a string that can be parsed by gin.parse_config_files_and_bindings."""
    return [f"{key} = {value}" for key, value in bindings.items()]
