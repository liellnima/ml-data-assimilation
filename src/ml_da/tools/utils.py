from typing import Any


def str_join_ls(name_and_params: list[Any]) -> str:
    """Returns a joined string of the items in the list."""
    return "-".join(map(str, name_and_params))
