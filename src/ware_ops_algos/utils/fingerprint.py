import ast
import hashlib
import importlib
import inspect
import pkgutil
import sys
import textwrap


def fingerprint(cls: type, *, length: int = 12) -> str:
    """Hash of a class: its source plus its in-package bases, so a change to an
    inherited method counts too. Bases outside the algorithm's own top-level
    package (object, abc.ABC, third-party) are skipped.

    `ast.unparse` regenerates canonical source from the AST, so comments and
    formatting drop out while the hash stays stable across Python versions.
    """
    root = cls.__module__.split(".")[0]
    parts = []
    for c in cls.__mro__:
        if c.__module__.split(".")[0] != root:
            continue
        src = textwrap.dedent(inspect.getsource(c))
        parts.append(ast.unparse(ast.parse(src)))
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:length]


def import_all(package) -> None:
    """Import every submodule so __subclasses__() sees all algorithms. Skip this
    if your package __init__ already imports them."""
    for m in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(m.name)


def inheritors(base: type) -> set[type]:
    seen: set[type] = set()
    work = [base]
    while work:
        for child in work.pop().__subclasses__():
            if child not in seen:
                seen.add(child)
                work.append(child)
    return seen


def build_map(base: type, *, epoch: str = "") -> dict[str, str]:
    """{qualified name -> hash} for every subclass of `base`."""
    return {
        f"{c.__module__}.{c.__qualname__}": fingerprint(c, epoch=epoch)
        for c in sorted(inheritors(base), key=lambda c: (c.__module__, c.__qualname__))
    }