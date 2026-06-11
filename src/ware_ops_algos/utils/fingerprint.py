""""
Sketch: Walk the algorithm directory and collect all implemented algorithms in all subfolders of algorithms/

maybe check if whatever we find implements the Algorithm interface.
hash the classes that we have found and return a map that contains this hash.
on every commit we execute the fingerprint script and check if the hashes changed.

steps:
[x] find all subclasses of Algorithm
[] build the string representation of all subclasses and their mro
[] build a hash from that
[] return a map of all algos and hashes

[] build something to detect two things: 1. if a new algo is added to th sc set. 2. if a hash changed

Question:
- How to build the hash?
- How to make sure we detect all relevant changes to an algorithm?
- How to deal with configurations (parameters)?
"""
import ast
import hashlib
import importlib
import inspect
import pkgutil
import textwrap

# from ware_ops_algos.algorithms import FifoBatching, Algorithm
#
# # print(inspect.getsource(FifoBatching))
# # print("#####\n")
# # print(inspect.getmro(FifoBatching))
#
# source_lines = inspect.getsource(FifoBatching)
#
# # Boundry is Algorithm
# def inheritors(base_class):
#     #https://stackoverflow.com/questions/5881873/python-find-all-classes-which-inherit-from-this-one
#     subclasses = set()
#     work = [base_class]
#     while work:
#         parent = work.pop()
#         for child in parent.__subclasses__():
#             if child not in subclasses:
#                 subclasses.add(child)
#                 work.append(child)
#     return subclasses
#
# all_subclasses = inheritors(Algorithm)


# for cls in all_subclasses:
#     # print(inspect.getsource(c))
#     # print(inspect.getmro(c))
#     target = inspect.getmodule(cls)
#     # print(target)
#     src = textwrap.dedent(inspect.getsource(target))
#     # print(src)
#     ast_dump = ast.dump(ast.parse(src), annotate_fields=False)
#     # print(ast_dump)
#
#     h = hashlib.sha256()
#     for p in ast_dump:
#         h.update(p.encode("utf-8"))
#         h.update(b"\x00")  # separator so concatenations can't collide
#     hashed = h.hexdigest()
#     print(cls, hashed)


def fingerprint(cls: type, length: int = 12) -> str:
    """Hash of a class: its source plus its in-package bases, so a change to an
    inherited method counts too. Bases outside the algorithm's own top-level
    package (object, abc.ABC, third-party) are skipped. Comments and formatting
    drop out via an AST round-trip; only logic changes register."""
    root = cls.__module__.split(".")[0]
    parts = []
    for c in cls.__mro__:
        if c.__module__.split(".")[0] != root:
            continue
        src = textwrap.dedent(inspect.getsource(c))
        parts.append(ast.dump(ast.parse(src), annotate_fields=False))
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:length]


def import_all(package) -> None:
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


def build_map(base: type) -> dict[str, str]:
    """{qualified name -> hash} for every subclass of `base`."""
    return {
        f"{c.__module__}.{c.__qualname__}": fingerprint(c)
        for c in sorted(inheritors(base), key=lambda c: (c.__module__, c.__qualname__))
    }


# hashmap = build_map(Algorithm)
# print(hashmap)
# print(len(hashmap))