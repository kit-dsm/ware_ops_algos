"""Keep simulator and pipeline dependencies outside the algorithm library."""

import ast
from importlib.util import resolve_name
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "ware_ops_algos"


def _imports(path: Path):
    module = "ware_ops_algos." + ".".join(path.relative_to(ROOT).with_suffix("").parts)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield resolve_name("." * node.level + (node.module or ""), package)
            elif node.module:
                yield node.module


def test_all_modules_respect_import_boundaries():
    banned_external = ("casim", "cosy", "cosy_luigi", "luigi", "hydra", "pydantic")
    for path in ROOT.rglob("*.py"):
        area = path.relative_to(ROOT).parts[0]
        for imported in _imports(path):
            assert not imported.startswith(banned_external), (path, imported)
            if not imported.startswith("ware_ops_algos."):
                continue
            dependency = imported.split(".")[1]
            allowed = {
                "domain_models": {"domain_models"},
                "algorithms": {"domain_models", "algorithms"},
                "planning": {"domain_models", "algorithms", "planning"},
                "taxonomy": {"domain_models", "algorithms", "taxonomy"},
                "domain_algo_mapper": {"domain_models", "algorithms",
                                       "domain_algo_mapper"},
            }.get(area)
            if allowed is not None:
                assert dependency in allowed, (path, imported)
