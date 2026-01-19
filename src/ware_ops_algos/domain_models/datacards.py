from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import yaml

from .base_domain import BaseWarehouseDomain
from .base_domain_object import BaseDomainObject


@dataclass
class DomainCard:
    name: str
    problem_class: str
    objective: str
    layout: Dict[str, Any]
    articles: Dict[str, Any]
    orders: Dict[str, Any]
    resources: Dict[str, Any]
    storage: Dict[str, Any]
    sources: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] | None = None


def _section(obj: BaseDomainObject, fallback_type: str) -> Dict[str, Any]:
    data = {"type": getattr(obj, "get_type_value", lambda: fallback_type)()}
    if hasattr(obj, "get_objects"):
        data["objects"] = obj.get_objects()
    data["features"] = {f: True for f in obj.get_features()}
    return data


def datacard_from_instance(domain: BaseWarehouseDomain,
                           name: str,
                           file_path: str | Path | None = None,
                           parser: str | None = None) -> DomainCard:

    sources = None
    if file_path is not None:
        if parser is not None:
            parser = parser
        sources = {
            "inputs": {
                "instance_file": {
                    "parser": parser,
                    "path": file_path.as_posix(),
                }
            }
        }
    return DomainCard(
        name=name,
        problem_class=domain.problem_class,
        objective=domain.objective,
        layout=_section(domain.layout, "layout"),
        articles=_section(domain.articles, "articles"),
        orders=_section(domain.orders, "orders"),
        resources=_section(domain.resources, "resources"),
        storage=_section(domain.storage, "storage"),
        sources=sources
    )


def validate_against_card(domain: BaseWarehouseDomain, card: DomainCard) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    def check(section_name: str, obj: BaseDomainObject, section: Dict[str, Any]):
        required = section.get("features", {})
        present = set(obj.get_features())
        for feat, must in required.items():
            if must and feat not in present:
                errors.append(f"{section_name}.features.{feat} required but missing")
        if "objects" in section and hasattr(obj, "get_objects"):
            have = obj.get_objects()
            for k, v in section["objects"].items():
                if bool(have.get(k, False)) != bool(v):
                    errors.append(f"{section_name}.objects.{k}={v} but instance has {have.get(k, False)}")
    check("layout", domain.layout, card.layout)
    check("articles", domain.articles, card.articles)
    check("orders", domain.orders, card.orders)
    check("resources", domain.resources, card.resources)
    check("storage", domain.storage, card.storage)
    return (len(errors) == 0, errors)


def save_card_yaml(card: DomainCard, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(card.__dict__, f, sort_keys=False)


def build_domain_from_card_path(card_path: str | Path) -> DomainCard:
    with open(card_path, "r", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    return card
