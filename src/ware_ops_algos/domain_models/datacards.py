from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import yaml

from .base_domain import BaseWarehouseDomain
from .base_domain_object import BaseDomainObject


@dataclass
class DataCard:
    name: str
    problem_class: str
    objective: str
    layout: dict[str, Any]
    articles: dict[str, Any]
    orders: dict[str, Any]
    resources: dict[str, Any]
    storage: dict[str, Any]
    warehouse_info: dict[str, Any]
    sources: Optional[dict[str, Any]] = None


def _section(obj: BaseDomainObject) -> dict[str, Any]:
    data = {"type": obj.get_type_value(),
            "features": obj.get_features()}
    return data


def load_and_flatten_data_card(card_path: str | Path) -> DataCard:
    with open(card_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def flatten_domain(domain: dict) -> dict:
        features = {}
        for obj in domain.get("objects", []):
            for feat in obj.get("features", []):
                name = feat["name"]
                if "value" in feat:
                    features[name] = feat["value"]
                else:
                    features[name] = True
            features.update(flatten_domain(obj))
        return features

    def section(domain: dict) -> dict:
        return {
            "type": domain.get("type"),
            "features": flatten_domain(domain),
        }

    return DataCard(
        name=raw.get("name", ""),
        problem_class=raw.get("problem_class", ""),
        objective=raw.get("objective", ""),
        layout=section(raw.get("layout", {})),
        articles=section(raw.get("articles", {})),
        orders=section(raw.get("orders", {})),
        resources=section(raw.get("resources", {})),
        storage=section(raw.get("storage", {})),
        warehouse_info=section(raw.get("warehouse_info", {})),
    )

def datacard_from_instance(domain: BaseWarehouseDomain,
                           name: str,
                           file_path: str | Path | None = None,
                           parser: str | None = None) -> DataCard:

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
    return DataCard(
        name=name,
        problem_class=domain.problem_class,
        objective=domain.objective,
        layout=_section(domain.layout),
        articles=_section(domain.articles),
        orders=_section(domain.orders),
        resources=_section(domain.resources),
        storage=_section(domain.storage),
        warehouse_info=_section(domain.warehouse_info),
        sources=sources
    )

def validate_against_card(domain: BaseWarehouseDomain, card: DataCard) -> tuple[bool, list[str]]:
    errors: list[str] = []
    def check(section_name: str, obj: BaseDomainObject, section: dict[str, Any]):
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


def save_card_yaml(card: DataCard, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(card.__dict__, f, sort_keys=False)

def load_card_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        dc = yaml.safe_load(f)
    return dc

def build_domain_from_card_path(card_path: str | Path) -> DataCard:
    with open(card_path, "r", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    return card
