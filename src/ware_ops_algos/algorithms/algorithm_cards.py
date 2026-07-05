import importlib
from importlib.resources import as_file, files
import os
from typing import List, Dict, Optional
import yaml
from pathlib import Path

from ware_ops_algos.domain_models import BaseDomainObject

CONFIGS_FILENAME = "configs.yaml"


class RequirementConflict(Exception):
    """Raised when a configuration's requirements are mutually unsatisfiable or invalid."""


class AlgorithmCard:
    """An algorithm card describes an algorithm's subproblem, objective, requirements,
    and implementation. A card may additionally declare an optional configuration.

    A configuration consists of one or more configuration options. Each option
    takes a value:
      - component option: {"problem_type": <subproblem>} — the value is an
        algorithm card of that subproblem (e.g., any routing card)
      - value option: {"values": {VALUE: {"requirements": {...}}, ...}} — the
        value is one of a fixed set declared on the card; values may carry
        requirements of their own

    Resolving a configuration (see resolve_configuration) selects one value per
    option and produces an ordinary algorithm card whose requirements combine
    those of this card and of every selected value.
    """

    def __init__(
        self,
        algo_name: str,
        problem_type: str,
        requirements: Dict,
        objective: Optional[str],
        implementation: Dict,
        description: Optional[str] = None,
        configuration: Optional[Dict] = None,
    ):
        self.algo_name = algo_name
        self.problem_type = problem_type
        self.requirements = requirements
        self.objective = objective
        self.implementation = implementation
        self.description = description
        self.configuration = configuration or {}

    def __repr__(self):
        return f"name={self.algo_name} problem={self.problem_type}>"


def _normalize_requirements(requirements: Dict, source: str = "") -> Dict:
    """Drop None/empty entries from type/features lists (stray YAML dashes).

    Warns with the card source so authoring errors surface instead of silently
    propagating a 'null' feature into the mapper.
    """
    out = {}
    for section, spec in (requirements or {}).items():
        spec = dict(spec or {})
        for key in ("type", "features"):
            if key in spec:
                raw = spec[key] or []
                cleaned = [x for x in raw if x]
                if len(cleaned) != len(raw):
                    print(f"{source}: dropped empty entry in {section}.{key}")
                if cleaned:
                    spec[key] = cleaned
                else:
                    spec.pop(key)
        out[section] = spec
    return out


def load_algo_card(path: str | Path) -> AlgorithmCard:
    path = Path(path)

    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return AlgorithmCard(
        algo_name=data["algo_name"],
        problem_type=data["problem_type"],
        requirements=_normalize_requirements(data.get("requirements", {}), source=path.name),
        objective=data.get("objective"),
        implementation=data.get("implementation", {}),
        description=data.get("description"),
        configuration=data.get("configuration"),
    )


def load_algo_cards(directory: str | Path) -> List[AlgorithmCard]:
    algo_cards = []
    for filename in os.listdir(directory):
        if filename == CONFIGS_FILENAME:
            continue
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            full_path = os.path.join(directory, filename)
            try:
                model_card = load_algo_card(full_path)
                algo_cards.append(model_card)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return algo_cards


# ---------------------------------------------------------------------------
# Requirement merging
#
# Semantics:
#   - type lists:  intersection ('any' acts as the universal set)
#   - features:    union
#   - constraints: union with conflict detection / tightening
#   - sections:    union (a section required by any contributor is required)
# ---------------------------------------------------------------------------

_ANY = "any"


def _merge_types(a: list, b: list) -> list:
    a = [t for t in (a or []) if t] or [_ANY]
    b = [t for t in (b or []) if t] or [_ANY]
    if _ANY in a:
        return sorted(set(b))
    if _ANY in b:
        return sorted(set(a))
    inter = set(a) & set(b)
    if not inter:
        raise RequirementConflict(f"type conflict: {a} vs {b}")
    return sorted(inter)


def _merge_features(a: list, b: list) -> list:
    seen, out = set(), []
    for f in list(a or []) + list(b or []):
        if not f:  # drop None / empty entries from stray YAML dashes
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _merge_constraint(key: str, a: dict, b: dict) -> dict:
    merged = dict(a)
    for op, val in b.items():
        if op not in merged:
            merged[op] = val
            continue
        cur = merged[op]
        if op == "equals":
            if cur != val:
                raise RequirementConflict(f"{key}: equals {cur} vs {val}")
        elif op == "greater_than":
            merged[op] = max(cur, val)
        elif op == "less_than":
            merged[op] = min(cur, val)
        elif cur != val:
            raise RequirementConflict(f"{key}.{op}: {cur} vs {val}")
    if "equals" in merged:
        eq = merged["equals"]
        if "greater_than" in merged and not eq > merged["greater_than"]:
            raise RequirementConflict(f"{key}: equals {eq} violates greater_than {merged['greater_than']}")
        if "less_than" in merged and not eq < merged["less_than"]:
            raise RequirementConflict(f"{key}: equals {eq} violates less_than {merged['less_than']}")
        merged = {"equals": eq}
    return merged


def _merge_section(name: str, a: dict, b: dict) -> dict:
    a, b = a or {}, b or {}
    out: dict = {"type": _merge_types(a.get("type"), b.get("type"))}
    feats = _merge_features(a.get("features"), b.get("features"))
    if feats:
        out["features"] = feats
    cons = dict(a.get("constraints") or {})
    for key, spec in (b.get("constraints") or {}).items():
        cons[key] = _merge_constraint(f"{name}.{key}", cons.get(key, {}), spec)
    if cons:
        out["constraints"] = cons
    return out


def merge_requirements(a: Dict, b: Dict) -> Dict:
    out = dict(a or {})
    for section, spec_b in (b or {}).items():
        out[section] = _merge_section(section, out.get(section), spec_b) if section in out else spec_b
    return out


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


def resolve_configuration(
    base: AlgorithmCard,
    configuration: Dict[str, str],
    name: str,
    cards_by_name: Dict[str, AlgorithmCard],
) -> AlgorithmCard:
    """Resolve a configuration (an assignment of the card's options) into an executable card.

    configuration maps option name -> card algo_name (component options) or option value (value options).
    """
    requirements = base.requirements or {}
    selected_values: Dict[str, str] = {}

    unknown = set(configuration) - set(base.configuration)
    if unknown:
        raise RequirementConflict(
            f"{name}: unknown option(s) {sorted(unknown)} (declared: {sorted(base.configuration)})")
    missing = set(base.configuration) - set(configuration)
    if missing:
        raise RequirementConflict(f"{name}: no value selected for option(s): {sorted(missing)}")

    for option, value in configuration.items():
        spec = base.configuration[option]

        if "values" in spec:  # value option
            if value not in spec["values"]:
                raise RequirementConflict(
                    f"{name}: '{value}' not a valid value for option '{option}' "
                    f"(values: {sorted(spec['values'])})")
            value_spec = spec["values"][value] or {}
            requirements = merge_requirements(requirements, value_spec.get("requirements", {}))
            selected_values[option] = value

        elif "problem_type" in spec:  # component option
            comp = cards_by_name.get(value)
            if comp is None:
                raise RequirementConflict(f"{name}: no card named '{value}' for option '{option}'")
            if comp.configuration:
                raise RequirementConflict(
                    f"{name}: option '{option}' refers to card '{value}' with unresolved "
                    f"options {sorted(comp.configuration)}; refer to a resolved configuration instead")
            if comp.problem_type != spec["problem_type"]:
                raise RequirementConflict(
                    f"{name}: option '{option}' expects problem_type={spec['problem_type']}, "
                    f"got {comp.problem_type}")
            if base.objective and comp.objective and comp.objective != base.objective:
                raise RequirementConflict(
                    f"{name}: objective mismatch: base={base.objective}, {comp.algo_name}={comp.objective}")
            requirements = merge_requirements(requirements, comp.requirements)
            selected_values[option] = comp.implementation["class_name"]

        else:
            raise RequirementConflict(
                f"{base.algo_name}: option '{option}' declares neither 'problem_type' nor 'options'")

    implementation = {
        "class_name": base.implementation["class_name"],
        "component_name": name,
        **selected_values,
        "type": base.implementation.get("type", "heuristic"),
    }

    description = (
        (base.description or "").strip()
        + " configured with "
        + ", ".join(f"{s}={v}" for s, v in configuration.items())
    )

    return AlgorithmCard(
        algo_name=name,
        problem_type=base.problem_type,
        requirements=requirements,
        objective=base.objective,
        implementation=implementation,
        description=description,
    )


def resolve_configured_cards(
    cards: List[AlgorithmCard],
    configurations: List[Dict],
) -> List[AlgorithmCard]:
    """Resolve all manifest entries against the card library."""
    by_name = {c.algo_name: c for c in cards}
    resolved = []
    for cfg in configurations:
        base = by_name[cfg["base"]]
        resolved.append(resolve_configuration(base, cfg["configuration"], cfg["name"], by_name))
    return resolved


def load_configurations(directory: str | Path) -> List[Dict]:
    path = Path(directory) / CONFIGS_FILENAME
    if not path.exists():
        return []
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data.get("configurations", [])


def import_algo_class(cls_name: str, module_path: str | Path):
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def flatten_data_card(domain: dict) -> dict:
    features = {}
    for obj in domain.get("objects", []):
        for feat in obj.get("features", []):
            name = feat["name"]
            if "value" in feat:
                features[name] = feat["value"]
            else:
                features[name] = True
        # recurse into nested objects
        features.update(flatten_data_card(obj))
    return features


def create_domain_from_data_card(domain_class, type_enum, domain_data: Dict) -> BaseDomainObject:
    """Factory to create domain object from data card."""
    domain_type = type_enum(domain_data["type"][0])
    features = domain_data.get("features", {})

    # Create empty domain object
    domain = domain_class(tpe=domain_type)

    # Override get_features to return the dict
    domain.get_features = lambda: features

    return domain


def load_packaged_algo_cards() -> List[AlgorithmCard]:
    cards_resource = files("ware_ops_algos").joinpath(
        "algorithms",
        "algorithm_cards",
    )

    with as_file(cards_resource) as cards_path:
        all_cards = load_algo_cards(cards_path)
        configurations = load_configurations(cards_path)

    resolved = resolve_configured_cards(all_cards, configurations)
    fully_specified = [c for c in all_cards if not c.configuration]
    return fully_specified + resolved


if __name__ == "__main__":
    # Inspection entry point: dump all cards as the mapper will see them.
    for card in load_packaged_algo_cards():
        print(yaml.dump(
            {k: v for k, v in vars(card).items() if v},
            sort_keys=False, allow_unicode=True, width=100), "---")