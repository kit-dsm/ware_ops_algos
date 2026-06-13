import importlib
from importlib.resources import as_file, files
import os
from typing import List, Dict
import yaml
from pathlib import Path

from ware_ops_algos.domain_models import BaseDomainObject

class AlgorithmCard:
    def __init__(self, algo_name: str, problem_type: str, requirements: Dict, objective: str, implementation: Dict):
        self.algo_name = algo_name
        self.problem_type = problem_type
        self.requirements = requirements
        self.objective = objective
        self.implementation = implementation

    def __repr__(self):
        return f"name={self.algo_name} problem={self.problem_type}>"


def load_algo_card(path: str | Path) -> AlgorithmCard:
    path = Path(path)
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return AlgorithmCard(
        algo_name=data["algo_name"],
        problem_type=data["problem_type"],
        requirements=data["requirements"],
        objective=data["objective"],
        implementation=data["implementation"]
    )


def load_algo_cards(directory: str | Path) -> List[AlgorithmCard]:
    algo_cards = []
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            full_path = os.path.join(directory, filename)
            try:
                model_card = load_algo_card(full_path)
                algo_cards.append(model_card)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return algo_cards


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
        return load_algo_cards(cards_path)