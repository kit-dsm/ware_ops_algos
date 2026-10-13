"""Every packaged card must point to an implemented algorithm of its subproblem."""

import importlib
import inspect
import pkgutil
from functools import lru_cache

import pytest

import ware_ops_algos.algorithms as algorithms
from ware_ops_algos.algorithms.admission import Admission
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.algorithms.batching import Batching
from ware_ops_algos.algorithms.item_assignment import ItemAssignment
from ware_ops_algos.algorithms.positioning import Positioning
from ware_ops_algos.algorithms.routing import Routing
from ware_ops_algos.algorithms.routing_and_batching import RoutingBatchingAssigning
from ware_ops_algos.algorithms.scheduling import Scheduler
from ware_ops_algos.algorithms.waiting import Waiting


BASES = {
    "admission": Admission,
    "batching": Batching,
    "batching_routing": RoutingBatchingAssigning,
    "item_assignment": ItemAssignment,
    "positioning": Positioning,
    "routing": Routing,
    "active_tour_routing": Routing,
    "scheduling": Scheduler,
    "waiting": Waiting,
}


@lru_cache(maxsize=1)
def _implemented_classes() -> dict[str, type]:
    classes = {}
    for module_info in pkgutil.walk_packages(
        algorithms.__path__, prefix=algorithms.__name__ + "."
    ):
        module = importlib.import_module(module_info.name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__:
                classes[name] = cls
    return classes


@pytest.mark.parametrize("card", load_packaged_algo_cards(),
                         ids=lambda card: card.algo_name)
def test_card_names_an_implemented_subproblem_class(card):
    class_name = card.implementation["class_name"]
    expected = BASES[card.problem_type]
    cls = _implemented_classes().get(class_name)
    assert cls is not None, f"{class_name} is not implemented; expected {expected.__name__}"
    assert issubclass(cls, expected), (
        f"{class_name} must subclass {expected.__name__} for {card.problem_type}"
    )
