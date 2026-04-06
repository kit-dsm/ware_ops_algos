from pathlib import Path
import pytest

from ware_ops_algos.algorithms import GreedyItemAssignment
from ware_ops_algos.data_loaders import HesslerIrnichLoader

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"
CACHE_DIR = INSTANCES_DIR / "caches"


def _load_domain(instance_set: str, instance_file: str):
    loader = HesslerIrnichLoader(
        instances_dir=INSTANCES_DIR / instance_set,
        cache_dir=CACHE_DIR / instance_set,
    )
    return loader.load(instance_file)


def _resolved_orders(domain):
    ia = GreedyItemAssignment(domain.storage)
    return ia.solve(domain.orders.orders).resolved_orders


# --- SPRP ---

@pytest.fixture(scope="session")
def sprp_domain():
    return _load_domain("SPRP", "unit_F1_m5_C30_a7_12.txt")

@pytest.fixture(scope="session")
def sprp_resolved_orders(sprp_domain):
    return _resolved_orders(sprp_domain)


# --- BahceciOencan ---

@pytest.fixture(scope="session")
def bahceci_domain():
    return _load_domain("BahceciOencan", "Pr_20_1_20_Store1_01.txt")

@pytest.fixture(scope="session")
def bahceci_resolved_orders(bahceci_domain):
    return _resolved_orders(bahceci_domain)


# --- MuterOencan ---

@pytest.fixture(scope="session")
def muter_domain():
    return _load_domain("MuterOencan", "100_48_5.txt")

@pytest.fixture(scope="session")
def muter_resolved_orders(muter_domain):
    return _resolved_orders(muter_domain)


# --- HennWaescherUniform ---

@pytest.fixture(scope="session")
def henn_domain():
    return _load_domain("HennWaescherUniform", "1l-20-30-0.txt")

@pytest.fixture(scope="session")
def henn_resolved_orders(henn_domain):
    return _resolved_orders(henn_domain)