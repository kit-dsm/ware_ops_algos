from pathlib import Path

from ware_ops_algos.algorithms import GreedyItemAssignment, OrderNrFifoBatching
from ware_ops_algos.data_loaders import FoodmartLoader


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = ROOT / "data" / "instances" / "FoodmartData"
INSTANCE = "instances_d5_ord5_MAL.txt"


def load_domain():
    return FoodmartLoader(INSTANCE_DIR).load(INSTANCE, use_cache=False)


def main() -> None:
    domain = load_domain()

    assignment = GreedyItemAssignment(domain.storage)
    assignment_solution = assignment.solve(domain.orders.orders)

    batching = OrderNrFifoBatching(
        pick_cart=domain.resources.resources[0].pick_cart,
        articles=domain.articles,
    )
    batching_solution = batching.solve(assignment_solution.resolved_orders)

    print(f"Loaded {len(domain.orders.orders)} orders from {INSTANCE}")
    print(f"Created {len(batching_solution.batches)} batches")
    print(f"Item assignment runtime: {assignment_solution.execution_time:.6f} s")
    print(f"Batching runtime: {batching_solution.execution_time:.6f} s")


if __name__ == "__main__":
    main()
