from ware_ops_algos.algorithms import GreedyItemAssignment, PriorityBatching

from getting_started import load_domain


class LargestOrderFirstBatching(PriorityBatching):
    algo_name = "LargestOrderFirstBatching"

    def _sorted_orders(self):
        return sorted(
            self.order_list,
            key=lambda order: -sum(position.amount for position in order.pick_positions),
        )


def main() -> None:
    domain = load_domain()
    resolved_orders = GreedyItemAssignment(domain.storage).solve(
        domain.orders.orders
    ).resolved_orders

    batching = LargestOrderFirstBatching(
        pick_cart=domain.resources.resources[0].pick_cart,
        articles=domain.articles,
    )
    solution = batching.solve(resolved_orders)

    for batch in solution.batches:
        order_ids = [order.order_id for order in batch.orders]
        print(f"Batch {batch.batch_id}: {order_ids}")


if __name__ == "__main__":
    main()
