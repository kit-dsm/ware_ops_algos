from ware_ops_algos.algorithms import GreedyItemAssignment, OrderNrFifoBatching

from model_domain import build_domain


def batch_load(batch, articles):
    article_by_id = {article.article_id: article for article in articles.articles}
    items = sum(position.in_store for position in batch.pick_positions)
    weight = sum(
        position.in_store * article_by_id[position.article_id].weight
        for position in batch.pick_positions
    )
    volume = sum(
        position.in_store * article_by_id[position.article_id].volume
        for position in batch.pick_positions
    )
    return items, weight, volume


def main() -> None:
    domain = build_domain()
    resolved_orders = GreedyItemAssignment(domain.storage).solve(
        domain.orders.orders
    ).resolved_orders

    batching = OrderNrFifoBatching(
        pick_cart=domain.resources.resources[0].pick_cart,
        articles=domain.articles,
    )
    solution = batching.solve(resolved_orders)

    assert all(batching.capacity_checker.orders_fit(batch.orders) for batch in solution.batches)

    for batch in solution.batches:
        order_ids = [order.order_id for order in batch.orders]
        items, weight, volume = batch_load(batch, domain.articles)
        print(
            f"Batch {batch.batch_id}: orders={order_ids}, "
            f"items={items}, weight={weight:g}, volume={volume:g}"
        )


if __name__ == "__main__":
    main()
