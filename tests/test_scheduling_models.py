import pytest

from ware_ops_algos.algorithms import OrderNrFifoBatching, SPTScheduler, build_jobs, BatchObject, Job
from ware_ops_algos.domain_models import Resources


def build_jobs_from_batches(
    batches: list[BatchObject],
    resources: Resources,
) -> list[Job]:
    picker = resources.resources[0]
    time_per_pick = picker.time_per_pick
    return [
        Job(
            job_id=i,
            processing_time=len(batch.pick_positions) * time_per_pick,
            release_time=0,
            due_date=batch.earliest_due_date,
            n_picks=len(batch.pick_positions),
            route=None,
            batch=batch,
        )
        for i, batch in enumerate(batches)
    ]

@pytest.mark.benchmark(group="list_scheduling")
def test_fifo_batching_kris(benchmark, kris_domain, kris_resolved_orders):
    pick_cart = kris_domain.resources.resources[0].pick_cart
    articles = kris_domain.articles
    resources = kris_domain.resources
    batching_algo = OrderNrFifoBatching(pick_cart=pick_cart, articles=articles)
    batching_sol = batching_algo.solve(kris_resolved_orders)

    batches = batching_sol.batches
    scheduler = SPTScheduler(resources=resources)
    jobs = build_jobs_from_batches(batches, resources)
    scheduling_sol = scheduler.solve(jobs)
    print(scheduling_sol)
    # result = benchmark(algo.solve, kris_resolved_orders)
