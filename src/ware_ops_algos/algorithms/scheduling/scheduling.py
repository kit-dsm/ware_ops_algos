import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ware_ops_algos.algorithms import Algorithm, I, O
from ware_ops_algos.algorithms.algorithm import SchedulingSolution, Job, Route, AlgorithmSolution, PickList
from ware_ops_algos.domain_models import OrdersDomain, Resources, Resource, Order


@dataclass
class SequencingInput:
    pick_lists: list[PickList]
    routes: list[Route]


@dataclass
class Sequencing:
    sequenced_pick_lists: list[Route]


@dataclass
class SequencingSolution(AlgorithmSolution):
    sequencing: Optional[list[PickList]] = None


# PickList vs Route Sequencing
class PickListSequencer(Algorithm[list[PickList], SequencingSolution], ABC):
    """Performs sequencing of pick list: takes pick list, returns them in sorted order."""

    def __init__(self, orders: OrdersDomain):
        super().__init__()
        self.orders_domain = orders


# class EDDSequencer(PickListSequencer):
#     def _sort_pick_lists(self, pick_lists: list[PickList]):
#         return sorted(pick_lists, key=lambda j: j.earliest_due_date)
#
#     def _run(self, input_data: list[PickList]) -> SequencingSolution:
#         sorted_pick_lists = self._sort_pick_lists(input_data)
#
#         return SequencingSolution(sequencing=sorted_pick_lists)


@dataclass
class SequencingInput:
    routes: list[Route]
    resources: Resources


@dataclass
class SequencingSolution(AlgorithmSolution):
    sequencing: Optional[list[Route]] = None


class SPTSequencer(Algorithm[SequencingInput, SchedulingSolution]):
    algo_name = "SPTSequencer"

    @staticmethod
    def _processing_time(route: Route, picker: Resource) -> float:
        if picker.speed is None or picker.speed <= 0:
            raise ValueError(f"Resource {picker.id} needs a positive 'speed'.")
        if picker.time_per_pick is None:
            raise ValueError(f"Resource {picker.id} needs 'time_per_pick'.")

        n_picks = len(route.item_sequence)
        setup_time = picker.tour_setup_time or 0.0

        return (
            route.distance / picker.speed
            + picker.time_per_pick * n_picks
            + setup_time
        )

    def _run(self, input_data: SequencingInput) -> SchedulingSolution:
        picker = input_data.resources.resources[0]

        sorted_routes = sorted(
            input_data.routes,
            key=lambda route: self._processing_time(route, picker),
        )

        jobs = []
        for idx, route in enumerate(sorted_routes):
            pl = route.pick_list
            pt = self._processing_time(route, picker)
            release = pl.release if pl.release is not None else 0.0

            jobs.append(Job(
                batch_idx=idx,
                picker_id=picker.id,
                start_time=release,
                end_time=release + pt,
                release_time=release,
                distance=route.distance,
                n_picks=len(route.item_sequence),
                travel_time=route.distance / picker.speed,
                handling_time=picker.time_per_pick * len(route.item_sequence),
                route=route,
            ))

        return SchedulingSolution(
            jobs=jobs,
            algo_name=self.algo_name,
        )


class EDDSequencer(Algorithm[SequencingInput, SchedulingSolution]):
    algo_name = "EDDSequencer"

    @staticmethod
    def _processing_time(route: Route, picker: Resource) -> float:
        if picker.speed is None or picker.speed <= 0:
            raise ValueError(f"Resource {picker.id} needs a positive 'speed'.")
        if picker.time_per_pick is None:
            raise ValueError(f"Resource {picker.id} needs 'time_per_pick'.")

        n_picks = len(route.item_sequence)
        setup_time = picker.tour_setup_time or 0.0

        return (
            route.distance / picker.speed
            + picker.time_per_pick * n_picks
            + setup_time
        )

    @staticmethod
    def _earliest_due_date(route: Route) -> float:
        pl = route.pick_list

        if pl.earliest_due_date is not None:
            return pl.earliest_due_date

        due_dates = [
            o.due_date
            for o in pl.orders
            if o.due_date is not None
        ]

        return min(due_dates) if due_dates else float("inf")

    def _run(self, input_data: SequencingInput) -> SchedulingSolution:
        picker = input_data.resources.resources[0]

        sorted_routes = sorted(
            input_data.routes,
            key=lambda route: self._earliest_due_date(route),
        )

        jobs = []
        for idx, route in enumerate(sorted_routes):
            pl = route.pick_list
            pt = self._processing_time(route, picker)
            release = pl.release if pl.release is not None else 0.0

            jobs.append(Job(
                batch_idx=idx,
                picker_id=picker.id,
                start_time=release,
                end_time=release + pt,
                release_time=release,
                distance=route.distance,
                n_picks=len(route.item_sequence),
                travel_time=route.distance / picker.speed,
                handling_time=picker.time_per_pick * len(route.item_sequence),
                route=route,
            ))

        return SchedulingSolution(
            jobs=jobs,
            algo_name=self.algo_name,
        )

@dataclass
class SchedulingInput:
    """
    Minimal info the scheduler needs.
    """
    routes: list[Route]
    orders: OrdersDomain
    resources: Resources


class PriorityScheduling(Algorithm[SchedulingInput, SchedulingSolution], ABC):
    """
    Schedule by: sort jobs once using a rule, then assign to earliest-free picker.
    Subclasses implement `_sorted_jobs`.
    """

    @abstractmethod
    def _sorted_jobs(self, jobs: list[dict], order_by_id: dict[int, Order], resources: Resources) -> list[dict]:
        pass

    def _run(self, input_data: SchedulingInput) -> SchedulingSolution:
        routes = input_data.routes
        orders = input_data.orders
        resources = input_data.resources

        # batch -> order_numbers
        batches_orders = []
        for route in routes:
            pl = route.pick_list
            order_nums = pl.order_numbers
            batches_orders.append(sorted(order_nums))

        order_by_id = {o.order_id: o for o in orders.orders}

        # build jobs
        jobs = []
        for idx, (route, order_nums) in enumerate(zip(routes, batches_orders)):
            jobs.append({
                "idx": idx,
                "solution": route,
                "order_numbers": order_nums,
                # "release": self._latest_order_arrival(order_nums, order_by_id),  # r_j
                # "first_due": self._first_due_date(order_nums, order_by_id),      # d_j^min
                "release": route.pick_list.release,
                "first_due": route.pick_list.earliest_due_date
            })

        # sort once by chosen rule
        jobs_sorted = self._sorted_jobs(jobs, order_by_id, resources)

        # earliest-free picker heap
        rheap = [(0.0, i, r) for i, r in enumerate(resources.resources)]
        heapq.heapify(rheap)

        jobs_scheduled: list[Job] = []
        for job in jobs_sorted:
            avail_time, ridx, picker = heapq.heappop(rheap)
            n_picks = len(job["solution"].item_sequence)
            pt = self._processing_time(job["solution"], picker, n_picks)

            start_time = max(job["release"], avail_time)
            if picker.tour_setup_time:
                start_time += picker.tour_setup_time
            end_time = start_time + pt

            travel_time = job["solution"].distance / picker.speed
            handling_time = picker.time_per_pick * n_picks

            jobs_scheduled.append(Job(
                batch_idx=job["idx"],
                picker_id=picker.id,
                start_time=start_time,
                end_time=end_time,
                release_time=job["release"],
                distance=job["solution"].distance,
                n_picks=n_picks,
                travel_time=travel_time,
                handling_time=handling_time,
                route=job["solution"],
            ))
            heapq.heappush(rheap, (end_time, ridx, picker))

        # assignments.sort(key=lambda a: a.pick_list_idx)
        return SchedulingSolution(jobs=jobs_scheduled, algo_name=self.algo_name)

    # ---- helpers ----
    @staticmethod
    def _latest_order_arrival(order_numbers: list[int], order_by_id: dict[int, Order]) -> float:
        ts = [order_by_id[on].order_date
              for on in order_numbers if on in order_by_id and order_by_id[on].order_date is not None]
        return max(ts) if ts else 0.0

    @staticmethod
    def _first_due_date(order_numbers: list[int], order_by_id: dict[int, Order]) -> float:
        ts = [order_by_id[on].due_date
              for on in order_numbers if on in order_by_id and order_by_id[on].due_date is not None]
        return min(ts) if ts else float("inf")

    @staticmethod
    def _processing_time(sol: Route, picker: Resource, n_picks: int) -> float:
        if picker.speed is None or picker.speed <= 0:
            raise ValueError(f"Resource {picker.id} needs a positive 'speed'.")
        if picker.time_per_pick is None:
            raise ValueError(f"Resource {picker.id} needs 'time_per_pick'.")
        return (sol.distance / picker.speed) + picker.time_per_pick * n_picks


class SPTScheduling(PriorityScheduling):
    """Shortest Processing Time first."""
    def _sorted_jobs(self, jobs, order_by_id, resources):
        rp = resources.resources[0]
        return sorted(jobs, key=lambda j: self._processing_time(j["solution"], rp, len(j["solution"].item_sequence)))


class LPTScheduling(PriorityScheduling):
    """Longest Processing Time first (often better for makespan)."""
    def _sorted_jobs(self, jobs, order_by_id, resources):
        rp = resources.resources[0]
        return sorted(jobs, key=lambda j: self._processing_time(j["solution"], rp, len(j["solution"].item_sequence)),
                      reverse=True)


class EDDScheduling(PriorityScheduling):
    """Earliest Due Date first (good for tardiness)."""
    def _sorted_jobs(self, jobs, order_by_id, resources):
        return sorted(jobs, key=lambda j: j["first_due"])


class ERDScheduling(PriorityScheduling):
    """Earliest Release Date first (start as soon as available)."""
    def _sorted_jobs(self, jobs, order_by_id, resources):
        return sorted(jobs, key=lambda j: j["release"])


class FIFOScheduling(PriorityScheduling):
    """By smallest order number in batch (proxy for input order)."""
    def _sorted_jobs(self, jobs, order_by_id, resources):
        return sorted(jobs, key=lambda j: (min(j["order_numbers"]) if j["order_numbers"] else float("inf")))