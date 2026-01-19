from abc import ABC
from dataclasses import dataclass, field

from ware_ops_algos.algorithms import Algorithm, AlgorithmSolution, PickList, PickerAssignment, AssignmentSolution
from ware_ops_algos.domain_models import Resource


class Assigner(Algorithm[list[PickList], AssignmentSolution], ABC):
    def __init__(self, resources: list[Resource]):
        super().__init__()
        self.resources = resources


class RoundRobinAssigner(Assigner):
    def _run(self, input_data: list[PickList]) -> AssignmentSolution:
        pick_lists = input_data
        pickers = self.resources
        print("Pick List", input_data)
        assignments = [
            PickerAssignment(pick_list=pick_list, picker=pickers[i % len(pickers)])
            for i, pick_list in enumerate(pick_lists)
        ]
        return AssignmentSolution(assignments=assignments)
