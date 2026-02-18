from typing import Any, Dict, List
import operator

from ware_ops_algos.domain_models.taxonomy import SUBPROBLEMS
from ware_ops_algos.utils.general_functions import load_model_card


class ConstraintEvaluator:
    """
    Evaluates feature constraints on domain objects.

    Supports:
    - Comparison: equals, not_equals, greater_than, less_than, etc.
    - Set operations: in, not_in
    - Logical combinations: and, or
    """

    OPERATORS = {
        'equals': operator.eq,
        'not_equals': operator.ne,
        'greater_than': operator.gt,
        'greater_than_or_equal': operator.ge,
        'less_than': operator.lt,
        'less_than_or_equal': operator.le,
        'in': lambda x, y: x in y,
        'not_in': lambda x, y: x not in y,
    }

    def evaluate(self, actual_value: Any, constraint: Dict[str, Any]) -> bool:
        """
        Evaluate a constraint against an actual feature value.

        Args:
            actual_value: The actual value from the domain
            constraint: The constraint specification

        Returns:
            True if constraint is satisfied, False otherwise
        """
        # Simple value constraint (backward compatibility)
        if not isinstance(constraint, dict):
            return actual_value == constraint

        # Logical AND: all conditions must be true
        if 'and' in constraint:
            return all(self.evaluate(actual_value, c) for c in constraint['and'])

        # Logical OR: at least one condition must be true
        if 'or' in constraint:
            return any(self.evaluate(actual_value, c) for c in constraint['or'])

        # Comparison operators
        for operator_name, operator_func in self.OPERATORS.items():
            if operator_name in constraint:
                try:
                    return operator_func(actual_value, constraint[operator_name])
                except (TypeError, AttributeError):
                    return False

        return False


class AlgorithmFilter:
    """
    Filters algorithms by feasibility for a given warehouse problem instance.

    The filter checks whether algorithms can be executed on an instance by validating:
    1. Problem type compatibility (e.g., routing, batching, storage)
    2. Domain requirements (layout, resources, orders, storage features)
    3. Feature constraints (e.g., n_blocks = 1, n_aisles > 2)

    Example:
        subproblems = {
            "routing": ["single_picker_routing", "tsp"],
            "batching": ["order_batching", "wave_planning"]
        }

        filter = AlgorithmFilter(subproblems)
        feasible_algos = filter.filter(all_algorithms, warehouse_instance)
    """

    def __init__(self, subproblems: Dict[str, List[str]]):
        """
        Initialize the algorithm filter.

        Args:
            subproblems: Maps each problem type to its subproblem variants.
        """
        self.subproblems = subproblems
        self.evaluator = ConstraintEvaluator()

    def filter(self, algorithms: List['ModelCard'], instance: 'BaseWarehouseDomain',
               verbose: bool = False) -> List['ModelCard']:
        """
        Filter algorithms to find those feasible for the given instance.

        Args:
            algorithms: List of all available algorithms
            instance: The warehouse problem instance
            verbose: If True, print detailed filtering information

        Returns:
            List of algorithms that are feasible for this instance
        """
        problem_compatible = self._filter_by_problem(algorithms, instance.problem_class, verbose)

        feasible_algorithms = self._filter_by_requirements(problem_compatible, instance, verbose)

        if verbose:
            print(f"\nFinal result: {len(feasible_algorithms)}/{len(algorithms)} algorithms are feasible")

        return feasible_algorithms

    def _filter_by_problem(self, algorithms: List['ModelCard'], problem_type: str,
                           verbose: bool = False) -> List['ModelCard']:
        """
        Filter algorithms by problem type compatibility.

        An algorithm is compatible if it solves the main problem or any of its subproblems.
        Example: For problem "routing", accept algorithms that solve "routing",
                 "single_picker_routing", or "tsp".

        Args:
            algorithms: List of algorithms to filter
            problem_type: The problem type (e.g., "routing", "batching")
            verbose: If True, print filtering details

        Returns:
            Algorithms compatible with the problem type
        """
        valid_problem_types = {problem_type} | set(self.subproblems.get(problem_type, {}).get("variables", []))
        compatible_algorithms = [algo for algo in algorithms if algo.problem_type in valid_problem_types]

        if verbose:
            print(f"Problem type filtering for '{problem_type}':")
            print(f"  Accepted types: {valid_problem_types}")
            print(f"  Result: {len(compatible_algorithms)}/{len(algorithms)} algorithms match")

        return compatible_algorithms

    def _filter_by_requirements(self, algorithms: List['ModelCard'],
                                instance: 'BaseWarehouseDomain',
                                verbose: bool = False) -> List['ModelCard']:
        """
        Filter algorithms by domain requirements.

        Checks for each algorithm:
        1. Objective matches (e.g., minimize distance vs minimize time)
        2. For each required domain (layout, resources, orders, storage):
           - Domain type matches (e.g., conventional layout vs fishbone)
           - Required features exist (e.g., start_node, n_aisles)
           - Feature constraints are satisfied (e.g., n_blocks = 1)

        Args:
            algorithms: Algorithms to check
            instance: The warehouse instance
            verbose: If True, print detailed checks

        Returns:
            Algorithms that satisfy all requirements
        """
        feasible_algorithms = []

        for algo in algorithms:
            if self._is_feasible(algo, instance, verbose):
                feasible_algorithms.append(algo)

        if verbose:
            print(f"\nRequirement filtering: {len(feasible_algorithms)}/{len(algorithms)} algorithms feasible")

        return feasible_algorithms

    def _is_feasible(self, algorithm: 'ModelCard', instance: 'BaseWarehouseDomain',
                     verbose: bool = False) -> bool:
        """
        Check if an algorithm is feasible for the given instance.

        Args:
            algorithm: The algorithm to check
            instance: The warehouse instance
            verbose: If True, print check details

        Returns:
            True if algorithm can be executed on this instance
        """
        if verbose:
            print(f"\n  Checking: {algorithm.model_name}")

        # Check objective compatibility
        if algorithm.objective != instance.objective:
            if verbose:
                print(f"    ✗ Objective mismatch: algorithm needs '{algorithm.objective}', "
                      f"instance has '{instance.objective}'")
            return False

        # Check each domain requirement (layout, resources, orders, storage)
        for domain_name, requirements in algorithm.requirements.items():
            domain = getattr(instance, domain_name)

            if not self._check_domain_requirements(domain_name, domain, requirements, verbose):
                return False

        if verbose:
            print(f"    ✓ Algorithm is feasible")

        return True

    def _check_domain_requirements(self, domain_name: str, domain: Any,
                                   requirements: Dict, verbose: bool = False) -> bool:
        """
        Check if a domain object satisfies the algorithm's requirements.

        Args:
            domain_name: Name of the domain (e.g., "layout", "resources")
            domain: The actual domain object from the instance
            requirements: Algorithm's requirements for this domain
            verbose: If True, print check details

        Returns:
            True if all requirements are satisfied
        """
        # Get domain properties
        try:
            domain_type = domain.get_type_value()
        except:
            domain_type = domain["type"]
        try:
            domain_features = domain.get_features()
        except:
            domain_features = domain["features"]

        # Get algorithm requirements
        required_types = requirements.get("type", [])
        required_features = requirements.get("features", [])
        required_features = [] if required_features in (None, [None]) else required_features
        feature_constraints = requirements.get("constraints", {})

        # Check 1: Domain type must match
        if "any" not in required_types and domain_type not in required_types:
            if verbose:
                print(f"    ✗ {domain_name}: type '{domain_type}' not in required types {required_types}")
            return False

        # Check 2: All required features must exist
        missing_features = [feat for feat in required_features if feat not in domain_features]
        if missing_features:
            if verbose:
                print(f"    ✗ {domain_name}: missing required features {missing_features}")
            return False

        # Check 3: Feature constraints must be satisfied
        for feature_name, constraint in feature_constraints.items():
            if feature_name not in domain_features:
                if verbose:
                    print(f"    ✗ {domain_name}: feature '{feature_name}' needed for constraint not found")
                return False

            actual_value = domain_features[feature_name]
            if not self.evaluator.evaluate(actual_value, constraint):
                if verbose:
                    print(f"    ✗ {domain_name}: constraint violated - {feature_name}={actual_value} "
                          f"does not satisfy {constraint}")
                return False

        return True


# Backward compatibility with old function names
def match_instance_solver(models: List['ModelCard'], predicate_func: Dict,
                          problem: str) -> List['ModelCard']:
    """
    Legacy function: Problem-based filtering only.
    Deprecated: Use AlgorithmFilter.filter() instead.
    """
    filter = AlgorithmFilter(predicate_func)
    return filter._filter_by_problem(models, problem, verbose=False)


def match_ontology(models: List['ModelCard'], instance: 'BaseWarehouseDomain',
                   verbose: bool = False) -> List['ModelCard']:
    """
    Legacy function: Requirement-based filtering only.
    Deprecated: Use AlgorithmFilter.filter() instead.
    """
    filter = AlgorithmFilter({})
    return filter._filter_by_requirements(models, instance, verbose)


# Example usage
if __name__ == "__main__":
    print("AlgorithmFilter - Filters algorithms by feasibility for warehouse instances")
    print("\nExample usage:")
    print("-" * 60)
    print("""
    # 1. Define problem hierarchy
    subproblems = {
        "routing": ["single_picker_routing", "tsp"],
        "batching": ["order_batching", "wave_planning"]
    }

    # 2. Create filter
    filter = AlgorithmFilter(subproblems)

    # 3. Load algorithms and instance
    algorithms = load_model_cards("path/to/algorithms/")
    warehouse = load_warehouse_instance("instance.yaml")

    # 4. Filter for feasible algorithms
    feasible = filter.filter(algorithms, warehouse, verbose=True)

    # 5. Use feasible algorithms
    for algo in feasible:
        print(f"Can use: {algo.model_name}")
    """)

    print("\n" + "=" * 60)
    print("ConstraintEvaluator Examples")
    print("=" * 60)

    evaluator = ConstraintEvaluator()

    algo1 = load_model_card("../src/project_4D4L/algorithms/opt_model_cards/sprp_dp.yaml")
    algo2 = load_model_card("../src/project_4D4L/algorithms/opt_model_cards/sshape_routing.yaml")

    algorithms = [algo1, algo2]

    layout_constraints = algo1.requirements["layout"]["constraints"]
    n_blocks_constraint = layout_constraints["n_blocks"]  # {'equals': 1}

    # Test against actual values
    actual_value = 1
    result = evaluator.evaluate(actual_value, n_blocks_constraint)
    print(f"Value {actual_value} satisfies constraint {layout_constraints}: {result}")

    actual_value = 2
    result = evaluator.evaluate(actual_value, n_blocks_constraint)
    print(f"Value {actual_value} satisfies constraint {layout_constraints}: {result}")
    print("Layout constraints:")
    for feature_name, constraint in layout_constraints.items():
        print(f"  {feature_name}: {constraint}")

    # domain = build_domain_from_order_files(order_list_path="../../../data/instances/IOPVRP/OrderList_LargeProblems_16_4.txt",
    #                                        order_line_list_path="../../../data/instances/IOPVRP/OrderLineList_LargeProblems_16_4.txt")
    #
    # algo_filter = AlgorithmFilter(SUBPROBLEMS)
    # feasible = algo_filter.filter(algorithms=algorithms, instance=domain, verbose=True)
    # print(feasible)

