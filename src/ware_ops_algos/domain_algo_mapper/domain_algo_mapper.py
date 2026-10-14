from typing import Any, Dict, List
import operator

from ware_ops_algos.domain_models import BaseWarehouseDomain, DataCard
from ware_ops_algos.algorithms.algorithm_cards import AlgorithmCard


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
        'contains': lambda x, y: y in x,
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


class DomainAlgorithmMapper:
    """
    Map algorithm cards to a warehouse instance or flattened data card.

    A card applies when its problem type occurs in the problem's taxonomy and
    each required domain has a compatible type, all named features, and values
    satisfying its constraints. ``rejection_reasons`` exposes every failed
    check; ``filter`` returns cards with no reasons. In a flattened data card,
    a feature whose value is ``None`` counts as absent.

    Example:
        subproblems = {
            "routing": ["single_picker_routing", "tsp"],
            "batching": ["order_batching", "wave_planning"]
        }

        mapper = DomainAlgorithmMapper(subproblems)
        feasible_algos = mapper.filter(all_algorithms, warehouse_instance)
    """

    def __init__(self, subproblems: Dict[str, List[str]]):
        """
        Initialize the algorithm filter.

        Args:
            subproblems: Maps each problem type to its subproblem variants.
        """
        self.subproblems = subproblems
        self.evaluator = ConstraintEvaluator()

    def filter(self, algorithms: List['AlgorithmCard'], instance: BaseWarehouseDomain | DataCard,
               verbose: bool = False) -> List['AlgorithmCard']:
        """
        Filter algorithms to find those feasible for the given instance.

        Args:
            algorithms: List of all available algorithms
            instance: The warehouse problem instance
            verbose: If True, print detailed filtering information

        Returns:
            List of algorithms that are feasible for this instance
        """
        feasible_algorithms = []
        for algorithm in algorithms:
            reasons = self.rejection_reasons(algorithm, instance)
            if verbose:
                print(f"{algorithm.algo_name}: {', '.join(reasons) if reasons else 'feasible'}")
            if not reasons:
                feasible_algorithms.append(algorithm)

        if verbose:
            print(f"\nFinal result: {len(feasible_algorithms)}/{len(algorithms)} algorithms are feasible")

        return feasible_algorithms

    def rejection_reasons(self, algorithm: AlgorithmCard,
                          instance: BaseWarehouseDomain | DataCard) -> list[str]:
        """Explain every failed applicability check for one algorithm card.

        Args:
            algorithm: Resolved card whose problem type and requirements are checked.
            instance: Warehouse instance or flattened data card. A dictionary
                feature set to ``None`` is absent; missing object domains are
                reported as reasons.

        Returns:
            Reasons for problem type, domain type, missing feature, or failed
            constraint checks; an empty list means the card applies.
        """
        valid = {instance.problem_class} | set(
            self.subproblems.get(instance.problem_class, {}).get("variables", []))
        reasons = []
        if algorithm.problem_type not in valid:
            reasons.append(f"problem type {algorithm.problem_type!r} not in {sorted(valid)}")
        for domain_name, requirements in algorithm.requirements.items():
            domain = getattr(instance, domain_name, None)
            if domain is None:
                reasons.append(f"missing domain {domain_name}")
                continue
            if isinstance(domain, dict):
                domain_type = domain["type"]
                features = {name: value for name, value in domain["features"].items()
                            if value is not None}
            else:
                domain_type = domain.get_type_value()
                features = domain.get_features()
            required_types = requirements.get("type", [])
            if "any" not in required_types and domain_type not in required_types:
                reasons.append(f"{domain_name}.type {domain_type!r} not in {required_types}")
            required_features = requirements.get("features", [])
            required_features = [] if required_features in (None, [None]) else required_features
            for feature in required_features:
                if feature not in features:
                    reasons.append(f"missing {domain_name}.{feature}")
            for feature, constraint in requirements.get("constraints", {}).items():
                if feature not in features:
                    reasons.append(f"missing {domain_name}.{feature}")
                elif not self.evaluator.evaluate(features[feature], constraint):
                    reasons.append(f"{domain_name}.{feature}={features[feature]!r} "
                                   f"does not satisfy {constraint}")
        return reasons
