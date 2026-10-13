"""Small, standalone DAG executor for fixed compositions of existing algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from ware_ops_algos.algorithms.algorithm_interfaces import Algorithm, AlgorithmSolution


@dataclass(frozen=True)
class ValueRef:
    pipeline: "Pipeline"
    source: str
    name: str
    attribute: str | None = None


class InputRefs:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def __getattr__(self, name: str) -> ValueRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return ValueRef(self.pipeline, "input", name)


class StepRef:
    def __init__(self, pipeline: "Pipeline", name: str):
        self.pipeline = pipeline
        self.name = name

    def __getattr__(self, attribute: str) -> ValueRef:
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        return ValueRef(self.pipeline, "step", self.name,
                        None if attribute in {"solution", "value"} else attribute)


@dataclass(frozen=True)
class PipelineRun:
    solution: Any
    stages: Mapping[str, Any]
    runtime_seconds: float
    stage_seconds: Mapping[str, float]

    @property
    def components(self) -> tuple[str, ...]:
        seen: set[int] = set()
        names = []
        for result in self.stages.values():
            if isinstance(result, AlgorithmSolution) and id(result) not in seen:
                names.append(result.algo_name)
                seen.add(id(result))
        return tuple(names)


class Pipeline:
    """Explicit acyclic data flow; solver interfaces and results stay intact."""

    def __init__(self, name: str):
        self.name = name
        self.input = InputRefs(self)
        self._steps: list[tuple[str, Any, dict[str, Any]]] = []
        self._output: ValueRef | None = None

    def _check_ref(self, ref: ValueRef) -> None:
        if ref.pipeline is not self:
            raise ValueError("A pipeline cannot consume another pipeline's reference")
        if ref.source == "step" and ref.name not in {name for name, _, _ in self._steps}:
            raise ValueError(f"Unknown or forward step reference {ref.name!r}")

    def step(self, name: str, method: Any, **inputs: Any) -> StepRef:
        if not name or any(existing == name for existing, _, _ in self._steps):
            raise ValueError(f"Duplicate or empty pipeline step {name!r}")
        if not inputs or not (isinstance(method, Algorithm) or callable(method)):
            raise TypeError("A step needs input bindings and an algorithm or callable")
        for value in inputs.values():
            if isinstance(value, ValueRef):
                self._check_ref(value)
        if isinstance(method, Algorithm) and set(inputs) != {"input"}:
            raise ValueError("An Algorithm step accepts exactly one 'input' binding")
        self._steps.append((name, method, inputs))
        return StepRef(self, name)

    def output(self, value: ValueRef) -> None:
        self._check_ref(value)
        self._output = value

    def _resolve(self, value: Any, inputs: Mapping[str, Any],
                 stages: Mapping[str, Any]) -> Any:
        if not isinstance(value, ValueRef):
            return value
        if value.source == "input":
            if value.name not in inputs:
                raise ValueError(f"Missing pipeline input {value.name!r}")
            return inputs[value.name]
        result = stages[value.name]
        if value.attribute is None:
            return result
        try:
            return getattr(result, value.attribute)
        except AttributeError as error:
            raise ValueError(f"Step {value.name!r} has no output {value.attribute!r}") from error

    def run(self, **inputs: Any) -> PipelineRun:
        if self._output is None:
            raise ValueError("Pipeline output was not selected")
        started = perf_counter()
        stages: dict[str, Any] = {}
        stage_seconds: dict[str, float] = {}
        for name, method, bindings in self._steps:
            values = {key: self._resolve(value, inputs, stages)
                      for key, value in bindings.items()}
            step_started = perf_counter()
            stages[name] = (method.solve(values["input"])
                            if isinstance(method, Algorithm) else method(**values))
            stage_seconds[name] = perf_counter() - step_started
        return PipelineRun(self._resolve(self._output, inputs, stages),
                           stages, perf_counter() - started, stage_seconds)
