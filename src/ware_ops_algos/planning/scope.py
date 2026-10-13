"""The state information a decision sees and its revision rights."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .lifecycle import Accepts, ResourceStates, TOUR_ORDER, TimeAttribute, TourStates


@dataclass(frozen=True)
class Window:
    __pydantic_config__ = {"extra": "forbid"}

    by: TimeAttribute
    horizon_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.horizon_s) or self.horizon_s <= 0:
            raise ValueError("window.horizon_s must be finite and positive")


@dataclass(frozen=True)
class StateInformation:
    __pydantic_config__ = {"extra": "forbid"}

    orders: bool = False
    pick_lists: bool = False
    tours: frozenset[TourStates] = frozenset()
    resources: frozenset[ResourceStates] = frozenset()
    window: Window | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tours", frozenset(TourStates(s) for s in self.tours))
        object.__setattr__(self, "resources", frozenset(ResourceStates(s)
                                                        for s in self.resources))


@dataclass(frozen=True)
class RevisionScope:
    __pydantic_config__ = {"extra": "forbid"}

    freeze_from: TourStates
    frozen_accepts: Accepts

    def __post_init__(self) -> None:
        object.__setattr__(self, "freeze_from", TourStates(self.freeze_from))
        object.__setattr__(self, "frozen_accepts", Accepts(self.frozen_accepts))
        if self.frozen_accepts is Accepts.REBUILD:
            raise ValueError("scope.frozen_accepts must be none or insert")

    def accepts(self, status: TourStates) -> Accepts:
        """Return the revision level this scope gives a tour status."""
        if TOUR_ORDER.index(status) < TOUR_ORDER.index(self.freeze_from):
            return Accepts.REBUILD
        return self.frozen_accepts
