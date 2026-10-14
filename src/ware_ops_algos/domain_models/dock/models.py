"""Warehouse dock facts shared by planning and simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dock:
    """Capacity and pallet yield of the warehouse dock."""

    capacity: int | None = None
    pallets_per_tour: int = 1
